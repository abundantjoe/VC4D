"""Compiled drop-in for trace._snap_ms (sheet-level mean-shift snap along the normal).

Same signature, same returns and the same arithmetic: the normal profiles are sampled with the fused
layer kernel (Volume.sample_layers, bit-identical to sampling p + o * n), every row sum reproduces
numpy's pairwise summation order and exp() is numpy's own, so positions, ok flags and widths are
bit-identical to trace._snap_ms (tests/test_perf_lean.py checks this on phantoms and on real Scroll 1).
What is removed is the per-operation numpy overhead: ~15 temporaries of N x K per mean-shift iteration.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import _kernels as K


def _profiles(vol, p, n, offs, level):
    if hasattr(vol, "sample_layers"):
        return vol.sample_layers(p, n, offs, level=level).T
    pts = p[:, None, :] + offs[None, :, None] * n[:, None, :]
    return vol.sample(pts.reshape(-1, 3), level=level).reshape(len(p), len(offs))


def basin_bounds(prof, step, prom, smooth=1.0):
    """Index range [lo, hi] of the basin around each profile's centre sample (index K // 2): it ends at
    the nearest dark minimum on each side (on the profile smoothed with ``smooth`` vx) lying >= prom
    below the highest point between it and the centre and >= prom / 2 below the highest point beyond
    it; else at the profile end.  Mean-shift confined to [lo, hi] cannot cross a gap between sheets."""
    Kn = prof.shape[1]
    c0 = Kn // 2
    f = gaussian_filter1d(prof, smooth / step, axis=1) if smooth > 0 else prof
    is_min = np.zeros_like(f, bool)
    is_min[:, 1:-1] = (f[:, 1:-1] < f[:, :-2]) & (f[:, 1:-1] <= f[:, 2:])
    lmax = np.maximum.accumulate(f, axis=1)
    rmax = np.maximum.accumulate(f[:, ::-1], axis=1)[:, ::-1]
    idx = np.arange(Kn)[None, :]
    pbl = np.full_like(f, -np.inf)
    pbl[:, :c0 + 1] = np.maximum.accumulate(f[:, c0::-1], axis=1)[:, ::-1]  # max f[k .. c0]
    pbr = np.full_like(f, -np.inf)
    pbr[:, c0:] = np.maximum.accumulate(f[:, c0:], axis=1)  # max f[c0 .. k]
    condl = is_min & (idx < c0) & (pbl - f >= prom) & (lmax - f >= 0.5 * prom)
    condr = is_min & (idx > c0) & (pbr - f >= prom) & (rmax - f >= 0.5 * prom)
    lo = np.where(condl, idx, -1).max(1)
    lo[lo < 0] = 0
    hi = np.where(condr, idx, Kn).min(1)
    hi[hi >= Kn] = Kn - 1
    return lo, hi


def snap_ms(vol, p, n, radius, sigma, min_contrast=10.0, level=0, iters=8, step=0.5, clip=None,
            max_width=None, return_width=False, tol=0.05, basin=None, basin_smooth=1.0, strict=True):
    """Exactly trace._snap_ms (see its docstring), compiled."""
    p = np.asarray(p)
    n = np.asarray(n)
    reach = radius + 3 * sigma
    offs = np.arange(-reach, reach + 1e-9, step)
    prof = np.ascontiguousarray(_profiles(vol, p, n, offs, level))
    N, Kn = prof.shape
    g = np.empty((N, Kn))
    sigma_f, radius_f = float(sigma), float(radius)
    if np.result_type(prof, clip if clip is not None else prof.dtype.type(0)) == prof.dtype:
        fi = np.finfo(prof.dtype)
        cv = prof.dtype.type(clip if clip is not None else 0)  # numpy's weak-scalar cast of clip
        K.ms_prepare(prof, cv, clip is not None, fi.min, fi.max, offs, sigma_f, g)
    else:  # clip of a wider dtype promotes the profiles: let numpy do exactly that
        prof = np.nan_to_num(prof)
        prof = np.minimum(prof, clip)
        prof = np.ascontiguousarray(prof - prof.min(1, keepdims=True))
        K.ms_gauss_arg(offs, np.zeros(N), sigma_f, g)
    lo = np.full(N, -radius_f)
    hi = np.full(N, radius_f)
    wprof = prof
    if basin is not None:
        bl, bh = basin_bounds(prof, step, basin, basin_smooth)
        idx = np.arange(Kn)[None, :]
        wprof = np.ascontiguousarray(np.where((idx >= bl[:, None]) & (idx <= bh[:, None]), prof, 0.0))
        lo = np.maximum(lo, offs[bl])  # == clipping to +-radius, then to the basin (they always overlap)
        hi = np.minimum(hi, offs[bh])
    d = np.zeros(N)
    dn = np.empty(N)
    tmp = np.empty((N, Kn))
    # identical results either way; threads only for large batches (a launch costs ms when cores are busy)
    shift = K.ms_shift_par if N * Kn >= K.SMALL_WORK else K.ms_shift
    for _ in range(iters):
        np.exp(g, out=g)
        done = shift(g, wprof, offs, d, sigma_f, lo, hi, dn, tmp) < tol
        d, dn = dn, d
        if done:
            break
    np.exp(g, out=g)  # kernel weights at the final d
    want = max_width is not None or return_width
    ok = np.empty(N, bool)
    width = np.empty(N)
    K.ms_finish(g, prof, d, radius_f, float(min_contrast), float(reach), float(step), prof.dtype.type(0.5),
                float(max_width) if max_width is not None else 0.0, max_width is not None, want, bool(strict),
                ok, width, tmp)
    if return_width:
        return p + d[:, None] * n, ok, width
    return p + d[:, None] * n, ok
