"""Post-hoc per-vertex channels of a trace (PLAN section 2 S6), so ``trace.py`` stays unchanged.

``compute_channels(surface, base_vol, ...)`` returns the ``TraceRec.channels`` dict, one (H, W) array per
channel on the trace grid (dtypes ``types.CHANNEL_DTYPES``; NaN / 0 at invalid vertices):

  H          the band field W at the vertex (NaN without a band)
  width      half-prominence width (vx) of the bright run at the vertex, along the surface normal: the
             threshold is peak - (peak - max(left min, right min)) / 2 (PLAN S1 item 1, not half-max against
             the air minimum); NaN where the run is not closed inside the profile window
  contrast   the prominence peak - max(left min, right min) (CT units)
  conflict   displacement (vx) of one unmasked ``snap_ms`` call from the vertex (PLAN: conflict = > 1.5 vx)
  trunc      1 where the band boundary cuts bright papyrus within +-4 vx along the normal
             (d3/topo_proto.py's "truncated" test: the mask changes between two samples both >= 0.6 x the
             vertex's own density)
  plane_res  RMS distance (vx) of the vertex's (2r+1)^2 grid window to its least-squares plane

All profiles are sampled unmasked from ``base_vol`` (``sample_layers``).
"""
from __future__ import annotations

import numpy as np

from .types import CHANNEL_DTYPES


def half_prominence(prof: np.ndarray, step: float, centre_reach: float = 1.5):
    """Per row of ``prof`` (N, K) sampled at offsets (k - K // 2) * step: the peak within +-centre_reach of
    the centre and its half-prominence run.  Returns (peak_offset, width, contrast), each (N,) float64;
    width is NaN where the run reaches the end of the profile or the peak has no prominence."""
    prof = np.asarray(prof, np.float64)
    N, K = prof.shape
    out = (np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan))
    if N == 0 or K < 3:
        return out
    c0 = K // 2
    r = int(round(centre_reach / step))
    lo, hi = max(c0 - r, 0), min(c0 + r + 1, K)
    pf = np.where(np.isfinite(prof), prof, -np.inf)
    ip = lo + np.argmax(pf[:, lo:hi], 1)
    rows = np.arange(N)
    peak = pf[rows, ip]
    idx = np.arange(K)[None, :]
    lmin = np.where(idx <= ip[:, None], np.where(np.isfinite(prof), prof, np.inf), np.inf).min(1)
    rmin = np.where(idx >= ip[:, None], np.where(np.isfinite(prof), prof, np.inf), np.inf).min(1)
    base = np.maximum(lmin, rmin)
    prom = peak - base
    thr = peak - 0.5 * prom
    below = ~(pf >= thr[:, None])
    li = np.where(below & (idx < ip[:, None]), idx, -1).max(1)
    ri = np.where(below & (idx > ip[:, None]), idx, K).min(1)
    ok = (li >= 0) & (ri < K) & (prom > 0) & np.isfinite(peak)
    offs = (np.arange(K) - c0) * step
    li_, ri_ = np.clip(li, 0, K - 2), np.clip(ri, 1, K - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fl = (thr - pf[rows, li_]) / (pf[rows, li_ + 1] - pf[rows, li_])
        fr = (pf[rows, ri_ - 1] - thr) / (pf[rows, ri_ - 1] - pf[rows, ri_])
    left = offs[li_] + np.clip(np.nan_to_num(fl), 0, 1) * step
    right = offs[ri_ - 1] + np.clip(np.nan_to_num(fr), 0, 1) * step
    peak_off = offs[ip]
    width = np.where(ok, right - left, np.nan)
    contrast = np.where(np.isfinite(peak) & np.isfinite(base), prom, np.nan)
    return peak_off.astype(np.float64), width, contrast


def plane_res(surface, radius: int = 2, min_count: int = 4) -> np.ndarray:
    """(H, W) float32: RMS distance (vx) of the valid vertices in each vertex's (2r+1)^2 grid window to their
    least-squares plane (sqrt of the smallest covariance eigenvalue); NaN at invalid vertices or with fewer
    than ``min_count`` valid neighbours."""
    P = np.asarray(surface.points, np.float64)
    H, W = P.shape[:2]
    ok = np.all(np.isfinite(P), 2)
    out = np.full((H, W), np.nan, np.float32)
    if not ok.any():
        return out
    c = P[ok].mean(0)
    Q = np.where(ok[..., None], P - c, 0.0)
    pad = radius
    Qp = np.pad(Q, ((pad, pad), (pad, pad), (0, 0)))
    Op = np.pad(ok.astype(np.float64), pad)
    n = np.zeros((H, W))
    s1 = np.zeros((H, W, 3))
    s2 = np.zeros((H, W, 3, 3))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            q = Qp[pad + dy:pad + dy + H, pad + dx:pad + dx + W]
            o = Op[pad + dy:pad + dy + H, pad + dx:pad + dx + W]
            n += o
            s1 += q
            s2 += q[..., :, None] * q[..., None, :]
    good = ok & (n >= min_count)
    nn = n[good][:, None]
    mu = s1[good] / nn
    cov = s2[good] / nn[..., None] - mu[:, :, None] * mu[:, None, :]
    lam = np.linalg.eigvalsh(cov)[:, 0]
    out[good] = np.sqrt(np.maximum(lam, 0.0)).astype(np.float32)
    return out


def snap_conflict(vol, pts, normals, snap_radius: float, sheet_sigma: float, level: int = 0) -> np.ndarray:
    """Displacement (vx) of one unmasked sheet-level mean-shift snap (``snap_fast.snap_ms``, no contrast gate)."""
    from ..snap_fast import snap_ms

    pts = np.asarray(pts, np.float64)
    if len(pts) == 0:
        return np.zeros(0)
    q = snap_ms(vol, pts, np.asarray(normals, np.float64), snap_radius, sheet_sigma, min_contrast=-np.inf,
                level=level)[0]
    return np.linalg.norm(q - pts, axis=1)


def band_trunc(mask_fn, vol, pts, normals, reach: float = 4.0, step: float = 0.5, bright_rel: float = 0.6,
               level: int = 0) -> np.ndarray:
    """True where the band mask changes between two consecutive samples along the normal within +-reach whose
    densities are both >= bright_rel x the density at the vertex (the band boundary cuts bright papyrus)."""
    pts = np.asarray(pts, np.float64)
    nrm = np.asarray(normals, np.float64)
    if len(pts) == 0:
        return np.zeros(0, bool)
    oo = np.arange(-reach, reach + 1e-9, step)
    q = pts[:, None, :] + oo[None, :, None] * nrm[:, None, :]
    mm = np.asarray(mask_fn(q.reshape(-1, 3)), bool).reshape(len(pts), len(oo))
    vv = np.asarray(vol.sample_layers(pts, nrm, oo, level=level), np.float64).T
    cen = vv[:, len(oo) // 2]
    tr = (mm[:, 1:] != mm[:, :-1]) & (np.minimum(vv[:, 1:], vv[:, :-1]) >= bright_rel * cen[:, None])
    return tr.any(1)


def compute_channels(surface, base_vol, band=None, mask_fn=None, snap_radius: float | None = None,
                     sheet_sigma: float | None = None, half_window: float = 6.0, step: float = 0.5,
                     plane_radius: int = 2, level: int = 0) -> dict:
    """All channels of one trace (see the module docstring).  ``band``: a ``BandQuery`` for H; ``mask_fn``:
    the trace's band mask (``BandVolume.mask``) for trunc; ``snap_radius`` / ``sheet_sigma``: the trace's
    own (``trace().meta``) for conflict.  Channels whose input is missing are NaN (trunc: 0)."""
    shp = surface.shape
    ok = surface.valid
    pts = surface.points[ok].astype(np.float64)
    nrm = np.nan_to_num(surface.normals()[ok].astype(np.float64))
    full = {k: np.full(shp, np.nan, np.float64) for k in CHANNEL_DTYPES}
    full["trunc"] = np.zeros(shp, np.float64)
    if len(pts):
        if band is not None:
            full["H"][ok] = np.asarray(band.W(pts)[0], np.float64)
        offs = np.arange(-half_window, half_window + 1e-9, step)
        prof = np.asarray(base_vol.sample_layers(pts, nrm, offs, level=level), np.float64).T
        _, width, contrast = half_prominence(prof, step)
        full["width"][ok] = width
        full["contrast"][ok] = contrast
        if snap_radius is not None and sheet_sigma is not None:
            full["conflict"][ok] = snap_conflict(base_vol, pts, nrm, snap_radius, sheet_sigma, level)
        if mask_fn is not None:
            full["trunc"][ok] = band_trunc(mask_fn, base_vol, pts, nrm, level=level)
    full["plane_res"] = plane_res(surface, plane_radius).astype(np.float64)
    return {k: v.astype(CHANNEL_DTYPES[k]) for k, v in full.items()}
