"""Sheet-centring calibration: move each vertex along its normal onto the local density peak.

The papyrus sheet is the bright band in CT.  For every vertex we sample a profile along the
(smoothed) normal, average profiles over a Gaussian neighbourhood (so single noisy voxels cannot pull
the surface), smooth it along the normal at the sheet scale (so recto and verso form one peak, as in the
tracer's snap), locate the sub-voxel peak, smooth the resulting offset field and apply it.
Offsets are clamped to +-max_shift so the surface can never jump to the neighbouring wrap.
For traced grids regularize.regularize is the better tool: it does not smooth across holes or jumps.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .regularize import SURFACE_SIGMA_VX, snap_params
from .surface import Surface
from .volume import Volume


def _nan_gauss(a, sigma):
    ok = np.isfinite(a)
    num = gaussian_filter(np.where(ok, a, 0.0), sigma)
    den = gaussian_filter(ok.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-3, num / den, np.nan)


def _nan_gauss_layers(samples, sigma):
    """[_nan_gauss(s, sigma) for s in samples], identical values: the validity mask of a profile sample is
    the same for every offset (NaN only where the vertex or its normal is invalid), so its smoothed weight
    is computed once when the masks agree; the per-offset filters run in threads (scipy releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    oks = np.isfinite(samples)
    shared = bool(np.all(oks == oks[:1]))
    den0 = gaussian_filter(oks[0].astype(float), sigma) if shared else None

    def one(k):
        ok = oks[k]
        num = gaussian_filter(np.where(ok, samples[k], 0.0), sigma)
        den = den0 if shared else gaussian_filter(ok.astype(float), sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 1e-3, num / den, np.nan)
    with ThreadPoolExecutor(4) as ex:
        return np.stack(list(ex.map(one, range(len(samples)))))


def peak_offsets(s: Surface, vol: Volume, radius: float = 3.0, step: float = 0.5, sigma_vx: float = 8.0,
                 level: int = 0, normals=None, sheet_sigma: float = 0.0) -> np.ndarray:
    """Per-vertex signed offset (voxels along +normal) of the sheet density peak. NaN if no clear peak.
    ``normals`` = s.normals() if already computed.  ``sheet_sigma`` > 0 (vx): the peak of the profile smoothed
    along the normal at the sheet scale (the tracer's sheet-level mode), so the two fiber layers of one sheet
    form one peak at the sheet's middle; 0 = the raw density peak (one lamina where they are resolved)."""
    from . import _kernels as K

    n = s.normals() if normals is None else normals
    ext = np.ceil(3 * sheet_sigma / step) * step if sheet_sigma > 0 else 0.0
    offs = np.arange(-radius - ext, radius + ext + 1e-9, step)
    sig = sigma_vx * np.sqrt(s.scale[0] * s.scale[1])  # neighbourhood in grid cells
    if hasattr(vol, "sample_layers"):  # all offsets in one fused pass: each chunk decoded once
        samples = vol.sample_layers(s.points, n, offs, level=level)
    else:
        samples = np.stack([vol.sample(s.points + o * n, level=level) for o in offs])
    prof = _nan_gauss_layers(samples, sig)
    if sheet_sigma > 0:
        k = int(round(ext / step))
        prof = gaussian_filter1d(prof, sheet_sigma / step, axis=0, mode="nearest")[k:len(offs) - k]
        offs = offs[k:len(offs) - k]
    out = np.empty(s.shape, np.float64)
    K.subvoxel_peak(prof.reshape(len(offs), -1), offs, float(step), s.valid.reshape(-1), out.reshape(-1))
    return out


def refine(s: Surface, vol: Volume, iterations: int = 3, radius: float = 3.0, sigma_vx: float = 8.0,
           max_shift: float = 3.0, level: int = 0, log=None, sheet_sigma: float | None = None,
           normal_sigma: float = SURFACE_SIGMA_VX) -> tuple[Surface, dict]:
    """Move every vertex along its (smoothed) normal onto the sheet-level density peak; see the module doc.

    ``sheet_sigma`` None = the tracer's kernel from the segment meta, or measured from the sheet spacing
    (regularize.snap_params); 0 = the raw density peak (rides one lamina of a two-layer sheet: on two-layer
    phantoms it doubled the distance to the true sheet centre).  ``normal_sigma`` = 0 moves along raw normals."""
    if sheet_sigma is None:
        sheet_sigma = snap_params(s, vol, level=level)["sigma"]
    pts = s.points.copy()
    total = np.zeros(s.shape)
    hist = []
    for it in range(iterations):
        cur = Surface(pts, s.scale, s.meta)
        nrm = cur.normals(normal_sigma)
        d = peak_offsets(cur, vol, radius, 0.5, sigma_vx, level, normals=nrm, sheet_sigma=sheet_sigma)
        d = _nan_gauss(d, sigma_vx * np.sqrt(s.scale[0] * s.scale[1]) / 2)
        d = np.nan_to_num(d, nan=0.0)
        d = np.clip(total + d, -max_shift, max_shift) - total  # cumulative clamp
        pts = pts + d[..., None] * nrm
        pts[~s.valid] = np.nan
        total += d
        hist.append(float(np.nanmedian(np.abs(d[s.valid]))))
        if log:
            log(f"refine iter {it}: median |shift| {hist[-1]:.3f} vx")
        if hist[-1] < 0.05:
            break
    rep = {"median_abs_shift_per_iter": hist, "sheet_sigma": float(sheet_sigma), "normal_sigma": float(normal_sigma),
           "total_shift_p50": float(np.median(np.abs(total[s.valid]))),
           "total_shift_p95": float(np.percentile(np.abs(total[s.valid]), 95)),
           "clamped_fraction": float((np.abs(total[s.valid]) >= max_shift - 1e-6).mean())}
    return Surface(pts, s.scale, s.meta), rep
