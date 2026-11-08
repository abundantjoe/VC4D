"""The 2D papyrus fibers of a flattened wrap.

From a rendered layer stack (layers along the normal, image rows/cols on the flattened sheet):

  1. find which layers are recto (fibers across the scroll axis) and verso (fibers along it) with
     ``metrics.fiber_profile``, and average a few layers around each into a composite image;
  2. fiber direction theta and coherence in [0, 1] (``orientation_field``: Hessian at the fiber
     scale, measured accuracy in its docstring);
  3. Hessian line filter at the fiber scale ``sigma_f``: bright-ridge strength = max(0, -l_min);
  4. fiber centrelines: ridge pixels that are local maxima across the fiber direction (non-maximum
     suppression) above a contrast threshold.

A face is marked ``reliable`` only if its layer shows the right fiber signature (R > 0.1 for recto,
R < -0.1 for verso) away from the edge of the stack; otherwise widen the stack.

Angles are measured in the image plane relative to the scroll-axis direction ``axis`` (d_row, d_col):
0 deg = fiber parallel to the axis (verso-like), 90 deg = across it (recto-like).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .metrics import fiber_profile


def _fill(img):
    ok = np.isfinite(img)
    if ok.all():
        return img, ok
    return np.where(ok, img, np.nanmean(img) if ok.any() else 0.0), ok


def orientation_field(img, fiber_width=2.0, sigma_i=None, axis=(1.0, 0.0), method="hessian", sigma_g=0.7):
    """Per-pixel fiber direction, measured against ground truth (line-integral-convolution fibers,
    wavy +-25 deg, acquisition noise; median / p95 angular error):

        fiber scale sigma   0.7 vx        2.0 vx        3.5 vx
        hessian (default)   0.94 / 2.7    1.97 / 5.6    3.40 / 10.3 deg
        structure tensor    1.02 / 3.1    2.13 / 6.2    3.60 / 11.0 deg

    method="hessian": axis of largest |curvature| of the Hessian at the fiber scale (across the
    fiber for bright fibers and dark gaps alike), averaged as doubled angles weighted by anisotropy.
    method="tensor": classic structure tensor (gradient scale sigma_g).
    Returns theta (deg from the scroll axis: 0 = along, 90 = across), coherence in [0,1], fiber
    direction field (dx, dy).
    """
    f, ok = _fill(np.asarray(img, float))
    si = sigma_i if sigma_i is not None else (6.0 if fiber_width < 1.0 else 10.0)
    if method == "hessian":
        sh = max(1.05, fiber_width)
        hyy = gaussian_filter(f, sh, order=(2, 0))
        hxx = gaussian_filter(f, sh, order=(0, 2))
        hxy = gaussian_filter(f, sh, order=(1, 1))
        tr = hxx + hyy
        disc = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
        l1, l2 = 0.5 * (tr + disc), 0.5 * (tr - disc)
        ang2 = np.arctan2(2 * hxy, hxx - hyy)
        ang2 = np.where(np.abs(l2) > np.abs(l1), ang2 + np.pi, ang2)  # across-fiber axis, doubled angle
        c = gaussian_filter(disc * np.cos(ang2), si)
        s_ = gaussian_filter(disc * np.sin(ang2), si)
        mag = gaussian_filter(disc, si)
        with np.errstate(invalid="ignore", divide="ignore"):
            coh = np.where(mag > 0, np.hypot(c, s_) / mag, 0.0)
        fang = 0.5 * np.arctan2(s_, c) + np.pi / 2
    else:
        gy = gaussian_filter(f, sigma_g, order=(1, 0))
        gx = gaussian_filter(f, sigma_g, order=(0, 1))
        Jyy = gaussian_filter(gy * gy, si)
        Jxx = gaussian_filter(gx * gx, si)
        Jxy = gaussian_filter(gx * gy, si)
        trJ = Jxx + Jyy
        dJ = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            coh = np.where(trJ > 0, (dJ / trJ) ** 2, 0.0)
        fang = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2
    fdir = np.stack([np.cos(fang), np.sin(fang)], -1)  # (dx, dy) of the fiber
    a = np.array([axis[1], axis[0]], float)
    cosang = np.abs(fdir @ a) / np.linalg.norm(a)
    theta = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
    theta[~ok] = np.nan
    coh = np.asarray(coh, float)
    coh[~ok] = np.nan
    return theta, coh, fdir


def ridge_filter(img, sigma_f=1.5):
    """Bright-line strength (scale-normalised Hessian): max(0, -lambda_min)."""
    f, ok = _fill(np.asarray(img, float))
    hyy = gaussian_filter(f, sigma_f, order=(2, 0))
    hxx = gaussian_filter(f, sigma_f, order=(0, 2))
    hxy = gaussian_filter(f, sigma_f, order=(1, 1))
    tr = hxx + hyy
    disc = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
    lmin = 0.5 * (tr - disc)
    r = np.maximum(0.0, -lmin) * sigma_f ** 2
    r[~ok] = np.nan
    return r


def centerlines(ridge, fdir, rel_thresh=0.25):
    """Non-maximum suppression across the fiber direction; returns a boolean centreline mask."""
    r = np.nan_to_num(ridge)
    H, W = r.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    nx, ny = -fdir[..., 1], fdir[..., 0]  # across-fiber direction
    a = map_coordinates(r, [yy + ny, xx + nx], order=1, mode="nearest")
    b = map_coordinates(r, [yy - ny, xx - nx], order=1, mode="nearest")
    thr = rel_thresh * np.percentile(r[r > 0], 99) if (r > 0).any() else np.inf
    return (r >= a) & (r >= b) & (r > thr)


def fiber_maps(stack, step=1.0, axis=(1.0, 0.0), half_width=1, sigma_f=1.5, fiber_width=2.0) -> dict:
    """Recto/verso composites and their fiber orientation, coherence, ridge and centreline maps."""
    prof = fiber_profile(stack, step, tile=None, axis=axis)
    n = len(stack)
    offs = (np.arange(n) - 0.5 * (n - 1)) * step
    out = {"profile": prof}
    for face in ("recto", "verso"):
        key = f"{face}_offset_vx"
        if key not in prof:
            continue
        i = int(np.argmin(np.abs(offs - prof[key])))
        lay = stack[max(i - half_width, 0):i + half_width + 1]
        comp = np.nanmean(lay, 0)
        theta, coh, fdir = orientation_field(comp, fiber_width=fiber_width, axis=axis)
        rid = ridge_filter(comp, sigma_f)
        cl = centerlines(rid, fdir)
        ok = np.isfinite(theta)
        R = np.asarray([np.nan if v is None else v for v in prof["R"]], float)
        # a face is only trusted if its fiber signature has the right sign and is not at the stack edge
        signed = R[i] if face == "recto" else -R[i]
        reliable = bool(signed > 0.1 and 0 < i < n - 1)
        out[face] = {"layer": i, "offset_vx": float(offs[i]), "reliable": reliable, "signature": float(signed),
                     "image": comp, "theta": theta, "coherence": coh,
                     "ridge": rid, "centerlines": cl,
                     "median_theta_deg": float(np.nanmedian(theta)) if ok.any() else None,
                     "frac_coherent": float(np.nanmean(coh > 0.3)) if ok.any() else None,
                     "centerline_density": float(cl[ok].mean()) if ok.any() else None}
    return out
