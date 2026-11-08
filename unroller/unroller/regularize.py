"""Clean-up of traced surfaces before flattening and rendering: folds, along-normal jitter, pinholes.

A traced grid jitters along the sheet normal (each vertex is snapped on its own), has pinholes where a
snap failed, and in places folds back over sheet it has already covered (the flap is a mirrored copy
in the unrolled image, with normals pointing the wrong way).  All three are fixed without leaving the sheet:

remove_folds axis-facing grid components that duplicate outward-facing sheet (the same sheet covered
             twice) are cut, with their crease; ``meta["folds_removed"]`` lists them.
regularize   per vertex, the sheet-level density mode along the *smoothed* normal (the tracer's
             mean-shift snap, so recto and verso of one sheet form one mode) gives the height t of the
             sheet centre above the smoothed surface; t is smoothed robustly (weighted median, then the
             mean of the values near it) over grid neighbours reachable without crossing a hole or an
             out-of-plane step > ``jump`` (a tear edge or a wrap jump), and each vertex moves along the
             smoothed normal to it (at most ``max_shift``).  Tangential positions are kept.  The smoothed
             surface and its normals do not cross holes or jumps either (cut_normals).
fill_holes   interior holes of at most ``max_size`` vertices are interpolated from their rim (discrete
             Laplace), then snapped like the tracer.  A hole stays a hole when its rim is not one sheet
             (out-of-plane spread > ``jump``) or not on its CT sheet centre, when the snap moves the
             interpolated vertex by more than ``max_snap_shift`` (interpolation and CT disagree), when the
             papyrus is absent (a tear), or when it is a cut flap.  Filled vertices are listed in
             ``meta["filled"]`` (Surface.filled_mask) and are never traced evidence.
"""
from __future__ import annotations

import numpy as np

from .surface import Surface

SURFACE_SIGMA_VX = 4.0  # reference surface of regularize / fill_holes / refine (vx), chosen on phantoms (README)


def snap_params(s: Surface, vol, normals=None, level: int = 0) -> dict:
    """The tracer's snap settings for ``s``: from its meta (unroller.trace) or measured like trace() does."""
    m = s.meta
    if all(k in m for k in ("snap_radius", "sheet_sigma")):
        return {"radius": float(m["snap_radius"]), "sigma": float(m["sheet_sigma"]), "clip": m.get("clip_density"),
                "max_width": m.get("max_width")}
    from .trace import estimate_spacing

    n = s.normals(SURFACE_SIGMA_VX) if normals is None else normals
    rr, cc = np.nonzero(s.valid & np.isfinite(n[..., 0]))
    pick = np.linspace(0, len(rr) - 1, min(len(rr), 15)).astype(int)
    sp = np.nanmedian([estimate_spacing(vol, s.points[rr[i], cc[i]], n[rr[i], cc[i]], level=level) for i in pick])
    return {"radius": float(np.clip(0.35 * sp, 1.0, 5.0)) if np.isfinite(sp) else 2.0,
            "sigma": float(np.clip(sp / 4, 1.0, 4.0)) if np.isfinite(sp) else 1.5, "clip": None, "max_width": None}


def _contrast(vol, p, n, sigma, reach, level=0):
    """Kernel-weighted density at p (gaussian ``sigma`` along n) minus the darkest point within +-reach."""
    offs = np.arange(-reach, reach + 1e-9, 0.5)
    prof = np.nan_to_num(vol.sample_layers(p, n, offs, level=level) if hasattr(vol, "sample_layers") else
                         np.stack([vol.sample(p + o * n, level=level) for o in offs]))
    g = np.exp(-0.5 * (offs / sigma) ** 2)[:, None]
    return (prof * g).sum(0) / g.sum() - prof.min(0)


def _jumps(pts, nrm, jump):
    """Grid edges (u: (H, W-1), v: (H-1, W)) whose out-of-plane step exceeds ``jump`` voxels."""
    def step(a, b, na, nb):
        d = np.abs(((b - a) * 0.5 * (na + nb)).sum(-1))
        return np.nan_to_num(d, nan=0.0) > jump
    return (step(pts[:, :-1], pts[:, 1:], nrm[:, :-1], nrm[:, 1:]),
            step(pts[:-1], pts[1:], nrm[:-1], nrm[1:]))


def _window(s: Surface, sigma_vx):
    sr, sc = max(sigma_vx * s.scale[1], 0.5), max(sigma_vx * s.scale[0], 0.5)
    return int(np.ceil(2 * sr)), int(np.ceil(2 * sc)), float(sr), float(sc)


def _robust_smooth(s: Surface, t, jump_u, jump_v, sigma_vx, tol):
    from . import _kernels as K

    rr, rc, sr, sc = _window(s, sigma_vx)
    out = np.empty(t.shape)
    K.call(K.robust_smooth, t.size * 4 * rr * rc + 1, np.ascontiguousarray(t, np.float64), s.valid, jump_u, jump_v,
           rr, rc, sr, sc, float(tol), out)
    return out


def cut_normals(s: Surface, jump_u, jump_v, sigma_vx: float = SURFACE_SIGMA_VX):
    """(smoothed points, unit normals) like Surface.smoothed_points / normals(sigma_vx), but the smoothing never
    crosses a hole or a flagged grid edge (a tear or a wrap jump) and the differences are one-sided there, so the
    normals next to a jump are those of the vertex's own side, not of the ramp a plain gaussian draws across it."""
    from . import _kernels as K

    ok = s.valid
    rr, rc, sr, sc = _window(s, sigma_vx)
    ps = np.empty(s.points.shape)
    K.call(K.reach_mean, ok.size * 4 * rr * rc + 1, s.points.astype(np.float64), ok, jump_u, jump_v, rr, rc, sr, sc, ps)

    def deriv(axis, cut):
        f = np.diff(ps, axis=axis)
        f[cut] = np.nan
        pad = [(0, 0)] * 3
        pad[axis] = (0, 1)
        fwd = np.pad(f, pad, constant_values=np.nan)
        pad[axis] = (1, 0)
        bwd = np.pad(f, pad, constant_values=np.nan)
        cen = (fwd + bwd) / 2
        return np.where(np.isfinite(cen), cen, np.where(np.isfinite(fwd), fwd, bwd))

    n = np.cross(deriv(1, jump_u), deriv(0, jump_v))
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n[~ok] = np.nan
    return ps, n


def regularize(s: Surface, vol, sigma_vx: float = 6.0, normal_sigma: float = SURFACE_SIGMA_VX, tol: float = 1.0,
               jump: float = 3.0, max_shift: float | None = None, level: int = 0, params: dict | None = None):
    """Move every vertex along its smoothed normal onto the robustly smoothed sheet-centre height (see module doc).

    ``sigma_vx`` = smoothing scale of the sheet-centre height (voxels along the sheet), ``tol`` = how far (vx) a
    neighbour's estimate may lie from the local median and still count, ``jump`` = out-of-plane step (vx) between
    grid neighbours treated as a discontinuity, ``max_shift`` = largest move (default: the snap radius).
    Returns (surface, report)."""
    from .snap_fast import snap_ms

    pp = s.points.astype(np.float64)
    ju, jv = _jumps(pp, s.normals(normal_sigma).astype(np.float64), jump)
    ps, ns = cut_normals(s, ju, jv, normal_sigma)
    ok = s.valid & np.isfinite(ns[..., 0])
    h = np.where(ok, ((pp - ps) * ns).sum(-1), np.nan)
    sp = params or snap_params(s, vol, ns, level)
    p, n = pp[ok], ns[ok]
    q, good = snap_ms(vol, p, n, sp["radius"], sp["sigma"], min_contrast=-np.inf, level=level, clip=sp["clip"],
                      max_width=sp["max_width"])
    t = np.full(s.shape, np.nan)
    t[ok] = np.where(good, h[ok] + ((q - p) * n).sum(1), np.nan)
    ts = _robust_smooth(s, t, ju, jv, sigma_vx, tol)
    lim = sp["radius"] if max_shift is None else max_shift
    d = np.clip(np.nan_to_num(ts - h, nan=0.0), -lim, lim)
    d[~ok] = 0.0
    out = s.points + (d[..., None] * np.nan_to_num(ns)).astype(np.float32)
    out[~s.valid] = np.nan
    rep = {"moved_vertices": int((np.abs(d) > 0).sum()), "shift_p50": float(np.median(np.abs(d[ok]))),
           "shift_p95": float(np.percentile(np.abs(d[ok]), 95)), "no_estimate_fraction": float(np.isnan(t[ok]).mean()),
           "jump_edges": int(ju.sum() + jv.sum()), "params": {"sigma_vx": sigma_vx, "normal_sigma": normal_sigma,
                                                             "tol": tol, "jump": jump, "max_shift": lim, **sp}}
    return Surface(out, s.scale, s.meta), rep


def _flaps(s: Surface, up, umbilicus, normal_sigma, radius, min_dup, min_grid_dist, min_cos):
    """One pass of remove_folds: (flap mask incl. crease, number of flap components, axis-facing vertices)."""
    from scipy.ndimage import binary_dilation, label
    from scipy.spatial import cKDTree

    ns = s.normals(normal_sigma).astype(np.float64)
    pts = s.points.astype(np.float64)
    rad = pts - umbilicus
    rad -= (rad @ up)[..., None] * up
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (ns * rad).sum(-1) / np.linalg.norm(rad, axis=-1)
    ok = s.valid & np.isfinite(cos)
    inward, outward = ok & (cos < -min_cos), ok & (cos > min_cos)
    none = np.zeros(s.shape, bool)
    if not inward.any() or not outward.any():
        return none, 0, int(inward.sum())
    ro, co = np.nonzero(outward)
    ri, ci = np.nonzero(inward)
    d, j = cKDTree(pts[ro, co]).query(pts[ri, ci], k=8, distance_upper_bound=radius)
    j = np.minimum(j, len(ro) - 1)  # misses (d = inf) point past the end
    far = np.maximum(np.abs(ro[j] - ri[:, None]), np.abs(co[j] - ci[:, None]))
    dup = np.zeros(s.shape, bool)
    dup[ri, ci] = (np.isfinite(d) & (far >= min_grid_dist)).any(1)
    lab, k = label(ok & (cos < 0), np.ones((3, 3), bool))
    n_in = np.bincount(lab[inward], minlength=k + 1)
    n_dup = np.bincount(lab[dup], minlength=k + 1)
    flap = np.nonzero((n_dup >= min_dup * n_in) & (n_dup >= 3))[0]
    flap = flap[flap > 0]
    if not len(flap):
        return none, 0, int(inward.sum())
    return binary_dilation(np.isin(lab, flap), np.ones((3, 3), bool)) & s.valid, len(flap), int(inward.sum())


def remove_folds(s: Surface, normal_sigma: float = 8.0, radius: float | None = None, min_dup: float = 0.5,
                 min_grid_dist: int = 4, min_cos: float = 0.3, passes: int = 1):
    """Cut flaps where the grid folded back over sheet it already covers (see module doc).  Returns (surface, report).

    Orientation: the smoothed normal (``normal_sigma`` vx) against the direction away from the scroll axis
    (meta ``umbilicus`` / ``up``; the tracer grows its grid with normals pointing away from the axis).  A component
    of axis-facing vertices is a flap when >= ``min_dup`` of its clearly axis-facing vertices (|cos| > ``min_cos``)
    lie within ``radius`` vx (default 0.4 x the sheet spacing, in [2, 5]) of an outward-facing vertex at least
    ``min_grid_dist`` grid cells away: the same sheet, covered twice.  An axis-facing part that covers new sheet
    (a real fold of the papyrus) is kept.  ``passes`` > 1 repeats the cut until nothing is cut (the smoothed normals
    next to a cut flap no longer average over it, so a smaller flap beside it can show up); on development phantoms
    3 passes lowered reversed render normals on sectors (0.59 -> 0.41 %) but not on two-layer scrolls (0.42 -> 0.43 %),
    so one pass is the default.  Without an umbilicus the surface is returned unchanged."""
    m = s.meta
    rep = {"flap_components": 0, "removed_vertices": 0, "axis_facing_vertices": 0, "passes": 0}
    if m.get("umbilicus") is None:
        return s, {**rep, "skipped": "no umbilicus in meta"}
    up = np.asarray(m.get("up", (0.0, 0.0, 1.0)), float)
    up /= np.linalg.norm(up)
    if radius is None:
        sp = m.get("sheet_spacing")
        radius = float(np.clip(0.4 * sp, 2.0, 5.0)) if sp and np.isfinite(sp) else 3.0
    cut = np.zeros(s.shape, bool)
    cur = s
    for p in range(passes):
        c, k, n_in = _flaps(cur, up, np.asarray(m["umbilicus"], float), normal_sigma, radius, min_dup, min_grid_dist,
                            min_cos)
        if p == 0:
            rep["axis_facing_vertices"] = n_in
        if not k:
            break
        cut |= c
        rep["flap_components"] += k
        rep["passes"] = p + 1
        pts = cur.points.copy()
        pts[c] = np.nan
        cur = Surface(pts, s.scale, m)
    if not cut.any():
        return s, rep
    meta = dict(m)
    meta["folds_removed"] = {"grid": list(s.shape), "flat_index": np.flatnonzero(cut).tolist()}
    rep.update(removed_vertices=int(cut.sum()), radius=radius)
    return Surface(cur.points, s.scale, meta), rep


def interior_holes(valid: np.ndarray):
    """Labels (H, W) of the invalid components not connected to the grid border (4-connected), and their sizes."""
    from scipy.ndimage import label

    lab, k = label(~valid)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    sizes = np.bincount(lab.ravel(), minlength=k + 1)
    sizes[border] = 0
    sizes[0] = 0
    lab[np.isin(lab, border)] = 0
    return lab, sizes


def fill_holes(s: Surface, vol, max_size: int = 4, jump: float = 3.0, normal_sigma: float = SURFACE_SIGMA_VX,
               level: int = 0, params: dict | None = None, max_snap_shift: float | None = 0.5,
               max_rim_offset: float | None = 0.5):
    """Fill interior holes of <= ``max_size`` vertices (see module doc).  Returns (surface, report).

    ``max_rim_offset`` (vx): the median distance of the hole's rim vertices from their CT sheet-level mode must not
    exceed it; ``max_snap_shift`` (vx): largest snap move of a filled vertex.  None disables a check.  The defaults
    were chosen on development phantoms by the filled vertices' distance from the true sheet centre (README)."""
    from scipy.ndimage import binary_dilation, grey_dilation

    from . import _kernels as K
    from .snap_fast import snap_ms

    lab, sizes = interior_holes(s.valid)
    fr = s.meta.get("folds_removed")  # a cut flap is not a hole: the sheet there is already covered
    if fr and tuple(fr.get("grid", ())) == tuple(s.shape):
        sizes[np.unique(lab.flat[np.asarray(fr["flat_index"], np.int64)])] = 0
    cand = np.nonzero((sizes > 0) & (sizes <= max_size))[0]
    rep = {"holes": int((sizes > 0).sum()), "hole_vertices": int(sizes.sum()), "candidates": int(len(cand)),
           "filled_holes": 0, "filled_vertices": 0, "rejected_rim": 0, "rejected_snap": 0}
    if not len(cand):
        return s, rep
    pts = s.points.astype(np.float64)
    ns = s.normals(normal_sigma).astype(np.float64)
    ps = s.smoothed_points(normal_sigma).astype(np.float64)
    fill = np.isin(lab, cand)
    # rim test: the valid vertices around a hole must be one sheet (no out-of-plane spread > jump about their plane)
    rim = binary_dilation(fill, np.ones((3, 3), bool)) & s.valid
    rl = lab.copy()
    rl[~fill] = 0
    rl = np.where(rim, grey_dilation(rl, size=(3, 3)), 0)
    sp = params or snap_params(s, vol, ns, level)
    if max_rim_offset is not None:  # the rim itself must sit on its CT sheet centre
        rr_, cc_ = np.nonzero(rl > 0)
        nr = np.nan_to_num(ns[rr_, cc_])
        qr, okr = snap_ms(vol, pts[rr_, cc_], nr, sp["radius"], sp["sigma"], min_contrast=-np.inf, level=level,
                          clip=sp["clip"], max_width=sp["max_width"])
        roff = np.full(s.shape, np.inf)
        roff[rr_, cc_] = np.where(okr, np.abs(((qr - pts[rr_, cc_]) * nr).sum(1)), np.inf)
    keep = set()
    for k in cand:
        q = pts[rl == k]
        if len(q) < 3:
            rep["rejected_rim"] += 1
            continue
        c = q - q.mean(0)
        nrm = np.linalg.svd(c, full_matrices=False)[2][-1]
        if np.ptp(c @ nrm) > jump or (max_rim_offset is not None and np.median(roff[rl == k]) > max_rim_offset):
            rep["rejected_rim"] += 1
            continue
        keep.add(int(k))
    fill = np.isin(lab, list(keep))
    if not fill.any():
        return s, rep
    P = pts.copy()
    P[fill] = ps[fill]
    P[fill & ~np.isfinite(P[..., 0])] = np.nanmean(pts[rim], 0)
    K.membrane_fill(P, fill, 500, 1e-3)
    rr, cc = np.nonzero(fill)
    nf = ns[rr, cc]
    bad = ~np.isfinite(nf[:, 0])
    nf[bad] = np.nanmean(ns[rim], 0)
    nf /= np.linalg.norm(nf, axis=1, keepdims=True)
    q, ok = snap_ms(vol, P[rr, cc], nf, sp["radius"], sp["sigma"], min_contrast=-np.inf, level=level, clip=sp["clip"],
                    max_width=sp["max_width"])
    if max_snap_shift is not None:  # interpolation and CT must agree
        ok &= np.abs(((q - P[rr, cc]) * nf).sum(1)) <= max_snap_shift
    # papyrus must be there: the same contrast the traced vertices show (30 % of their median, as trace())
    sub = max(1, s.valid.sum() // 2000)
    reach = sp["radius"] + 3 * sp["sigma"]
    ref = _contrast(vol, pts[s.valid][::sub], np.nan_to_num(ns[s.valid][::sub]), sp["sigma"], reach, level)
    ok &= _contrast(vol, q, nf, sp["sigma"], reach, level) >= 0.3 * np.median(ref)
    out = s.points.copy()
    out[rr[ok], cc[ok]] = q[ok]
    # a filled vertex must not make a jump with its neighbours, nor bridge a stretched (torn) part of the grid
    S2 = Surface(out, s.scale, s.meta)
    ju, jv = _jumps(S2.points.astype(np.float64), np.nan_to_num(S2.normals(normal_sigma).astype(np.float64)), jump)
    eu, ev = S2.edge_lengths()
    ju |= np.nan_to_num(eu, nan=0.0) > 2.0 / s.scale[0]
    jv |= np.nan_to_num(ev, nan=0.0) > 2.0 / s.scale[1]
    bad = np.zeros(s.shape, bool)
    bad[:, :-1] |= ju
    bad[:, 1:] |= ju
    bad[:-1] |= jv
    bad[1:] |= jv
    newv = np.zeros(s.shape, bool)
    newv[rr[ok], cc[ok]] = True
    drop = newv & bad
    out[drop] = np.nan
    newv &= ~drop
    rep["rejected_snap"] = int((~ok).sum() + drop.sum())
    rep["filled_vertices"] = int(newv.sum())
    rep["filled_holes"] = int(len(np.unique(lab[newv])))
    meta = dict(s.meta)
    prev = s.filled_mask() | newv
    meta["filled"] = {"grid": list(s.shape), "flat_index": np.flatnonzero(prev).tolist()}
    return Surface(out, s.scale, meta), rep
