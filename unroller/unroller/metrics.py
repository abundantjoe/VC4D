"""Quantitative checks.  Every number here is something you can threshold in CI or a review.

surface_report   - grid health: valid fraction, spacing error vs nominal, normal flips (folds)
flatten_report   - distortion of the UV map: per-triangle singular values of the 3D->2D Jacobian
stack_report     - sheet centring: where the mean density peak sits in the rendered layer stack
procrustes_rms   - UV vs. a known ground-truth parameterisation (phantoms / annotated points)
ncc              - normalised cross-correlation between a render and a reference image
equivariance_switch - do two traces that should coincide lie on the same sheet (foot-point matching)
offsheet_vs_labels  - off-sheet rate of a trace vs human sheet labels at matched generation
"""
from __future__ import annotations

import warnings

import numpy as np

from .flatten import _clean, _grid_folds, triangle_singular_values
from .surface import Surface


def _pct(a, ps=(50, 95, 99)):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def surface_report(s: Surface, voxel_size_um: float | None = None) -> dict:
    eu, ev = s.edge_lengths()
    nom_u, nom_v = 1 / s.scale[0], 1 / s.scale[1]
    rel = np.concatenate([(eu / nom_u - 1).ravel(), (ev / nom_v - 1).ravel()])
    n = s.normals()
    flips = []
    for ax in (0, 1):
        a = np.take(n, range(n.shape[ax] - 1), axis=ax)
        b = np.take(n, range(1, n.shape[ax]), axis=ax)
        d = (a * b).sum(-1)
        flips.append(d[np.isfinite(d)] < 0)
    flips = np.concatenate(flips)
    area = s.area()
    r = {"grid": list(s.shape), "scale": list(s.scale), "valid_fraction": float(s.valid.mean()),
         "area_vx2": area, "spacing_rel_error": _pct(np.abs(rel)),
         "normal_flip_fraction": float(flips.mean()) if flips.size else None}
    if voxel_size_um:
        r["area_cm2"] = area * (voxel_size_um * 1e-4) ** 2
    return r


def flatten_report(s: Surface, uv_grid: np.ndarray) -> dict:
    """Distortion of a flattening.  An ideal unrolling has s1 = s2 = 1 for every triangle."""
    idx, V, T = s.triangles()
    T = _clean(V, T)
    uv = uv_grid[idx >= 0]
    ok = np.all(np.isfinite(uv[T]).reshape(len(T), -1), axis=1)
    mirrored = _grid_folds(s, idx, V, T[ok])
    sv, det = triangle_singular_values(V, T[ok], uv, mirrored)
    det_raw = np.where(mirrored, -det, det)  # orientation in the triangles' own (unmirrored) 3D frames
    s1, s2 = sv[:, 0], sv[:, 1]
    sign = np.sign(np.median(det))
    a3 = 0.5 * np.linalg.norm(np.cross(V[T[ok, 1]] - V[T[ok, 0]], V[T[ok, 2]] - V[T[ok, 0]]), axis=1)
    w = a3 / a3.sum()
    return {
        "triangles": int(ok.sum()),
        # orientation of a valid map is globally consistent; count triangles disagreeing with it.  Grid-folded
        # triangles (see flatten._grid_folds) are judged in their mirrored frames, where the true unrolling
        # keeps them upright; they are also counted in their own frames (*_own_frame), so a flip that the
        # mirroring absorbed stays visible (it is only correct if the fold really is a grid crossover).
        "flipped_triangles": int((np.sign(det) != sign).sum()),
        "mirrored_triangles": int(mirrored.sum()),
        "mirrored_flipped": int((np.sign(det[mirrored]) != sign).sum()),
        "mirrored_flipped_own_frame": int((np.sign(det_raw[mirrored]) != sign).sum()),
        "flipped_triangles_own_frame": int((np.sign(det_raw) != sign).sum()),
        "area_ratio_2d_3d": float((w * s1 * s2).sum()),
        "abs_log_area_distortion": _pct(np.abs(np.log(s1 * s2))),
        "anisotropy_s1_over_s2": _pct(s1 / s2),
        "max_stretch_s1": _pct(s1),
        "min_stretch_s2": _pct(s2, ps=(1, 5, 50)),
        "l2_stretch": float(np.sqrt((w * (s1**2 + s2**2) / 2).sum())),  # Sander et al.; 1 = isometric
    }


def stack_report(stack: np.ndarray, step: float = 1.0) -> dict:
    """Mean intensity per layer.  For a well placed surface on a bright sheet the peak is central.

    ``peak_offset_vx`` is the sub-layer (parabolic) peak position relative to the centre layer,
    in voxels along +normal.  |offset| > ~1 voxel means the surface drifts off the sheet.
    A stack with no finite sample (a degenerate surface: nothing rendered) has no peak: ``peak_layer`` and
    ``peak_offset_vx`` are None, so no caller can take it for a centred surface.
    """
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nanmean of an all-NaN layer: NaN, which is the answer
        prof = np.array([np.nanmean(l) for l in stack])
    n = len(prof)
    if not np.isfinite(prof).any():  # nanargmax would raise on an all-NaN profile
        return {"layers": n, "profile": [float(p) for p in prof], "peak_layer": None, "peak_offset_vx": None,
                "valid_fraction": float(np.isfinite(stack[n // 2]).mean())}
    i = int(np.nanargmax(prof))
    off = float(i)
    if 0 < i < n - 1:
        a, b, c = prof[i - 1], prof[i], prof[i + 1]
        den = a - 2 * b + c
        if den != 0:
            off = i + 0.5 * (a - c) / den
    return {"layers": n, "profile": [float(p) for p in prof], "peak_layer": i,
            "peak_offset_vx": float((off - 0.5 * (n - 1)) * step),
            "valid_fraction": float(np.isfinite(stack[n // 2]).mean())}


def procrustes_rms(uv: np.ndarray, ref: np.ndarray) -> dict:
    """RMS residual (same units as ref) after the best rigid (rotation+translation, reflection allowed) fit."""
    ok = np.all(np.isfinite(uv), 1) & np.all(np.isfinite(ref), 1)
    a, b = uv[ok], ref[ok]
    a0, b0 = a - a.mean(0), b - b.mean(0)
    U, _, Vt = np.linalg.svd(a0.T @ b0)
    res = a0 @ (U @ Vt) - b0
    d = np.linalg.norm(res, axis=1)
    ext = np.ptp(b, axis=0).max()
    return {"rms": float(np.sqrt((d**2).mean())), "p99": float(np.percentile(d, 99)),
            "max": float(d.max()), "rms_relative_to_extent": float(np.sqrt((d**2).mean()) / ext)}


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    x, y = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def surface_agreement(test: Surface, ref: Surface, lateral_tol: float = 2.0, sheet_tol: float = 4.0) -> dict:
    """Compare a surface against a reference surface of the same sheet (e.g. a human segment).

    For each test vertex, find the nearest reference vertex q and split p - q into the signed
    distance along the reference normal (sd) and the lateral remainder.  Only vertices that lie
    over the reference (lateral < lateral_tol) are scored.  A constant sd is a calibration offset
    (both on the same sheet); vertices with |sd - median(sd)| > sheet_tol are candidate sheet
    switches (adjacent wraps are typically >= 8 voxels apart).
    """
    from scipy.spatial import cKDTree

    rv = ref.valid
    rp = ref.points[rv].astype(np.float64)
    rn = ref.normals()[rv].astype(np.float64)
    good = np.all(np.isfinite(rn), 1)
    rp, rn = rp[good], rn[good]
    tp = test.points[test.valid].astype(np.float64)
    _, j = cKDTree(rp).query(tp)
    d = tp - rp[j]
    sd = (d * rn[j]).sum(1)
    lat = np.linalg.norm(d - sd[:, None] * rn[j], axis=1)
    over = lat < lateral_tol
    if not over.any():
        return {"overlap_vertices": 0}
    s = sd[over]
    med = float(np.median(s))
    dev = np.abs(s - med)
    return {"overlap_vertices": int(over.sum()), "overlap_fraction": float(over.mean()),
            "signed_offset_median_vx": med, "signed_offset_mad_vx": float(np.median(dev)),
            "abs_dev_p95_vx": float(np.percentile(dev, 95)),
            "sheet_switch_fraction": float((dev > sheet_tol).mean())}


def sheet_continuity(s: Surface, vol, radius: float = 1.5, rel_thresh: float = 0.35, level: int = 0) -> dict:
    """Is there papyrus under the surface?  For each vertex take the max density within +-radius
    along the normal; vertices below rel_thresh x (median of that max over the segment) are gaps:
    rips, holes, or places where the surface left the sheet.  Returns the gap map and fractions.
    """
    n = s.normals()
    best = np.full(s.shape, -np.inf, np.float32)
    for o in np.arange(-radius, radius + 1e-9, 0.5):
        best = np.fmax(best, vol.sample(s.points + o * n, level=level))
    best[~s.valid] = np.nan
    med = float(np.nanmedian(best))
    gap = best < rel_thresh * med
    v = s.valid & np.isfinite(best)
    return {"median_sheet_density": med, "gap_fraction": float(gap[v].mean()) if v.any() else None,
            "gap_map": gap}


def vertex_labels(s: Surface, labels: np.ndarray, origin_xyz=(0, 0, 0), reach: float = 2.0) -> np.ndarray:
    """Per grid cell: the first non-zero label at offsets 0, +-0.5, ... +-reach along the normal
    (labels: ZYX array whose voxel 0 sits at origin_xyz).  0 = no labelled papyrus nearby,
    -1 = invalid vertex or vertex outside the labelled block."""
    out = np.full(s.shape, -1, np.int64)
    v = s.valid
    pts = s.points[v].astype(np.float64)
    nn = np.nan_to_num(s.normals()[v].astype(np.float64))  # no normal -> only the vertex itself is probed
    o = np.asarray(origin_xyz, float)
    lab = np.zeros(len(pts), np.int64)
    inside = np.zeros(len(pts), bool)
    for k, off in enumerate(sorted(np.arange(-reach, reach + 1e-9, 0.5), key=abs)):
        q = np.rint(pts + off * nn - o).astype(np.int64)[:, ::-1]  # -> zyx index
        ok = np.all((q >= 0) & (q < np.array(labels.shape)), 1)
        if k == 0:
            inside = ok
        w = np.zeros(len(q), np.int64)
        w[ok] = labels[q[ok, 0], q[ok, 1], q[ok, 2]]
        take = (lab == 0) & (w != 0)
        lab[take] = w[take]
    lab[~inside] = -1
    out[v] = lab
    return out


def label_agreement(s: Surface, labels: np.ndarray, origin_xyz, seed_label: int, reach: float = 2.0) -> dict:
    """Score a traced surface against human per-sheet instance labels (ZYX array at origin_xyz).

    Each vertex takes the first non-zero label found at offsets 0, +-0.5, ... +-reach along its
    normal.  same = the seed's sheet, other = a different sheet (a sheet switch), none = no
    labelled papyrus nearby; outside = vertex not inside the labelled block (not scored).
    """
    lab = vertex_labels(s, labels, origin_xyz, reach)
    lab = lab[lab >= 0]
    if lab.size == 0:
        return {"scored": 0}
    same = lab == seed_label
    other = (lab != 0) & ~same
    return {"scored": int(lab.size), "same": float(same.mean()), "other_sheet": float(other.mean()),
            "none": float((lab == 0).mean()), "other_ids": sorted(set(lab[other].tolist()))[:10]}


def grid_rings(s: Surface, seed=None) -> np.ndarray:
    """Generation in which each vertex of a ``trace`` grid was grown (0 = seed block, -1 = invalid).

    Reconstructed from the final grid alone by replaying the tracer's growth rule: a missing cell is
    grown in generation g iff at the start of g it has two valid cells in a row along a grid axis
    (the extrapolation chain p1, p2); valid cells were accepted at their first attempt (failed cells
    are never retried).  The seed block is the 3x3 block snapped along one normal and never relaxed:
    the cell near ``seed`` (default ``s.meta['seed']``) whose 8 neighbours sit exactly one grid step
    (sqrt 2 diagonally) away orthogonally to the seed displacement.
    """
    v = s.valid
    ring = np.full(v.shape, -1, np.int64)
    if not v.any():
        return ring
    P = s.points.astype(np.float64)
    d = P - np.asarray(s.meta["seed"] if seed is None else seed, float)
    dist = np.where(v, np.linalg.norm(d, axis=-1), np.inf)
    step = 1 / s.scale[0]
    H, W = v.shape
    k = np.array([[2, 1, 2], [1, 0, 1], [2, 1, 2]]) ** 0.5 * step
    best, rc = np.inf, np.unravel_index(np.argmin(dist), v.shape)
    for r, c in np.argwhere(dist <= max(dist.min(), 5.0) + 0.01):
        if r < 1 or c < 1 or r > H - 2 or c > W - 2 or not v[r - 1:r + 2, c - 1:c + 2].all():
            continue
        nrm = np.linalg.norm(d[r, c])
        n = d[r, c] / nrm if nrm > 1e-3 else np.nan_to_num(s.normals()[r, c].astype(np.float64))
        e = P[r - 1:r + 2, c - 1:c + 2] - P[r, c]
        res = np.abs(np.linalg.norm(e - (e @ n)[..., None] * n, axis=-1) - k).max()
        if res < best:
            best, rc = res, (r, c)
    cur = np.zeros(v.shape, bool)
    cur[max(rc[0] - 1, 0):rc[0] + 2, max(rc[1] - 1, 0):rc[1] + 2] = True
    cur &= v
    ring[cur] = 0
    g = 0
    while True:
        g += 1
        cp = np.pad(cur, 2)
        ch = np.zeros(v.shape, bool)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ch |= cp[2 + dr:2 + dr + H, 2 + dc:2 + dc + W] & cp[2 + 2 * dr:2 + 2 * dr + H, 2 + 2 * dc:2 + 2 * dc + W]
        new = v & ~cur & ch
        if not new.any():
            break
        ring[new] = g
        cur |= new
    return ring


def offsheet_vs_labels(s: Surface, labels: np.ndarray, seed_label: int, origin_xyz=(0, 0, 0), max_ring: int = 30,
                       rings: np.ndarray | None = None, reach: float = 2.0) -> dict:
    """Primary wrap-jump measure: the share of a trace that lies on a DIFFERENT human-labelled sheet
    than the seed, counted only over vertices grown in generations 0..max_ring (``grid_rings``).

    A trace that grows further has more chances to jump, so comparing tracer variants by their
    whole-trace off-sheet rate rewards stalling; at matched generation it does not.  Denominator:
    vertices inside the labelled block (as ``label_agreement``: unlabelled vertices count as scored,
    reported in 'none').  'off_sheet' > 0.02 is the flag used by the benches.
    """
    lab = vertex_labels(s, labels, origin_xyz, reach)
    ring = grid_rings(s) if rings is None else rings
    m = (ring >= 0) & (ring <= max_ring) & (lab >= 0)
    lm = lab[m]
    if lm.size == 0:
        return {"scored": 0, "max_ring": max_ring}
    other = (lm != 0) & (lm != seed_label)
    return {"scored": int(lm.size), "max_ring": max_ring, "rings_present": int(ring[m].max()),
            "off_sheet": float(other.mean()), "same": float((lm == seed_label).mean()),
            "none": float((lm == 0).mean())}


def _inner(valid, rings, hole_rims=False):
    """valid minus its outermost `rings` rings (4-neighbour erosion; holes filled first unless hole_rims)."""
    from scipy.ndimage import binary_erosion, binary_fill_holes

    if not rings:
        return valid.copy()
    return valid & binary_erosion(valid if hole_rims else binary_fill_holes(valid), iterations=rings, border_value=0)


def surface_spacing(s: Surface, vol, k: int = 16, level: int = 0) -> float:
    """Typical sheet period (voxels) under a surface: median of ``trace.estimate_spacing`` along the
    normal at up to k valid vertices spread evenly over the grid.  More robust than the single
    estimate at the seed (meta 'sheet_spacing', NaN for ~10% of real Scroll 1 seeds).  NaN if no
    estimate succeeds."""
    from .trace import estimate_spacing

    n = s.normals()
    idx = np.argwhere(s.valid & np.all(np.isfinite(n), -1))
    if not len(idx):
        return float("nan")
    sel = idx[np.unique(np.rint(np.linspace(0, len(idx) - 1, k)).astype(int))]
    est = np.array([estimate_spacing(vol, s.points[r, c].astype(np.float64), n[r, c].astype(np.float64), level=level)
                    for r, c in sel])
    est = est[np.isfinite(est)]
    return float(np.median(est)) if len(est) else float("nan")


def _closest_on_triangles(p, a, b, c):
    """Closest points q on triangles (a, b, c) to points p (row-wise) and the Voronoi region of q:
    0 inside, 1/2/3 corner a/b/c, 4/5/6 edge ab/bc/ca (Ericson, Real-Time Collision Detection 5.1.5)."""
    dot = lambda x, y: (x * y).sum(1)  # noqa: E731
    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(invalid="ignore", divide="ignore"):
        den = va + vb + vc
        conds = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (d6 >= 0) & (d5 <= d6),
                 (vc <= 0) & (d1 >= 0) & (d3 <= 0), (va <= 0) & (d4 >= d3) & (d5 >= d6), (vb <= 0) & (d2 >= 0) & (d6 <= 0)]
        reg = np.select(conds, [1, 2, 3, 4, 5, 6], 0)
        tab, tbc, tca = d1 / (d1 - d3), (d4 - d3) / ((d4 - d3) + (d5 - d6)), d2 / (d2 - d6)
        v, w = vb / den, vc / den
    q = np.select([r[:, None] for r in (reg == 1, reg == 2, reg == 3, reg == 4, reg == 5, reg == 6)],
                  [a, b, c, a + tab[:, None] * ab, b + tbc[:, None] * (c - b), a + tca[:, None] * ac],
                  a + v[:, None] * ab + w[:, None] * ac)
    return q, reg


def sheet_crossing(P, Q, volume=None, label_volume=None, label_origin=(0, 0, 0), ext: float = 2.0, gap: float = 2.0,
                   step: float = 0.5) -> tuple:
    """Does the segment p -> q (rows of P, Q; xyz voxels) pass from one sheet onto another?  Returns (lab, dip):

    lab  1 = the human-labelled run at p is not the run at q: another ID, another ID between them, or an
           unlabelled gap >= ``gap`` vx between two runs of one ID (adjacent wraps can share an ID);
         0 = one continuous run of one ID (a depth offset inside one labelled sheet); -1 = unknown (no label
           within ``ext`` vx of an end, the segment leaves the labelled block, or no ``label_volume``).
           ``label_volume``: ZYX array whose voxel 0 sits at ``label_origin``; nearest-voxel lookup.
    dip  CT minimum strictly between the ends (> ext vx from both) over the lower of the two end densities
         (max within ext vx of p, of q); trilinear samples every ``step`` vx, 3-sample mean.  An inter-sheet
         gap gives a dip, a depth offset inside one sheet does not.  NaN without ``volume`` (anything with
         ``.sample(xyz)``) or when the ends are < 2 ext apart."""
    P, Q = np.asarray(P, np.float64).reshape(-1, 3), np.asarray(Q, np.float64).reshape(-1, 3)
    n = len(P)
    lab, dip = np.full(n, -1, np.int8), np.full(n, np.nan)
    if not n:
        return lab, dip
    L = np.linalg.norm(Q - P, axis=1)
    u = (Q - P) / np.maximum(L, 1e-9)[:, None]
    K = int(np.ceil((L.max() + 2 * ext) / step)) + 1
    k = np.arange(K)[None, :]
    t = -ext + step * k
    Lc = L[:, None]
    live = t <= Lc + ext + 1e-9
    X = P[:, None, :] + t[..., None] * u[:, None, :]
    near_p, near_q = live & (np.abs(t) <= ext + 1e-9), live & (np.abs(t - Lc) <= ext + 1e-9)
    if label_volume is not None:
        lv_ = np.asarray(label_volume)
        idx = np.rint(X - np.asarray(label_origin, float)).astype(np.int64)[..., ::-1]
        ins = np.all((idx >= 0) & (idx < np.array(lv_.shape)), -1)
        idx = np.where(ins[..., None], idx, 0)
        lv = np.where(ins, lv_[idx[..., 0], idx[..., 1], idx[..., 2]], 0).astype(np.int64)
        dp = np.where(near_p & (lv > 0), np.abs(t), np.inf)
        dq = np.where(near_q & (lv > 0), np.abs(t - Lc), np.inf)
        ia, ib = dp.argmin(1), dq.argmin(1)
        r = np.arange(n)
        a, b = lv[r, ia][:, None], lv[r, ib][:, None]
        seg = live & (k >= ia[:, None]) & (k <= ib[:, None])
        off = seg & (lv != a)
        c = np.cumsum(off, 1)
        run = (c - np.maximum.accumulate(np.where(off, 0, c), 1)).max(1)
        crossed = (a[:, 0] != b[:, 0]) | np.any(seg & (lv > 0) & (lv != a) & (lv != b), 1) | (run * step >= gap)
        known = np.isfinite(dp.min(1)) & np.isfinite(dq.min(1)) & (ia < ib) & np.all(ins | ~seg, 1)
        lab = np.where(known, crossed, -1).astype(np.int8)
    if volume is not None:
        v = np.asarray(volume.sample(X.reshape(-1, 3)), np.float64).reshape(n, K)
        v = np.concatenate([v[:, :1], (v[:, :-2] + v[:, 1:-1] + v[:, 2:]) / 3, v[:, -1:]], 1) if K > 2 else v
        ends = np.minimum(np.where(near_p, v, -np.inf).max(1), np.where(near_q, v, -np.inf).max(1))
        mid = np.where(live & (t > ext) & (t < Lc - ext), v, np.inf).min(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dip = np.where(np.isfinite(mid) & (ends > 0), mid / ends, np.nan)
    return lab, dip


def equivariance_switch(test: Surface, ref: Surface, spacing: float, test_labels: np.ndarray | None = None,
                        ref_labels: np.ndarray | None = None, rings: int = 1, ref_rings: int | None = None,
                        hole_rims: bool = False, window: float = 0.5, reach: float = 1.5, flag: float = 0.02,
                        volume=None, label_volume=None, label_origin=(0, 0, 0), confirm: str | None = "labels_first",
                        dip: float = 0.6, gap: float = 2.0) -> dict:
    """Sheet-switch rate between two surfaces that should lie on the same sheet, scored per test
    vertex against the reference surface UNDERNEATH it.

    Use: deformation equivariance (test = trace(W(CT)), ref = W(trace(CT))), seed-jitter or
    rotation controls, or any trace vs a reference segment of the same sheet.

    Matching: each test vertex p is matched to its foot point, the exact closest point q on the
    reference triangle mesh (p projected along the REFERENCE normal; not the nearest reference
    vertex, so no lateral tolerance and no grid-resampling mismatch).  The test normal is not used
    for matching: at a jump it tilts by tens of degrees and a line along it misses lacy references.
    Not scored: test vertices in the outermost ``rings`` rings of their grid (newest, least relaxed;
    ``hole_rims`` also drops vertices bordering interior holes); vertices whose foot lies on the
    reference mesh boundary more than half the window sideways (p is beside the reference, not over
    it) or farther than reach * spacing; and vertices whose foot triangle is unreliable: a corner in
    the reference's outermost ``ref_rings`` (default ``rings``) rings, or corners spread more than
    the window along the test normal (a cliff between sheets, a spike).
    A scored vertex is a SWITCH when
      * |p - q| <= window * spacing but q carries a different human label than p (both labelled;
        per-cell label grids as from ``vertex_labels``), or
      * |p - q| > window * spacing, no reference vertex (mesh or 1-cell streak) lies within
        window * spacing of p, AND the segment p -> q crosses onto a neighbouring sheet
        (``sheet_crossing``): with confirm="labels_first" (default) the human labels decide where
        both ends are labelled (another ID, or an unlabelled gap >= ``gap`` vx: adjacent wraps can
        share an ID near the core), else a CT dip (minimum between the ends < ``dip`` x the end
        density); "either" = label crossing or CT dip, "ct" / "labels" = that evidence alone.
        Without the test (confirm=None, or no ``volume`` and no ``label_volume``) every such vertex
        is a switch; that also counts depth offsets of 0.5-1 spacing inside one sheet, which two
        traces of one sheet show where they diverge (dev Scroll 1 controls: 63 of 144 such vertices
        had one continuous label between p and q, 60 of them no CT dip either).  Vertices that fail
        the test are scored, not switches (status 5).
    ``volume`` (``.sample(xyz)``) and ``label_volume`` (ZYX at ``label_origin``) are in the space of
    both surfaces.  Labels are optional (geometry only without them).  ``spacing`` = sheet period
    in voxels (e.g. ``surface_spacing``).  Returns counts, switch_fraction, flagged (> flag), the
    median signed offset (along the reference normal) of matched vertices, switches by clause,
    geom_unconfirmed, and per-cell maps: scored_map, switch_map, label_switch_map,
    geom_candidate_map, unconfirmed_map, offset_map (|p - q|), foot_map (q), cross_label_map and
    dip_map (``sheet_crossing`` of geometric candidates), status_map (-1 not a candidate, 0 no foot
    within reach, 1 foot on the mesh boundary or an unreliable triangle, 2 matched, 3 matched by a
    reference vertex only, 4 switch, 5 about a sheet away but no neighbouring sheet crossed).
    """
    from scipy.spatial import cKDTree

    tv = test.valid
    tin, rin = _inner(tv, rings, hole_rims), _inner(ref.valid, rings if ref_rings is None else ref_rings, hole_rims)
    _, V, T = ref.triangles()
    P = test.points[tin].astype(np.float64)
    D = test.normals()[tin].astype(np.float64)
    L, w = reach * spacing, window * spacing
    out = {"spacing": float(spacing), "window_vx": float(w), "candidates": int(len(P))}
    dist = np.full(len(P), np.inf)
    sdist = np.full(len(P), np.nan)
    ok_foot = np.zeros(len(P), bool)
    qlab = np.zeros(len(P), np.int64)
    Qf = np.full((len(P), 3), np.nan)
    dv = np.full(len(P), np.inf)
    if len(V) and len(P) and np.isfinite(L) and L > 0:
        dv = cKDTree(V).query(P, distance_upper_bound=L)[0]
    if len(T) and len(P) and np.isfinite(L) and L > 0:
        A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        ctr = (A + B + C) / 3
        rct = np.max([np.linalg.norm(X - ctr, axis=1) for X in (A, B, C)], 0)
        # the foot is no farther than the nearest mesh vertex: candidate centroids lie within that + the
        # triangle's radius; triangles are searched in radius classes (doubling) so a spike stays cheap
        dm = np.minimum(cKDTree(V[np.unique(T)]).query(P, distance_upper_bound=L)[0], L)
        cls = np.ceil(np.log2(np.maximum(rct / np.median(rct), 1.0))).astype(int)
        vis, tis = [], []
        for c in np.unique(cls):
            ix = np.flatnonzero(cls == c)
            cand = cKDTree(ctr[ix]).query_ball_point(P, dm + rct[ix].max() + 1e-6)
            vis.append(np.repeat(np.arange(len(P)), np.fromiter((len(x) for x in cand), np.int64, len(cand))))
            tis.append(ix[np.concatenate([np.asarray(x, np.int64) for x in cand] + [np.zeros(0, np.int64)])])
        vi, ti = np.concatenate(vis), np.concatenate(tis)
        if len(vi):
            q, reg = _closest_on_triangles(P[vi], A[ti], B[ti], C[ti])
            d = np.linalg.norm(P[vi] - q, axis=1)
            nt = np.cross(B[ti] - A[ti], C[ti] - A[ti])
            nt /= np.maximum(np.linalg.norm(nt, axis=1, keepdims=True), 1e-12)
            sd = ((P[vi] - q) * nt).sum(1)
            side = np.sqrt(np.maximum(d * d - sd * sd, 0))  # tangential part of p - q (0 for an interior foot)
            # boundary of the mesh: edges used by one triangle, and their corners
            E = np.stack([T, np.roll(T, -1, 1)], 2).reshape(-1, 2)
            key = np.minimum(E[:, 0], E[:, 1]) * len(V) + np.maximum(E[:, 0], E[:, 1])
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            eb = (cnt[inv] == 1).reshape(-1, 3)  # edge i of a triangle = corners i, i+1
            vb = np.zeros(len(V), bool)
            vb[E[cnt[inv] == 1].ravel()] = True
            tri = T[ti]
            onb = np.where(reg == 0, False, np.where(reg <= 3, vb[tri[np.arange(len(ti)), np.clip(reg - 1, 0, 2)]],
                                                     eb[ti, np.clip(reg - 4, 0, 2)])) & (side > 0.5 * w)
            n = np.where(np.all(np.isfinite(D[vi]), 1)[:, None], D[vi], 0.0)
            spread = np.ptp(np.stack([((V[tri[:, i]] - P[vi]) * n).sum(1) for i in range(3)], 1), 1)
            rel = ~onb & rin[ref.valid][tri].all(1) & (spread <= w)
            o = np.lexsort((~rel, np.round(d, 6), vi))  # nearest foot; on a tie (a vertex hit) a reliable one
            f = o[np.r_[True, vi[o][1:] != vi[o][:-1]]]
            vi, q, tri = vi[f], q[f], tri[f]
            dist[vi], sdist[vi], ok_foot[vi], Qf[vi] = d[f], sd[f], rel[f], q
            if ref_labels is not None:
                corner = np.argmin(np.stack([np.linalg.norm(V[tri[:, i]] - q, axis=1) for i in range(3)], 1), 1)
                qlab[vi] = np.asarray(ref_labels)[ref.valid][tri[np.arange(len(vi)), corner]]
    scored = ok_foot & (dist <= L)
    inwin = scored & (dist <= w)
    near = scored & ~inwin & (dv <= w)
    gc = scored & ~inwin & ~near
    xl, xd = np.full(len(P), -2, np.int8), np.full(len(P), np.nan)
    xl[gc], xd[gc] = sheet_crossing(P[gc], Qf[gc], volume, label_volume, label_origin, gap=gap)
    ct = xd < dip  # NaN (no volume) -> False
    conf = {None: gc, "either": (xl == 1) | ct, "labels_first": np.where(xl >= 0, xl == 1, ct), "ct": ct,
            "labels": xl == 1}[confirm if volume is not None or label_volume is not None else None]
    geo_sw, unconf = gc & conf, gc & ~conf
    lab_sw = geom_same = np.zeros(len(P), bool)
    if test_labels is not None and ref_labels is not None:
        tl = np.asarray(test_labels)[tin]
        both = (tl > 0) & (qlab > 0)
        lab_sw = inwin & both & (tl != qlab)
        geom_same = geo_sw & both & (tl == qlab)
    sw = lab_sw | geo_sw
    k = int(scored.sum())
    out.update(scored=k, switches=int(sw.sum()), switch_fraction=float(sw.sum() / k) if k else None,
               flagged=bool(k and sw.sum() / k > flag), switch_label=int(lab_sw.sum()), switch_geom=int(geo_sw.sum()),
               switch_geom_same_label=int(geom_same.sum()), geom_unconfirmed=int(unconf.sum()),
               matched_by_vertex=int(near.sum()),
               dropped_unreliable=int((~ok_foot & (dist <= L)).sum()),
               offset_median_vx=float(np.median(sdist[inwin])) if inwin.any() else None)
    for name, val in (("scored_map", scored), ("switch_map", sw), ("label_switch_map", lab_sw), ("geom_candidate_map", gc),
                      ("unconfirmed_map", unconf)):
        out[name] = np.zeros(tv.shape, bool)
        out[name][tin] = val
    for name, val, fill in (("offset_map", np.where(scored, dist, np.nan), np.nan), ("cross_label_map", xl, -2),
                            ("dip_map", xd, np.nan)):
        out[name] = np.full(tv.shape, fill, np.asarray(val).dtype)
        out[name][tin] = val
    out["foot_map"] = np.full(tv.shape + (3,), np.nan)
    out["foot_map"][tin] = Qf
    out["status_map"] = np.full(tv.shape, -1, np.int8)
    out["status_map"][tin] = np.select([sw, unconf, near, inwin, dist <= L], [4, 5, 3, 2, 1], 0)
    return out


def axis_direction(surface: Surface, up=(0.0, 0.0, 1.0)) -> tuple[float, float]:
    """Unit (d_row, d_col) direction in which the scroll axis runs across the surface's image grid."""
    up = np.asarray(up, float)
    h = surface.points @ up
    dr = np.nanmean(np.diff(h, axis=0))
    dc = np.nanmean(np.diff(h, axis=1))
    n = np.hypot(dr, dc)
    return (float(dr / n), float(dc / n)) if n > 0 else (1.0, 0.0)


def sheet_band(mean_profile, frac: float = 0.5, min_contrast: float = 0.05):
    """(first, last) layer of the sheet the surface sits on: the contiguous run of layers around the brightest
    layer of the central half of the stack whose mean is >= ``frac`` of the way from the stack minimum to it.
    None if the central half has no data or the profile has no sheet in it (Michelson contrast of peak vs
    minimum <= ``min_contrast``; measured 0.17-0.55 on real Scroll 1 renders, more on phantoms)."""
    mu = np.asarray(mean_profile, float)
    n = len(mu)
    c, w = (n - 1) // 2, max(1, n // 4)
    mid = mu[max(c - w, 0):c + w + 1]
    if not np.isfinite(mid).any():
        return None
    i = max(c - w, 0) + int(np.nanargmax(mid))
    lo = np.nanmin(mu)
    if mu[i] - lo <= min_contrast * abs(mu[i] + lo):
        return None
    thr = lo + frac * (mu[i] - lo)
    a, b = i, i
    while a > 0 and mu[a - 1] >= thr:
        a -= 1
    while b < n - 1 and mu[b + 1] >= thr:
        b += 1
    return a, b


def fiber_profile(stack: np.ndarray, step: float = 1.0, tile: int | None = 64, hp_sigma: float = 4.0,
                  axis=(1.0, 0.0), min_pixels: int = 256, band_frac: float | None = 0.5) -> dict:
    """Which papyrus layer is where?  Label-free, from fiber orientation in the rendered layers.

    Recto fibers run around the roll (across the scroll axis), so they make intensity vary ALONG
    the axis; verso fibers run parallel to the axis.  ``axis`` = (d_row, d_col) direction of the
    scroll axis in the image: pass ``axis_direction(surface, up)``.  (1, 0) is right only when image
    rows follow the axis; with the axis along columns the call inverts (real segment 20230530164535 is
    such a case).  Per layer:
    R = (E_along_axis - E_across_axis) / (sum)  on high-passed images; R > 0 recto-like, R < 0 verso-like.
    R is a ratio and blind to amplitude, and rendering alone gives R ~ +0.03..0.2 on isotropic noise, so
    only the relative position of the extremes is used.

    Recto and verso are searched only inside the sheet the surface sits on (``sheet_band`` of the
    mean-intensity profile, ``band_frac``; None, or a stack without a sheet in it = whole stack).  A wide
    stack also holds the faint facing layers of the neighbouring wraps -- the inner wrap's verso and the
    outer wrap's recto, the reverse order -- and a whole-stack argmax/argmin picked them on 5/70
    ground-truth phantom surfaces and on 3/3 real Scroll 1 crops (verso found in the next gap,
    +10..+14 vx).  A layer is scored if it has >= ``min_pixels`` valid pixels: a
    flattened segment often covers < 50% of its bounding image (34/70 traced phantoms), which a
    coverage-fraction gate turned into no call.  Gradients are taken between valid pixels only (not at
    the image border or a mask edge) of a NaN-aware high-pass.

    Returns the global R profile, the recto/verso peak offsets (voxels along +normal; ink is on
    the recto face away from the verso), and per-tile interface depths -- a jump of the interface
    between neighbouring tiles is a layer hop of the surface.
    """
    from scipy.ndimage import binary_erosion, gaussian_filter

    n = len(stack)
    offs = (np.arange(n) - 0.5 * (n - 1)) * step
    k3 = np.ones((3, 3), bool)

    def energies(img, min_px):
        ok = np.isfinite(img)
        m = binary_erosion(ok, k3)  # central differences between valid pixels only, none at the border
        if ok.sum() < min_px or not m.any():
            return np.nan, np.nan, np.nan
        # NaN-aware high-pass: filling holes with a constant would add edges along the mask boundary
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(ok, img - gaussian_filter(np.where(ok, img, 0.0), hp_sigma)
                         / gaussian_filter(ok.astype(float), hp_sigma), 0.0)
        gy, gx = np.gradient(f)
        ga = gy * axis[0] + gx * axis[1]  # derivative along the scroll axis
        gc = -gy * axis[1] + gx * axis[0]  # derivative across it
        return float((ga[m] ** 2).mean()), float((gc[m] ** 2).mean()), float(img[ok].mean())

    def profile(sub, min_px):
        e = np.array([energies(layer, min_px) for layer in sub]).reshape(-1, 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            R = (e[:, 0] - e[:, 1]) / (e[:, 0] + e[:, 1])
        R[~(e[:, 0] + e[:, 1] > 0)] = np.nan
        ab = sheet_band(e[:, 2], band_frac) if band_frac is not None and np.isfinite(e[:, 2]).any() else None
        if ab is None:
            return R, R, None
        inb = np.zeros(len(R), bool)
        inb[ab[0]:ab[1] + 1] = True
        return R, np.where(inb, R, np.nan), ab

    R, Rb, ab = profile(stack, min_pixels)
    out = {"offsets_vx": offs.tolist(), "R": [None if not np.isfinite(r) else round(float(r), 4) for r in R]}
    if ab is not None:
        out["sheet_band_vx"] = [float(offs[ab[0]]), float(offs[ab[1]])]
    if np.isfinite(R).sum() >= 3 and np.isfinite(Rb).sum() >= 2:
        out["recto_offset_vx"] = float(offs[np.nanargmax(Rb)])
        out["verso_offset_vx"] = float(offs[np.nanargmin(Rb)])
        out["recto_verso_contrast"] = float(np.nanmax(Rb) - np.nanmin(Rb))
        # ink face: the recto side facing away from the verso
        out["ink_side"] = int(np.sign(out["recto_offset_vx"] - out["verso_offset_vx"]))
    if tile:
        H, W = stack.shape[1:]
        depths = []
        for y in range(0, H - tile + 1, tile):
            row = []
            for x in range(0, W - tile + 1, tile):
                Rt, Rtb, _ = profile(stack[:, y:y + tile, x:x + tile], tile * tile // 2)
                if np.isfinite(Rt).sum() >= 3 and np.isfinite(Rtb).sum() >= 2 \
                        and np.nanmax(Rtb) - np.nanmin(Rtb) > 0.2:
                    row.append(0.5 * (offs[np.nanargmax(Rtb)] + offs[np.nanargmin(Rtb)]))
                else:
                    row.append(np.nan)
            depths.append(row)
        d = np.array(depths, float)
        out["tile_interface_depth_vx"] = d.tolist()
        if np.isfinite(d).sum() >= 2:
            med = np.nanmedian(d)
            out["interface_depth_median_vx"] = float(med)
            out["layer_hop_tile_fraction"] = float(np.nanmean(np.abs(d - med) > 2.0))
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion k / n (95% for z = 1.96)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(c - h, 0.0)), float(min(c + h, 1.0)))
