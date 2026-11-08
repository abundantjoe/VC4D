"""Score a whole-scroll result against a WholeScroll's exact ground truth (vectorised).

A result is a list of tifxyz ``Surface``s (or one stitched surface).  Each vertex may carry a claim:
  kind="winding"  global winding number per vertex (any constant offset; the best global offset is found)
  kind="theta"    unwrapped angle per vertex in radians (any constant offset)
  kind="s"        arc length per vertex, e.g. the u coordinate of a stitched flattening (any offset)
  claims=None     no claim: only geometric / topological scores (coverage, jumps, duplicates, distortion)

Reported (areas in vx^2 on the sheet; cells of ``cell`` x ``cell`` vx in the true (s, z) layout, aligned to
every wrap's start so that no cell spans two wraps):
  on_sheet_fraction        vertices within the sheet (|depth| <= half local thickness + sheet_tol)
  winding_accuracy         fraction of on-sheet vertices whose (globally aligned) claim is the true wrap;
                           also per-surface-aligned, and per true wrap
  coverage[w]              fraction of the true present sheet area of wrap w covered by on-sheet vertices
                           (any claim) and by correctly wound vertices.  Cell resolution: a perfect
                           result scores >= ~0.95 (cells cut by the edges of tears / missing pieces may
                           hold no vertex); compare results at the same ``cell``
  jump_edges, jump_frac    grid edges whose endpoints lie on different wraps (|Delta s| or |Delta z| far
                           larger than the grid step): topological cross-wrap connections
  wrong_wrap_area          area of on-sheet vertices claimed on the wrong wrap, and its connected components
  duplicate_area           true sheet cells covered by >= 2 surfaces; self_overlap_area: cells covered by
                           one surface from grid positions far apart (fold-over)
  flatten                  given uv (vx) per vertex: rigid Procrustes RMS vs the true (s, z) layout and
                           triangle singular values of the true->uv map (isometry: s1 = s2 = 1)
  ink_ncc                  given a per-vertex render: NCC with the true ink at the vertex's true (s, z)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

TWO_PI = 2 * np.pi


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def present_cells(P, cell: float):
    """Cells of ``cell`` x ``cell`` vx over the true (s, z) layout, aligned to the start of every wrap (a
    cell never spans two wraps), and which of them hold papyrus inside the volume.  Cached on P.
    Returns (present (n_cells,) bool, wrap (n_cells,) int, s_start (W+1,), first cell of each wrap, nz)."""
    cache = P.__dict__.setdefault("_present_cells", {})
    if cell in cache:
        return cache[cell]
    Z = P.size[0]
    nz = int(np.ceil(Z / cell))
    W = int(np.floor(P._tc[-1] / TWO_PI)) + 1
    s_start = np.interp(TWO_PI * np.arange(W + 1), P._tc, P._sc)
    s_start[-1] = P.length
    nsw = np.ceil((s_start[1:] - s_start[:-1]) / cell).astype(int)
    first = np.r_[0, np.cumsum(nsw)[:-1]] * nz
    pres, wrap = [], []
    zc = (np.arange(nz) + 0.5) * cell
    for w in range(W):
        sc = np.minimum(s_start[w] + (np.arange(nsw[w]) + 0.5) * cell, s_start[w + 1] - 1e-6)
        S, Zc = np.meshgrid(sc, zc, indexing="ij")
        p = P.point(S, Zc)
        inside = np.all((p >= 0) & (p <= np.array(P.size[::-1]) - 1), -1)
        pres.append((inside & P.present(S, Zc)).ravel())
        wrap.append(np.full(S.size, w))
    cache[cell] = (np.concatenate(pres), np.concatenate(wrap), s_start, first, nz)
    return cache[cell]


def _align(claim, truth, kind, P, on):
    """Best global offset of a claim -> claimed winding per vertex."""
    if kind == "winding":
        d = (truth["w"] - claim)[on]
        off = np.bincount((d - d.min()).astype(int)).argmax() + d.min() if d.size else 0
        return claim + off, off
    if kind == "theta":
        off = float(np.median((truth["theta"] - claim)[on])) if on.any() else 0.0
        return np.floor((claim + off) / TWO_PI), off
    if kind == "s":
        off = float(np.median((truth["s"] - claim)[on])) if on.any() else 0.0
        return np.floor(np.interp(claim + off, P._sc, P._tc) / TWO_PI), off
    raise ValueError(kind)


def _triangle_sv(src, dst, valid):
    """Singular values / det of the affine map src -> dst on the two triangles of every grid quad."""
    out_s, out_d = [], []
    for a, b, c in (((0, 0), (1, 0), (0, 1)), ((1, 1), (0, 1), (1, 0))):
        def sl(o):
            return (slice(o[0], src.shape[0] - 1 + o[0]), slice(o[1], src.shape[1] - 1 + o[1]))
        ok = valid[sl(a)] & valid[sl(b)] & valid[sl(c)]
        E1 = np.stack([src[sl(b)] - src[sl(a)], src[sl(c)] - src[sl(a)]], -1)[ok]
        E2 = np.stack([dst[sl(b)] - dst[sl(a)], dst[sl(c)] - dst[sl(a)]], -1)[ok]
        good = np.abs(np.linalg.det(E1)) > 1e-9
        J = E2[good] @ np.linalg.inv(E1[good])
        out_s.append(np.linalg.svd(J, compute_uv=False))
        out_d.append(np.linalg.det(J))
    return np.concatenate(out_s), np.concatenate(out_d)


def _procrustes(a, b):
    """Rigid (rotation + reflection + translation) fit of a onto b; RMS residual."""
    am, bm = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - am).T @ (b - bm))
    R = U @ Vt
    return float(np.sqrt(np.mean(np.sum(((a - am) @ R - (b - bm)) ** 2, 1))))


def score_whole(P, surfaces, claims=None, kind="winding", uvs=None, renders=None, cell=None,
                sheet_tol: float = 1.0) -> dict:
    surfaces = _as_list(surfaces)
    claims = None if claims is None else _as_list(claims)
    uvs = None if uvs is None else _as_list(uvs)
    renders = None if renders is None else _as_list(renders)
    steps = [1.0 / np.sqrt(s.scale[0] * s.scale[1]) for s in surfaces]
    cell = float(cell or max(4.0, 2 * max(steps)))
    # ---- truth for every valid vertex of every surface
    V, sid, rc, cl = [], [], [], []
    for i, S in enumerate(surfaces):
        m = S.valid
        r, c = np.nonzero(m)
        V.append(S.points[m])
        sid.append(np.full(len(r), i))
        rc.append(np.stack([r, c], 1))
        if claims is not None:
            cl.append(np.asarray(claims[i], float)[m])
    V, sid, rc = np.concatenate(V), np.concatenate(sid), np.concatenate(rc)
    T = P.sheet_coords(V)
    half = 0.5 * P.local_thickness(T["s"]) + sheet_tol
    on = np.abs(T["depth"]) <= half
    rep = {"vertices": int(len(V)), "cell_vx": cell, "on_sheet_fraction": float(on.mean()) if len(V) else 0.0}
    w_true = T["w"].astype(int)
    # ---- winding claims
    correct = on.copy()
    if claims is not None:
        claim = np.concatenate(cl)
        wc, off = _align(claim, T, kind, P, on)
        correct = on & (wc == T["w"])
        rep["winding_offset"] = float(off)
        rep["winding_accuracy"] = float(correct[on].mean()) if on.any() else 0.0
        acc_s = []
        for i in range(len(surfaces)):  # per-surface alignment (surfaces individually consistent?)
            k = sid == i
            wci, _ = _align(claim[k], {q: v[k] for q, v in T.items()}, kind, P, on[k])
            acc_s.append(float((wci == T["w"][k])[on[k]].mean()) if on[k].any() else 0.0)
        rep["winding_accuracy_per_surface_aligned"] = acc_s
        if kind == "theta":
            e = np.abs(claim + off - T["theta"])[on]
            rep["theta_error_rad"] = {"median": float(np.median(e)), "p95": float(np.percentile(e, 95))} if e.size else None
    # ---- coverage per true wrap
    pres_f, wrap_f, s_start, first, nz = present_cells(P, cell)
    wv = np.clip(w_true, 0, len(first) - 1)
    cs = np.maximum(((T["s"] - s_start[wv]) // cell).astype(int), 0)
    ncell_w = np.diff(np.r_[first, len(pres_f)]) // nz
    cs = np.minimum(cs, ncell_w[wv] - 1)
    cz = np.clip((T["z"] // cell).astype(int), 0, nz - 1)
    cid = first[wv] + cs * nz + cz
    n_cells = len(pres_f)
    cov_any = np.zeros(n_cells, bool)
    cov_any[cid[on]] = True
    cov_ok = np.zeros(n_cells, bool)
    cov_ok[cid[correct]] = True
    W = len(first)
    tot = np.bincount(wrap_f[pres_f], minlength=W)
    a_any = np.bincount(wrap_f[pres_f & cov_any], minlength=W)
    a_ok = np.bincount(wrap_f[pres_f & cov_ok], minlength=W)
    rep["coverage"] = {"total_present_area": float(tot.sum() * cell**2),
                       "covered_fraction": float(a_any.sum() / max(tot.sum(), 1)),
                       "correct_fraction": float(a_ok.sum() / max(tot.sum(), 1)),
                       "per_wrap": [{"w": w, "present_area": float(tot[w] * cell**2),
                                     "covered": float(a_any[w] / tot[w]) if tot[w] else None,
                                     "correct": float(a_ok[w] / tot[w]) if tot[w] else None} for w in range(W)]}
    # ---- duplicates and self-overlap
    pairs = np.unique(np.stack([cid[on], sid[on]], 1), axis=0)
    n_surf = np.bincount(pairs[:, 0], minlength=n_cells)
    rep["duplicate_area"] = float(np.sum((n_surf >= 2) & pres_f) * cell**2)
    key = sid[on].astype(np.int64) * n_cells + cid[on]
    o = np.argsort(key, kind="stable")
    ks, r_, c_ = key[o], rc[on][o, 0], rc[on][o, 1]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    span = np.maximum(np.maximum.reduceat(r_, starts) - np.minimum.reduceat(r_, starts),
                      np.maximum.reduceat(c_, starts) - np.minimum.reduceat(c_, starts))
    lim = np.array([cell / steps[s] + 3 for s in sid[on][o][starts]])
    rep["self_overlap_area"] = float(np.sum(span > lim) * cell**2)
    # ---- topological cross-wrap errors (grid edges between wraps) and wrong-wrap area
    jumps, edges, wrong_area, comps, comp_areas = 0, 0, 0.0, 0, []
    per = []
    for i, S in enumerate(surfaces):
        k = sid == i
        H, Wd = S.shape
        g = lambda v, fill=np.nan: _grid(v[k], rc[k], (H, Wd), fill)  # noqa: E731
        s_g, z_g, on_g = g(T["s"]), g(T["z"]), g(on.astype(float), 0.0) > 0
        thr = 4 * steps[i] + 2
        nj = 0
        for ax in (0, 1):
            a0 = tuple(slice(0, -1) if j == ax else slice(None) for j in range(2))
            a1 = tuple(slice(1, None) if j == ax else slice(None) for j in range(2))
            both = on_g[a0] & on_g[a1]
            jmp = both & ((np.abs(s_g[a1] - s_g[a0]) > thr) | (np.abs(z_g[a1] - z_g[a0]) > thr))
            nj += int(jmp.sum())
            edges += int(both.sum())
        jumps += nj
        if claims is not None:
            bad = on_g & ~(g(correct.astype(float), 0.0) > 0)
            lab, n = label(bad)
            areas = np.bincount(lab.ravel())[1:] * steps[i] ** 2
            wrong_area += float(areas.sum())
            comps += int(n)
            comp_areas += areas.tolist()
        per.append({"surface": i, "jump_edges": nj})
    rep["jump_edges"] = jumps
    rep["jump_frac"] = jumps / max(edges, 1)
    if claims is not None:
        rep["wrong_wrap_area"] = wrong_area
        rep["wrong_wrap_components"] = comps
        rep["wrong_wrap_largest_component_area"] = float(max(comp_areas)) if comp_areas else 0.0
    rep["per_surface"] = per
    # ---- flattening distortion against the true layout
    if uvs is not None:
        fl = []
        for i, S in enumerate(surfaces):
            k = sid == i
            uv = np.asarray(uvs[i], float)
            good_g = _grid(correct[k].astype(float), rc[k], S.shape, 0.0) > 0
            good_g &= np.all(np.isfinite(uv), -1)
            if good_g.sum() < 10:
                fl.append(None)
                continue
            st = np.stack([_grid(T["s"][k], rc[k], S.shape), _grid(T["z"][k], rc[k], S.shape)], -1)
            rms = _procrustes(uv[good_g], st[good_g])
            sv, det = _triangle_sv(st, uv, good_g)
            flip = np.sign(det) != np.sign(np.median(det)) if det.size else np.zeros(0, bool)
            fl.append({"procrustes_rms_vx": rms,
                       "abs_log_area_distortion_median": float(np.median(np.abs(np.log(sv[:, 0] * sv[:, 1])))) if sv.size else None,
                       "anisotropy_s1_over_s2_p95": float(np.percentile(sv[:, 0] / sv[:, 1], 95)) if sv.size else None,
                       "flipped_triangle_frac": float(flip.mean()) if flip.size else None})
        rep["flatten"] = fl
    # ---- ink in the flattened frame
    if renders is not None:
        vals = np.concatenate([np.asarray(r, float)[S.valid] for r, S in zip(renders, surfaces)])
        ink = P.ink_at(T["s"], T["z"])
        m = correct & np.isfinite(vals)
        rep["ink_ncc"] = float(np.corrcoef(vals[m], ink[m])[0, 1]) if m.sum() > 10 and ink[m].std() > 0 else None
    return rep


def _grid(v, rc, shape, fill=np.nan):
    out = np.full(shape, fill, float)
    out[rc[:, 0], rc[:, 1]] = v
    return out
