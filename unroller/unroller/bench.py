"""Large-scale benchmark over randomly drawn phantom scrolls.

For every phantom (``Phantom.random(seed)``) two tasks are scored against exact ground truth:

A. *known segment*: flatten a deliberately non-isometric ground-truth grid;
B. *blind unroll* (a scroll never segmented before): seed point -> trace -> remove_folds -> regularize ->
   fill_holes -> flatten -> render along smoothed normals, using only the CT volume (``pipeline="refine"``: the
   previous trace -> refine (lamina peak) -> render along raw normals, kept for before/after comparisons).

Ink is scored twice.  ``B_ink_layout_ncc`` compares one rendered layer with the text.  On two-layer phantoms the
ink lies on the recto face, where the lamina is at half density and carries +-35 % fiber texture (~26 grey levels
against an ink step of 20-60 on ~10 % of the pixels), so even the ink sampled at its exact depth under every pixel
reaches only ~0.24 (``B_ink_ceiling_ncc``) and the 0.5 threshold cannot be met with any geometry.  The
``B_ink_signal_*`` scores render the same surface through the ink alone (the phantom volume minus the same volume
without ink: identical noise and texture), i.e. they test what the bench is for: whether geometry, layer offsets
and registration carry the ink to the right pixel.

Aggregates report the median / p95 of each metric and pass rates with Wilson 95% confidence
intervals, so "it works" is a statement with error bars.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .flatten import flatten
from .metrics import axis_direction, fiber_profile, flatten_report, ncc, procrustes_rms
from .phantom import Phantom
from .refine import refine
from .regularize import fill_holes, interior_holes, regularize, remove_folds
from .render import render
from .resample import flat_surface
from .surface import NORMAL_SIGMA_VX
from .trace import trace
from .volume import ArrayVolume

PASS = {
    "A_uv_rms_vx": 0.25,
    "B_offsheet_rms_vx": 0.5,
    "B_wrap_jumps": 0,
    "B_uv_rel_rms": 0.005,       # flattened layout vs truth, relative to extent
    "B_ink_layout_ncc": 0.5,     # rendered text vs truth text in the *unrolled layout* frame
    "B_ink_signal_layout_ncc": 0.5,  # the same for the ink signal alone (texture and noise removed)
}


HIST_BINS = np.r_[np.arange(0, 3.0001, 0.01), np.inf]  # |distance from the true sheet centre| histograms (vx)


def _rigid_fit(src, dst):
    """Best rigid map src->dst (reflection allowed). Returns function."""
    a0, b0 = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - a0).T @ (dst - b0))
    R = U @ Vt
    return lambda p: (p - a0) @ R + b0


def wrap_report(P: Phantom, pts: np.ndarray, step: float) -> dict:
    """Grid defects of a traced (H, W, 3) grid against phantom truth.  A jump is an edge between adjacent
    vertices whose true arc lengths differ by > 4 step: *cross-wrap* if their true spiral angles differ by a
    whole turn (round(dt / 2 pi) != 0: the trace continues on another wrap), else a same-wrap *tear*.
    fold_frac: grid quads (without a jump) whose orientation in true (s, z), sign of det d(s,z)/d(col,row),
    opposes the majority -- the grid folded over itself."""
    s, z, _ = P._locate(np.asarray(pts, float))
    t = np.interp(s, P._sc, P._tc)
    t[~np.isfinite(s)] = np.nan
    cross = tears = 0
    for ax in (0, 1):
        j = np.abs(np.diff(s, axis=ax)) > 4 * step
        k = np.round(np.diff(t, axis=ax) / (2 * np.pi))
        cross += int(np.sum(j & (k != 0)))
        tears += int(np.sum(j & (k == 0)))
    ds_dc, ds_dr = np.diff(s, axis=1)[:-1], np.diff(s, axis=0)[:, :-1]
    dz_dc, dz_dr = np.diff(z, axis=1)[:-1], np.diff(z, axis=0)[:, :-1]
    det = ds_dc * dz_dr - ds_dr * dz_dc
    with np.errstate(invalid="ignore"):
        q = np.isfinite(det) & (np.abs(ds_dc) < 4 * step) & (np.abs(ds_dr) < 4 * step)
    sg = np.sign(np.median(det[q])) if q.any() else 1.0
    fold = float((q & (np.sign(det) != sg)).sum() / max(q.sum(), 1))
    return {"cross_wrap_edges": cross, "tears": tears, "fold_frac": fold}


def _layout_ncc(P, F, tF, img):
    """NCC of img with the true text placed by ONE rigid transform of the flat image grid."""
    rr, cc = np.nonzero(F.valid)
    fit = _rigid_fit(np.c_[cc, rr].astype(float), tF[rr, cc])
    lay = np.full(F.shape, np.nan)
    st_ = fit(np.c_[cc, rr].astype(float))
    lay[rr, cc] = P.ink_at(st_[:, 0], st_[:, 1])
    return ncc(img, lay)


def _normal_checks(P, F, tF, nrm, okF, s_, z_):
    """Render normals vs the TRUE sheet normal per pixel (sign fixed per surface by the median).  ``_all`` counts every
    pixel; the others leave out pixels where the unrolling is locally mirrored (d(s, z)/d(col, row) has the minority
    sign: a fold of the trace, or a triangle flipped by jitter)."""
    nt = np.full(F.points.shape, np.nan)
    d = np.cross(P.point(s_ + 0.5, z_) - P.point(s_ - 0.5, z_), P.point(s_, z_ + 0.5) - P.point(s_, z_ - 0.5))
    d *= np.sign((d * (P.point(s_, z_, 1.0) - P.point(s_, z_, 0.0))).sum(1))[:, None]  # outward, like the material
    nt[okF] = d / np.linalg.norm(d, axis=1, keepdims=True)
    J = np.gradient(tF[..., 0], axis=1) * np.gradient(tF[..., 1], axis=0) - \
        np.gradient(tF[..., 0], axis=0) * np.gradient(tF[..., 1], axis=1)
    m = okF & np.isfinite(nrm[..., 0]) & np.isfinite(J) & np.isfinite(nt[..., 0])
    mir = m & (J * np.sign(np.median(J[m])) < 0)
    c = (nrm * nt).sum(-1)
    good = m & ~mir
    c = c * np.sign(np.median(c[good]))
    ang = np.degrees(np.arccos(np.clip(c[good], -1, 1)))
    return {"B_render_normal_px": int(good.sum()), "B_render_normal_reversed_px": int((c[good] < 0).sum()),
            "B_render_normal_all_px": int(m.sum()), "B_render_normal_reversed_all_px": int((c[m] < 0).sum()),
            "B_render_normal_angle_p50_p90_deg": [float(v) for v in np.percentile(ang, [50, 90])],
            "B_layout_mirrored_frac": float(mir.sum() / max(m.sum(), 1))}


def _normal_frame(P, F, n=None) -> int:
    """+1 if the render's +normal is the phantom's outward normal (the frame of ``P.ink_depth()``), else -1.
    ``n`` = the normals the stack was rendered with (default: the raw grid normals)."""
    n = F.normals() if n is None else n
    ok = F.valid & np.all(np.isfinite(n), -1)
    p, q = F.points[ok][::7].astype(float), n[ok][::7].astype(float)
    return int(np.sign(np.nanmedian(P.sheet_distance(p + q) - P.sheet_distance(p))))


def run_one(seed: int, size=(64, 192, 192), step: float = 3.0, generations: int = 70, two_layer: bool = True,
            pipeline: str = "regularize") -> dict:
    t0 = time.time()
    P = Phantom.random(seed, size=size, two_layer=two_layer)
    a = P.volume()
    V = ArrayVolume(a)
    out = {"seed": seed, "params": P.params, "min_gap": P.min_gap, "pipeline": pipeline}

    # --- A: flatten a known, distorted grid
    S = P.surface(distort=0.3)
    uv = flatten(S)
    gt = P.unrolled_coords(S.points)
    out["A_uv_rms_vx"] = procrustes_rms(uv.reshape(-1, 2), gt.reshape(-1, 2))["rms"]
    out["A_flips"] = flatten_report(S, uv)["flipped_triangles"]

    # --- B: blind unroll from an imperfect seed
    rng = np.random.default_rng(seed + 7)
    s_seed = rng.uniform(0.35, 0.65) * P.length
    seed_pt = P.point(s_seed, size[0] / 2, offset=rng.uniform(-1, 1))
    Z = size[0]
    try:
        T = trace(V, seed_pt, step=step, generations=generations,
                  bounds=((2, 2, 2), (size[2] - 3, size[1] - 3, Z - 3)), umbilicus=P.center,
                  up=P.R[:, 2])
    except Exception as e:  # a failed trace is a result, not a crash
        out["B_error"] = repr(e)
        out["sec"] = time.time() - t0
        return out
    out["B_snap_radius"] = T.meta["snap_radius"]
    out["B_vertices"] = int(T.valid.sum())
    out["B_area_vx2"] = T.area() if T.valid.sum() > 4 else 0.0
    d0 = P.sheet_distance(T.points)
    if pipeline == "refine":
        Tr, _ = refine(T, V, iterations=2, sigma_vx=6.0, sheet_sigma=0.0, normal_sigma=0.0)
        nsig = 0.0
    else:
        Tr, frep = remove_folds(T)
        out["B_fold_removed_vertices"] = frep["removed_vertices"]
        Tr, _ = regularize(Tr, V)
        Tr, frep = fill_holes(Tr, V)
        out["B_filled_vertices"] = frep["filled_vertices"]
        nsig = NORMAL_SIGMA_VX
    out["B_kept_fraction"] = float(Tr.valid.sum() / max(T.valid.sum(), 1))
    d1 = P.sheet_distance(Tr.points)
    filled = Tr.filled_mask()
    ok = Tr.valid & ~filled  # filled vertices are not traced evidence: scored separately
    out["B_offsheet_rms_vx_traced"] = float(np.sqrt(np.nanmean(d0[T.valid] ** 2)))
    out["B_offsheet_rms_vx"] = float(np.sqrt(np.nanmean(d1[ok] ** 2)))
    out["B_offsheet_abs_p50_p90_vx"] = [float(v) for v in np.nanpercentile(np.abs(d1[ok]), [50, 90])]
    hist = lambda d: np.histogram(np.abs(d[np.isfinite(d)]), HIST_BINS)[0].tolist()  # for pooled percentiles
    out["B_offsheet_abs_hist_traced"], out["B_offsheet_abs_hist"] = hist(d0[T.valid]), hist(d1[ok])
    if filled.any():
        out["B_filled_offsheet_abs_p50_p90_vx"] = [float(v) for v in np.nanpercentile(np.abs(d1[filled]), [50, 90])]
        out["B_filled_offsheet_abs_hist"] = hist(d1[filled])
    # wrap jumps: adjacent vertices whose true arc length differs by far more than the grid step
    s_t = P.unrolled_coords(Tr.points)[..., 0]
    ju, jv = np.abs(np.diff(s_t, axis=1)) > 4 * step, np.abs(np.diff(s_t, axis=0)) > 4 * step
    out["B_wrap_jumps"] = int(ju.sum() + jv.sum())  # = B_cross_wrap_edges + B_tears
    wr = wrap_report(P, Tr.points, step)
    out["B_cross_wrap_edges"], out["B_tears"], out["B_fold_frac"] = wr["cross_wrap_edges"], wr["tears"], wr["fold_frac"]
    if pipeline != "refine":  # pinholes (interior holes <= the fill size) and fills next to a jump (must be 0)
        out["B_pinhole_vertices_traced"], out["B_pinhole_vertices"] = (
            int(z[(z > 0) & (z <= 4)].sum()) for z in (interior_holes(T.valid)[1], interior_holes(Tr.valid)[1]))
        out["B_filled_at_jump"] = int((ju & (filled[:, 1:] | filled[:, :-1])).sum() +
                                      (jv & (filled[1:] | filled[:-1])).sum())
    uvB = flatten(Tr)
    gtB = P.unrolled_coords(Tr.points)
    ok = Tr.valid
    pr = procrustes_rms(uvB[ok], gtB[ok])
    out["B_uv_rms_vx"], out["B_uv_rel_rms"] = pr["rms"], pr["rms_relative_to_extent"]
    out["B_flips"] = flatten_report(Tr, uvB)["flipped_triangles"]
    # render the unrolled sheet and compare to the text that was written into it
    F = flat_surface(Tr, uvB, out_scale=1.0)
    n = 4 * int(np.ceil(P.thickness)) + 1  # +-2 sheet thicknesses at 0.5 vx
    lstep = 0.5
    nrm = F.normals(nsig)
    st = render(F, V, n_layers=n, step=lstep, normals=nrm)
    tF = P.unrolled_coords(F.points)
    truth_px = P.ink_at(tF[..., 0], tF[..., 1])
    truth_px[~F.valid] = np.nan
    per = [ncc(l, truth_px) for l in st]
    best = int(np.nanargmax(per))
    out["B_ink_pixel_ncc"] = per[best]
    out["B_ink_layer_offset_vx"] = (best - (n - 1) / 2) * lstep
    out["B_ink_true_side"] = int(np.sign(P.ink_depth())) * _normal_frame(P, F, nrm)  # in the render's offset frame
    # ceiling: the ink sampled at its exact depth under every pixel (no geometry error at all)
    okF = F.valid & np.isfinite(tF[..., 0])
    s_, z_ = tF[okF, 0], tF[okF, 1]
    out.update(_normal_checks(P, F, tF, nrm, okF, s_, z_))
    dep = P.ink_depth(P._img(P.gap, s_, z_), s_) if P.two_layer else np.full(len(s_), P.ink_depth())
    ceil = np.full(F.shape, np.nan)
    ceil[okF] = V.sample(P.point(s_, z_, dep))
    out["B_ink_ceiling_ncc"] = ncc(ceil, truth_px)
    # the ink signal alone: same render through (volume - the same volume without ink)
    ink_gain, P.ink_gain = P.ink_gain, 0.0
    sig = ArrayVolume(a.astype(np.float32) - P.volume().astype(np.float32))
    P.ink_gain = ink_gain
    ss = render(F, sig, n_layers=n, step=lstep, normals=nrm)
    per_s = [ncc(l, truth_px) for l in ss]
    bs = int(np.nanargmax(per_s))
    out["B_ink_signal_pixel_ncc"] = per_s[bs]
    out["B_ink_signal_layer_offset_vx"] = (bs - (n - 1) / 2) * lstep
    out["B_ink_signal_layout_ncc"] = _layout_ncc(P, F, tF, ss[bs])
    if P.two_layer:
        # label-free: which face carries ink, decided from fiber orientation alone
        ax = axis_direction(F, up=P.R[:, 2])
        fp = fiber_profile(st, lstep, tile=None, axis=ax)
        out["B_fiber_ink_side"] = fp.get("ink_side")
        out["B_fiber_contrast"] = fp.get("recto_verso_contrast")
        # the true side is the inner face in every phantom (a constant answer would pass): the same stack with
        # the normal flipped, as on real segments whose normals point inward, must give the opposite call
        out["B_fiber_ink_side_flipped"] = fiber_profile(st[::-1], lstep, tile=None, axis=ax).get("ink_side")
        # layer hops: where the layers are separated, which layer did the trace follow?
        s_, z_, d_ = P._locate(Tr.points[ok & ~filled])
        g = P._img(P.gap, s_, z_)
        sep = g > 1.5
        if sep.sum() >= 20:
            side = np.sign(d_[sep])
            out["B_layer_minority_fraction"] = float(min((side > 0).mean(), (side < 0).mean()))
            # riding: nearer a lamina centre (|d| = T/4 + gap/2) than the sheet centre, where the laminae are apart
            lam = P.local_thickness(s_[sep]) / 4 + g[sep] / 2
            out["B_lamina_riding_fraction"] = float((np.abs(d_[sep]) > lam / 2).mean())
    # layout NCC: place the true text by ONE rigid transform of the flat image grid. Any residual
    # stretch/shear of the unrolling misplaces letters and lowers this score.
    out["B_ink_layout_ncc"] = _layout_ncc(P, F, tF, st[best])
    out["B_layout_quality_ncc"] = _layout_ncc(P, F, tF, truth_px)  # 1 = the unrolling places every letter right
    out["sec"] = time.time() - t0
    return out


def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def summarize(results: list[dict]) -> dict:
    n = len(results)
    summ = {"n_phantoms": n}
    keys = ["A_uv_rms_vx", "B_offsheet_rms_vx_traced", "B_offsheet_rms_vx", "B_uv_rms_vx", "B_uv_rel_rms",
            "B_ink_pixel_ncc", "B_ink_layout_ncc", "B_ink_ceiling_ncc", "B_ink_signal_pixel_ncc",
            "B_ink_signal_layout_ncc", "B_layout_quality_ncc", "B_area_vx2", "B_snap_radius", "B_fiber_contrast",
            "B_layer_minority_fraction", "B_lamina_riding_fraction", "B_filled_vertices", "B_fold_removed_vertices",
            "B_kept_fraction", "B_pinhole_vertices_traced", "B_pinhole_vertices", "B_fold_frac", "sec"]
    for k in keys:
        v = np.array([r[k] for r in results if k in r and r[k] is not None], float)
        if v.size:
            summ[k] = {"median": float(np.median(v)), "p5": float(np.percentile(v, 5)),
                       "p95": float(np.percentile(v, 95)), "worst": float(v.max() if "ncc" not in k else v.min())}
    checks = {
        "A_uv_rms": lambda r: r["A_uv_rms_vx"] < PASS["A_uv_rms_vx"] and r["A_flips"] == 0,
        "B_completed": lambda r: "B_error" not in r,
        "B_on_sheet": lambda r: r.get("B_offsheet_rms_vx", 9e9) < PASS["B_offsheet_rms_vx"],
        "B_no_wrap_jump": lambda r: r.get("B_wrap_jumps", 1) <= PASS["B_wrap_jumps"],
        "B_no_cross_wrap": lambda r: r.get("B_cross_wrap_edges", 1) == 0,
        "B_geometry": lambda r: r.get("B_uv_rel_rms", 1) < PASS["B_uv_rel_rms"] and r.get("B_flips", 1) == 0,
        "B_ink_layout": lambda r: r.get("B_ink_layout_ncc", -1) > PASS["B_ink_layout_ncc"],
        "B_ink_signal_layout": lambda r: r.get("B_ink_signal_layout_ncc", -1) > PASS["B_ink_signal_layout_ncc"],
        "B_ink_correct_face": lambda r: np.sign(r.get("B_ink_layer_offset_vx", 0)) == r.get("B_ink_true_side", 9),
        "B_fiber_finds_ink_side": lambda r: r.get("B_fiber_ink_side", None) == r.get("B_ink_true_side", 9),
        "B_fiber_flipped_normal": lambda r: r.get("B_fiber_ink_side_flipped", None) == -r.get("B_ink_true_side", 9),
        "B_fiber_calls_ink_side": lambda r: r.get("B_fiber_ink_side", None) is not None,  # no-call fails the 2 above
    }
    for sfx in ("", "_all"):  # render normals against the true sheet normal, pooled over all pixels
        k = sum(r.get(f"B_render_normal_reversed{sfx}_px", 0) for r in results)
        npx = sum(r.get(f"B_render_normal{sfx}_px", 0) for r in results)
        summ[f"B_render_normal_reversed{sfx}_frac"] = k / max(npx, 1)
    summ["pass_rate"] = {}
    for name, f in checks.items():
        k = sum(bool(f(r)) for r in results)
        lo, hi = _wilson(k, n)
        summ["pass_rate"][name] = {"passed": k, "of": n, "rate": k / n if n else None, "ci95": [lo, hi]}
    summ["failures"] = {name: [r["seed"] for r in results if not f(r)][:20] for name, f in checks.items()}
    summ["thresholds"] = PASS
    return summ


def run(n: int = 100, start: int = 0, workers: int = 4, out: str | None = None, log=print, **kw) -> dict:
    results = []
    from ._kernels import limit_threads

    # each process gets cpu_count // workers numba threads (no oversubscription)
    with ProcessPoolExecutor(workers, initializer=limit_threads, initargs=(workers,)) as ex:
        futs = [ex.submit(run_one, s, **kw) for s in range(start, start + n)]
        for i, f in enumerate(futs):
            results.append(f.result())
            if log and (i + 1) % 10 == 0:
                log(f"bench: {i + 1}/{n} phantoms")
    summ = summarize(results)
    if out:
        with open(out, "w") as fh:
            json.dump({"summary": summ, "results": results}, fh, indent=1, default=float)
    return summ
