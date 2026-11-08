"""Phantom benchmark with explicit pass/fail thresholds.

Every check compares against ground truth that was *put into* the synthetic scroll, so a
pass means the geometry and the ink came back, not that the output merely looks plausible.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from .flatten import flatten
from .metrics import flatten_report, ncc, procrustes_rms, stack_report
from .phantom import SpiralScroll
from .render import render
from .resample import flat_surface
from .surface import Surface
from .volume import Volume, write_ome_zarr

THRESH = {
    "uv_rms_vx": 0.10,           # flattened layout vs true arc-length coords
    "holes_uv_rms_vx": 0.10,     # same, with ~15% of the segment missing in blobs
    "ink_ncc": 0.85,             # render vs injected ink, best layer
    "sheet_offset_err_vx": 0.30,  # detection of a deliberate 2-voxel misplacement
    "dome_l2_stretch": 1.02,     # non-developable cap: distortion must stay near theoretical minimum
}


def run(log=print, workdir=None) -> dict:
    res, checks = {}, {}
    P = SpiralScroll()
    work = workdir or tempfile.mkdtemp(prefix="unroller_validate_")
    zpath = os.path.join(work, "phantom.zarr")
    write_ome_zarr(zpath, P.volume(), levels=2)
    V = Volume(zpath)

    # 1. flattening accuracy on a deliberately non-isometric (uniform-angle) grid
    S = P.surface()
    gt = P.unrolled_coords(S.points).reshape(-1, 2)
    raw = np.stack(np.mgrid[0:S.shape[0], 0:S.shape[1]][::-1], -1).reshape(-1, 2) / S.scale[0]
    uv = flatten(S)
    res["raw_grid_rms_vx"] = procrustes_rms(raw, gt)["rms"]
    res["uv"] = procrustes_rms(uv.reshape(-1, 2), gt)
    res["flatten"] = flatten_report(S, uv)
    checks["uv_rms"] = res["uv"]["rms"] < THRESH["uv_rms_vx"]
    checks["no_flips"] = res["flatten"]["flipped_triangles"] == 0
    log(f"flatten: raw grid rms {res['raw_grid_rms_vx']:.2f} vx -> unrolled rms {res['uv']['rms']:.4f} vx")

    # 2. handedness: the flat grid keeps the input's orientation (text is never mirrored)
    F = flat_surface(S, uv, out_scale=1.0)
    tF = P.unrolled_coords(F.points)
    rr, cc = np.nonzero(F.valid)
    A = np.c_[cc, rr, np.ones_like(cc)]
    coef, *_ = np.linalg.lstsq(A, tF[rr, cc], rcond=None)
    s_in = np.sign(np.linalg.det(np.linalg.lstsq(np.c_[raw, np.ones(len(raw))], gt, rcond=None)[0][:2]))
    res["handedness_preserved"] = bool(np.sign(np.linalg.det(coef[:2])) == s_in)
    checks["handedness"] = res["handedness_preserved"]

    # 3. ink recovery: render the flat surface, compare with the injected ink per pixel
    n = 9
    st = render(F, V, n_layers=n)
    truth = P.ink_at(tF[..., 0], tF[..., 1])
    truth[~F.valid] = np.nan
    res["ink_ncc_per_layer"] = [round(ncc(l, truth), 4) for l in st]
    best = int(np.argmax(res["ink_ncc_per_layer"]))
    res["ink_best_layer_offset_vx"] = best - (n - 1) / 2
    checks["ink_ncc"] = max(res["ink_ncc_per_layer"]) > THRESH["ink_ncc"]
    checks["ink_on_correct_face"] = abs(res["ink_best_layer_offset_vx"] - P.ink_offset) <= 0.5
    log(f"ink: best NCC {max(res['ink_ncc_per_layer']):.3f} at layer offset {res['ink_best_layer_offset_vx']:+.0f} vx "
        f"(injected at {P.ink_offset:+.0f})")

    # 4. sheet-centring metric detects a known misplacement
    nrm = F.normals()
    Fm = Surface(F.points + 2.0 * nrm, F.scale)
    rep = stack_report(render(Fm, V, n_layers=11))
    res["sheet_offset_measured_vx"] = rep["peak_offset_vx"]
    rep0 = stack_report(render(F, V, n_layers=11))
    res["sheet_offset_baseline_vx"] = rep0["peak_offset_vx"]
    checks["sheet_offset"] = abs(rep["peak_offset_vx"] - rep0["peak_offset_vx"] + 2.0) < THRESH["sheet_offset_err_vx"]
    log(f"sheet centring: baseline {rep0['peak_offset_vx']:+.2f} vx, displaced +2 -> measured {rep['peak_offset_vx']:+.2f} vx")

    # 5. holes: knock out blobs of the segment, flattening must stay accurate
    rng = np.random.default_rng(1)
    pts = S.points.copy()
    H, W = S.shape
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(14):
        cy, cx, r = rng.uniform(0, H), rng.uniform(0, W), rng.uniform(8, 15)
        pts[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = np.nan
    Sh = Surface(pts, S.scale)
    uvh = flatten(Sh)
    gth = P.unrolled_coords(Sh.points).reshape(-1, 2)
    res["holes_missing_fraction"] = float(1 - Sh.valid.mean())
    res["holes_uv"] = procrustes_rms(uvh.reshape(-1, 2), gth)
    checks["holes_uv_rms"] = res["holes_uv"]["rms"] < THRESH["holes_uv_rms_vx"]
    log(f"holes ({res['holes_missing_fraction']:.0%} missing): rms {res['holes_uv']['rms']:.4f} vx")

    # 6. non-developable dome: no isometry exists; check no folds and bounded stretch
    R = 400.0
    v, u = np.mgrid[-60:61, -60:61].astype(float)
    dome = np.stack([u, v, np.sqrt(R * R - u * u - v * v)], -1)
    D = Surface(dome, (1.0, 1.0))
    rd = flatten_report(D, flatten(D))
    res["dome"] = rd
    checks["dome_no_flips"] = rd["flipped_triangles"] == 0
    checks["dome_stretch"] = rd["l2_stretch"] < THRESH["dome_l2_stretch"]

    res["thresholds"] = THRESH
    res["checks"] = checks
    res["passed"] = all(checks.values())
    log("PASS" if res["passed"] else f"FAIL: {[k for k, v in checks.items() if not v]}")
    return res
