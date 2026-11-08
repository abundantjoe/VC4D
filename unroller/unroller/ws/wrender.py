"""Whole-scroll render in chart space (WP9 S9) and the warp of chart-space images to the flattened UV space.

* ``render_chart``: render the fused chart surface (all cells, virtual ones included and marked) tile by
  tile: a 21-layer mean composite, the max over the +-10 layers, the mean layer profile (stack_report: its
  peak offset tells how well the surface sits on the sheet centre) and a status channel; 65-layer stacks
  only for ROIs.
* ``warp_to_uv``: the UV-space product without CT access - the chart grid's own coordinates are rasterised
  over the flattened triangles and the chart-space images are sampled there bilinearly.
* ``cut_wraps``: per-wrap tifxyz (and UV) cut out of the chart by solved winding H.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..metrics import stack_report
from ..render import layer_offsets, render
from ..resample import rasterize
from ..surface import Surface, bilerp_grid
from .wflatten import EMPTY, VIRTUAL, FusedGrid, _as_grid


def render_chart(fused, vol, out_dir, n_layers=21, roi=None, roi_layers=65, workers=2, scale=1.0, tile=512,
                 step=1.0):
    """Render the chart surface at ``scale`` px / vx.  Writes out_dir/{mean,max,status}.npy (chart pixels) and,
    for ``roi`` = (x0, y0, w, h) in chart pixels, out_dir/roi_stack.npy (roi_layers layers).  Returns a dict
    (shapes, the mean layer profile, its stack_report and timing)."""
    fg = _as_grid(fused)
    os.makedirs(out_dir, exist_ok=True)
    S = fg.to_surface(exclude=None)
    nrm = S.normals()
    sx, sy = S.scale
    H0, W0 = S.shape
    W = int(np.floor((W0 - 1) / sx * scale)) + 1
    H = int(np.floor((H0 - 1) / sy * scale)) + 1
    mean = np.full((H, W), np.nan, np.float32)
    mx = np.full((H, W), np.nan, np.float32)
    prof_sum = np.zeros(n_layers)
    prof_cnt = np.zeros(n_layers)
    tiles = [(y, x) for y in range(0, H, tile) for x in range(0, W, tile)]
    t0 = time.perf_counter()

    def work(t):
        y, x = t
        h, w = min(tile, H - y), min(tile, W - x)
        st = render(S, vol, n_layers, step, scale, crop=(x, y, w, h), normals=nrm)
        ok = np.isfinite(st)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(np.where(ok, st, np.nan), 0) if ok.any() else np.full((h, w), np.nan)
            mm = np.nanmax(np.where(ok, st, -np.inf), 0)
        mm[~np.isfinite(mm)] = np.nan
        return y, x, h, w, m.astype(np.float32), mm.astype(np.float32), np.where(ok, st, 0).sum((1, 2)), ok.sum((1, 2))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with ThreadPoolExecutor(workers) as ex:
            for y, x, h, w, m, mm, ps, pc in ex.map(work, tiles):
                mean[y:y + h, x:x + w] = m
                mx[y:y + h, x:x + w] = mm
                prof_sum += ps
                prof_cnt += pc
    # status channel at chart pixels (nearest cell)
    rr = np.clip(np.rint(np.arange(H) / scale * sy).astype(int), 0, H0 - 1)
    cc = np.clip(np.rint(np.arange(W) / scale * sx).astype(int), 0, W0 - 1)
    status = fg.status[rr][:, cc]
    np.save(os.path.join(out_dir, "mean.npy"), mean)
    np.save(os.path.join(out_dir, "max.npy"), mx)
    np.save(os.path.join(out_dir, "status.npy"), status)
    with np.errstate(invalid="ignore"):
        prof = prof_sum / prof_cnt
    rep = stack_report(prof[:, None, None], step)
    out = {"shape": [H, W], "n_layers": n_layers, "profile": prof.tolist(), "stack_report": rep,
           "virtual_px_frac": float((status == VIRTUAL).mean()), "wall_s": time.perf_counter() - t0,
           "scale": scale, "chart_step": fg.step}
    if roi is not None:
        st = render(S, vol, roi_layers, step, scale, crop=tuple(roi), normals=nrm)
        np.save(os.path.join(out_dir, "roi_stack.npy"), st)
        out["roi"] = list(roi)
    json.dump(out, open(os.path.join(out_dir, "render_chart.json"), "w"), indent=1, default=float)
    return out


def _bilerp_points(img, r, c):
    H, W = img.shape
    out = np.full(r.shape, np.nan, np.float32)
    ok = np.isfinite(r) & np.isfinite(c) & (r >= 0) & (c >= 0) & (r <= H - 1) & (c <= W - 1)
    r, c = r[ok], c[ok]
    r0 = np.minimum(np.floor(r).astype(int), H - 2) if H > 1 else np.zeros(len(r), int)
    c0 = np.minimum(np.floor(c).astype(int), W - 2) if W > 1 else np.zeros(len(c), int)
    fr, fc = r - r0, c - c0
    r1, c1 = np.minimum(r0 + 1, H - 1), np.minimum(c0 + 1, W - 1)
    v = (img[r0, c0] * (1 - fr) * (1 - fc) + img[r0, c1] * (1 - fr) * fc + img[r1, c0] * fr * (1 - fc)
         + img[r1, c1] * fr * fc)
    out[ok] = v
    return out


def warp_to_uv(chart_img, fused, uv, out=None, scale=1.0, chart_scale=1.0, origin=None, shape=None):
    """Warp a chart-space image (``chart_img``: array or .npy path, rendered at ``chart_scale`` px / vx of the
    chart surface) to UV space at ``scale`` px / vx, through the flatten map ``uv`` (R, C, 2).  No CT access:
    the chart grid coordinates are rasterised over the UV triangles, then the image is sampled there.
    Returns (image, dict(origin, shape)); writes ``out`` (.npy) if given."""
    fg = _as_grid(fused)
    img = np.load(chart_img) if isinstance(chart_img, str) else np.asarray(chart_img)
    S = Surface(np.where(np.isfinite(uv[..., :1]), fg.xyz, np.nan).astype(np.float32), (1.0 / fg.step,) * 2)
    idx, V, T = S.triangles()
    R, C = fg.shape
    rr, cc = np.nonzero(idx >= 0)
    gridc = np.stack([cc, rr, np.zeros_like(rr)], 1).astype(np.float64)  # chart grid coordinates as "positions"
    u = uv[idx >= 0].astype(np.float64)
    if origin is None:
        origin = np.nanmin(u, 0)
    G = rasterize(gridc, T, u, 1.0 / scale, origin=origin, shape=shape)  # (h, w, 3): chart (col, row, 0)
    # chart pixel = grid coordinate * cells-to-pixels (chart surface: step vx per cell, chart_scale px per vx)
    f = fg.step * chart_scale
    warped = _bilerp_points(img.astype(np.float32), G[..., 1] * f, G[..., 0] * f)
    info = {"origin": [float(o) for o in origin], "shape": list(warped.shape), "scale": scale}
    if out is not None:
        np.save(out, warped)
    return warped, info


def direct_uv_render(fused, vol, uv, n_layers=21, scale=1.0, step=1.0, origin=None, shape=None):
    """Reference for the warp: resample the 3D surface on the UV grid and render there (needs the CT)."""
    fg = _as_grid(fused)
    S = Surface(np.where(np.isfinite(uv[..., :1]), fg.xyz, np.nan).astype(np.float32), (1.0 / fg.step,) * 2)
    idx, V, T = S.triangles()
    u = uv[idx >= 0].astype(np.float64)
    if origin is None:
        origin = np.nanmin(u, 0)
    P = rasterize(V, T, u, 1.0 / scale, origin=origin, shape=shape)
    F = Surface(P.astype(np.float32), (scale, scale))
    st = render(F, vol, n_layers, step, 1.0, normals=F.normals())  # raw normals, as render_chart; never the default
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(st, 0), np.nanmax(st, 0)


def ncc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())), int(m.sum())


def check_handedness(fused, outward_xyz=None):
    """Fraction of cells whose surface normal du x dv (u = column, v = row) agrees with the outward direction
    (fg.normal, or the radial direction from ``outward_xyz`` = a point on the scroll axis)."""
    fg = _as_grid(fused)
    P = fg.xyz.astype(np.float64)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    if outward_xyz is not None:
        ax = np.asarray(outward_xyz, float)
        out = P - ax
        out[..., 2] = 0
    else:
        out = fg.normal.astype(np.float64)
    dot = (n * out).sum(-1)
    ok = np.isfinite(dot) & (fg.status != EMPTY)
    return float(np.mean(dot[ok] > 0))


def cut_wraps(fused, uv, out_dir=None):
    """Per-wrap tifxyz cut from the chart by solved winding H (wrap w = floor(H)): points outside the wrap are
    NaN and the grid is cropped to the wrap's column range.  Returns {w: dict(cols, cells, surface, uv)}; with
    ``out_dir`` each wrap is saved as out_dir/wrap_WWW (tifxyz) plus uv.npy."""
    fg = _as_grid(fused)
    W = np.floor(fg.H).astype(np.int64)
    valid = np.isfinite(fg.xyz[..., 0]) & np.isfinite(fg.H)
    out = {}
    for w in np.unique(W[valid]):
        m = valid & (W == w)
        cols = np.flatnonzero(m.any(0))
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        pts = np.where(m[:, c0:c1, None], fg.xyz[:, c0:c1], np.nan).astype(np.float32)
        S = Surface(pts, (1.0 / fg.step,) * 2, {"source": "cut_wraps", "wrap": int(w), "cols": [c0, c1]})
        u = np.where(m[:, c0:c1, None], uv[:, c0:c1], np.nan).astype(np.float32)
        out[int(w)] = {"cols": [c0, c1], "cells": int(m.sum()), "surface": S, "uv": u}
        if out_dir is not None:
            d = os.path.join(out_dir, f"wrap_{int(w):03d}")
            S.save(d)
            np.save(os.path.join(d, "uv.npy"), u)
    return out
