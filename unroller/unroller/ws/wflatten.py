"""Whole-scroll flatten (WP9 S8): a coarse global map of the fused chart, then soft-anchored fine tiles.

* ``flatten_coarse``: the whole chart at a coarse step (e.g. 64 vx), initialised from the chart's own (u, v)
  (column, row) layout, insets (pieces with a known flattening) pinned by stiff anchors.
* ``flatten_tiled``: tiles of ``tile`` cells plus a ``margin`` on every side, each flattened with ARAP
  soft-anchored to the coarse map (bilinearly upsampled), blended with an EDT partition of unity (weights fall
  to 0 at a tile's outer edge).  The seam metric is the UV disagreement of overlapping tiles *before* blending.

UV is in voxels.  The fused chart's cells are ``fg.chart['step']`` vx apart.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..flatten import _clean, flatten, triangle_singular_values, _grid_folds
from ..surface import Surface, bilerp_grid

# FusedGrid.status
REAL, VIRTUAL, EMPTY = 0, 1, 2


@dataclass
class FusedGrid:
    """Minimal stand-in for WP8's fused chart (rows follow the scroll axis, columns the unrolled spiral)."""
    xyz: np.ndarray  # (R, C, 3) f32, NaN where empty
    H: np.ndarray  # (R, C) f32 continuous winding
    normal: np.ndarray = None  # (R, C, 3) f16, outward
    support: np.ndarray = None  # u8
    spread: np.ndarray = None  # f16
    conf: np.ndarray = None  # f16
    flags: np.ndarray = None  # u32
    status: np.ndarray = None  # u8: REAL / VIRTUAL / EMPTY
    chart: dict = field(default_factory=lambda: {"step": 1.0})
    z0: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        R, C = self.xyz.shape[:2]
        ok = np.isfinite(self.xyz[..., 0])
        if self.status is None:
            self.status = np.where(ok, REAL, EMPTY).astype(np.uint8)
        for name, dt, shape in (("support", np.uint8, (R, C)), ("spread", np.float16, (R, C)),
                                ("conf", np.float16, (R, C)), ("flags", np.uint32, (R, C))):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(shape, dt))
        if self.normal is None:
            self.normal = Surface(self.xyz, (1.0, 1.0)).normals().astype(np.float16)

    @property
    def shape(self):
        return self.xyz.shape[:2]

    @property
    def step(self):
        return float(self.chart.get("step", 1.0))

    def to_surface(self, exclude=VIRTUAL) -> Surface:
        pts = self.xyz.astype(np.float32).copy()
        drop = self.status == EMPTY
        if exclude is not None:
            drop |= self.status == exclude
        pts[drop] = np.nan
        return Surface(pts, (1.0 / self.step, 1.0 / self.step), {"source": "fused", **self.meta})

    def save(self, path):
        import json
        np.savez(path, xyz=self.xyz, H=self.H, normal=self.normal, support=self.support, spread=self.spread,
                 conf=self.conf, flags=self.flags, status=self.status, z0=self.z0,
                 chart=json.dumps(self.chart), meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        import json
        a = np.load(path)
        return cls(a["xyz"], a["H"], a["normal"], a["support"], a["spread"], a["conf"], a["flags"], a["status"],
                   json.loads(str(a["chart"])), float(a["z0"]), json.loads(str(a["meta"])))


def _as_grid(fg):
    if isinstance(fg, (str, bytes)) or hasattr(fg, "__fspath__"):
        return FusedGrid.load(fg)
    return fg


def chart_uv(shape, step):
    """The chart's own layout: u = column * step, v = row * step (vx)."""
    R, C = shape
    rr, cc = np.mgrid[0:R, 0:C]
    return np.stack([cc * step, rr * step], -1).astype(np.float64)


def flatten_coarse(fused, step=64.0, insets=None, inset_weight=1e4, gauge_weight=1e-4, **kw):
    """Coarse global flatten of the fused chart at ``step`` vx.  Returns (uv (R', C', 2) for the strided grid
    fg.xyz[::k, ::k], k = step / fg.step, info).

    The chart is usually in many pieces (tears, missing papyrus, gaps at wrap ends): every piece is flattened,
    each held in place by weak gauge anchors (``gauge_weight``) to the chart's own (u, v) = (column, row) * step
    layout, which places the pieces consistently along the spiral; the anchors are far weaker than the ARAP
    stiffness, so the shape of each piece is its own.  ``insets``: list of (mask (R, C) bool, uv (R, C, 2)) on the
    full chart whose vertices are pinned (stiff anchors, ``inset_weight``).  Virtual cells take part (the coarse
    map must span the gaps)."""
    fg = _as_grid(fused)
    k = max(1, int(round(step / fg.step)))
    S = fg.to_surface(exclude=None)
    Sk = Surface(S.points[::k, ::k], (S.scale[0] / k, S.scale[1] / k))
    init = chart_uv(fg.shape, fg.step)[::k, ::k]
    mask = Sk.valid.copy()
    target = init.copy()
    vw = np.ones(Sk.shape)
    if insets:
        for mk, uv in insets:
            m = mk[::k, ::k] & mask
            target[m] = uv[::k, ::k][m]
            vw[m] = inset_weight / gauge_weight
    info = {}
    t = time.perf_counter()
    uv = flatten(Sk, init_uv=init, anchor_mask=mask, anchor_uv=target, anchor_weight=gauge_weight, vertex_weight=vw,
                 info=info, **kw)
    info.update({"stride": k, "step": k * fg.step, "vertices": int(Sk.valid.sum()), "wall_s": time.perf_counter() - t,
                 "flattened_frac": float(np.isfinite(uv[..., 0]).sum() / max(Sk.valid.sum(), 1))})
    return uv, info


def upsample(coarse_uv, shape, k):
    """Bilinear upsampling of a [::k]-strided map to ``shape``; the strided grid stops short of the last
    rows / columns, so it is linearly extrapolated by one row and column first (NaN cells: nearest valid)."""
    R, C = shape
    c = np.asarray(coarse_uv, np.float64)
    bad = ~np.isfinite(c[..., 0])
    if bad.any() and (~bad).any():
        _, (ii, jj) = ndimage.distance_transform_edt(bad, return_indices=True)
        c = c[ii, jj]
    if c.shape[0] > 1:
        c = np.concatenate([c, 2 * c[-1:] - c[-2:-1]], 0)
    if c.shape[1] > 1:
        c = np.concatenate([c, 2 * c[:, -1:] - c[:, -2:-1]], 1)
    out = bilerp_grid(c, np.arange(R) / k, np.arange(C) / k)
    return out


def coarse_stride(fine_shape, coarse_shape):
    """The stride k of a [::k] sub-grid: the smallest k with ceil(R / k) = R' and ceil(C / k) = C'."""
    (R, C), (Rc, Cc) = fine_shape, coarse_shape[:2]
    for k in range(1, max(R, C) + 1):
        if -(-R // k) == Rc and -(-C // k) == Cc:
            return k
    raise ValueError(f"{coarse_shape} is not a strided sub-grid of {fine_shape}")


def flatten_tiled(fused, coarse_uv, tile=2048, margin=128, anchor_weight=1.0, virtual_weight=0.01, k=None,
                  log=None, **kw):
    """Fine flatten in tiles soft-anchored to the coarse map.  Returns (UV (R, C, 2) f32, info) with the seam
    statistics (p50 / p95 / max of the UV disagreement of overlapping tiles before blending), per-tile timings,
    convergence and flip counts.  ``k``: stride of ``coarse_uv`` on the fused grid (inferred from shapes)."""
    fg = _as_grid(fused)
    R, C = fg.shape
    if k is None:
        k = coarse_stride((R, C), coarse_uv.shape)
    up = upsample(coarse_uv, (R, C), k).astype(np.float32)
    S = fg.to_surface(exclude=None)
    vw = np.where(fg.status == VIRTUAL, virtual_weight, 1.0).astype(np.float32)
    acc = np.zeros((R, C, 2))
    wsum = np.zeros((R, C))
    seam = []
    tiles = []
    t_all = time.perf_counter()
    for y0 in range(0, R, tile):
        for x0 in range(0, C, tile):
            ya, yb = max(0, y0 - margin), min(R, y0 + tile + margin)
            xa, xb = max(0, x0 - margin), min(C, x0 + tile + margin)
            sub = Surface(S.points[ya:yb, xa:xb], S.scale)
            if sub.valid.sum() < 4:
                continue
            ok = sub.valid & np.isfinite(up[ya:yb, xa:xb, 0])
            info = {}
            t = time.perf_counter()
            U = flatten(sub, init_uv=np.where(ok[..., None], up[ya:yb, xa:xb], np.nan), anchor_mask=ok,
                        anchor_uv=up[ya:yb, xa:xb], anchor_weight=anchor_weight, vertex_weight=vw[ya:yb, xa:xb],
                        info=info, **kw)
            dt = time.perf_counter() - t
            # partition of unity: distance to the tile's outer edge (where it was cut), 0 at the edge
            inner = np.ones((yb - ya, xb - xa), bool)
            edge_cut = np.zeros_like(inner)
            if ya > 0: edge_cut[0, :] = True
            if yb < R: edge_cut[-1, :] = True
            if xa > 0: edge_cut[:, 0] = True
            if xb < C: edge_cut[:, -1] = True
            w = ndimage.distance_transform_edt(~edge_cut) if edge_cut.any() else np.full(inner.shape, 1e6)
            w = w * np.isfinite(U[..., 0])
            ov = (wsum[ya:yb, xa:xb] > 0) & (w > 0)
            if ov.any():  # disagreement with the tiles already placed (their blend so far), before blending
                prev = acc[ya:yb, xa:xb][ov] / wsum[ya:yb, xa:xb][ov][:, None]
                seam.append(np.linalg.norm(U[ov] - prev, axis=1))
            Uz = np.nan_to_num(U)
            acc[ya:yb, xa:xb] += Uz * w[..., None]
            wsum[ya:yb, xa:xb] += w
            tiles.append({"y0": y0, "x0": x0, "vertices": int(sub.valid.sum()), "wall_s": dt,
                          "converged": info.get("converged"), "iterations": info["levels"][-1]["iterations"]})
            if log:
                log(f"tile ({y0},{x0}): {tiles[-1]['vertices']} vertices, {dt:.1f} s")
    UV = np.full((R, C, 2), np.nan, np.float32)
    m = wsum > 0
    UV[m] = (acc[m] / wsum[m][:, None]).astype(np.float32)
    UV[~S.valid] = np.nan
    s = np.concatenate(seam) if seam else np.zeros(0)
    info = {"tiles": tiles, "n_tiles": len(tiles), "k": k, "wall_s": time.perf_counter() - t_all,
            "seam_p50": float(np.percentile(s, 50)) if len(s) else 0.0,
            "seam_p95": float(np.percentile(s, 95)) if len(s) else 0.0,
            "seam_max": float(s.max()) if len(s) else 0.0, "seam_pixels": int(len(s)),
            "all_converged": all(t["converged"] for t in tiles), "flips": flips(S, UV)}
    return UV, info


def flips(S: Surface, uv) -> int:
    """Triangles whose UV orientation disagrees with the majority (grid-folded triangles judged in their
    mirrored frames, as flatten_report does)."""
    idx, V, T = S.triangles()
    T = _clean(V, T)
    u = uv[idx >= 0]
    ok = np.all(np.isfinite(u[T]).reshape(len(T), -1), axis=1)
    if not ok.any():
        return 0
    _, det = triangle_singular_values(V, T[ok], u.astype(np.float64), _grid_folds(S, idx, V, T[ok]))
    return int((np.sign(det) != np.sign(np.median(det))).sum())


def window_procrustes(uv, ref, valid, win, step):
    """Rigid (rotation + translation) residual of uv against ref in windows of ``win`` vx: per-window rms and
    max (vx).  Measures local shape agreement independent of the global placement."""
    R, C = valid.shape
    n = max(1, int(round(win / step)))
    out = []
    for y in range(0, R, n):
        for x in range(0, C, n):
            m = valid[y:y + n, x:x + n] & np.isfinite(uv[y:y + n, x:x + n, 0]) & np.isfinite(ref[y:y + n, x:x + n, 0])
            if m.sum() < 8:
                continue
            a = uv[y:y + n, x:x + n][m].astype(np.float64)
            b = ref[y:y + n, x:x + n][m].astype(np.float64)
            a0, b0 = a - a.mean(0), b - b.mean(0)
            U, _, Vt = np.linalg.svd(a0.T @ b0)
            D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
            d = np.linalg.norm(a0 @ (U @ D @ Vt) - b0, axis=1)
            out.append((float(np.sqrt((d ** 2).mean())), float(d.max())))
    return np.array(out)
