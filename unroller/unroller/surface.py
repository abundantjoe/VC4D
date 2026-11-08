"""tifxyz surfaces (VC3D QuadSurface on-disk format).

A segment directory holds x.tif, y.tif, z.tif (float32, rows = v, cols = u) giving the
level-0 voxel position of each grid vertex, plus meta.json with ``scale`` = grid cells per
voxel (e.g. 0.05 => one vertex every 20 voxels).  On disk, invalid vertices are (-1,-1,-1);
in memory we use NaN so that invalid data can never silently leak into arithmetic.
"""
from __future__ import annotations

import json
import os
import uuid as _uuid

import numpy as np
import tifffile

NORMAL_SIGMA_VX = 6.0  # render normals: grid smoothed over this many voxels, chosen on phantoms (README "Render path")


class Surface:
    def __init__(self, points: np.ndarray, scale=(1.0, 1.0), meta: dict | None = None):
        pts = np.asarray(points, dtype=np.float32).copy()
        if pts.ndim != 3 or pts.shape[2] != 3:
            raise ValueError("points must be (rows, cols, 3)")
        bad = ~np.all(np.isfinite(pts), axis=2) | np.all(pts == -1, axis=2)
        pts[bad] = np.nan
        self.points = pts
        self.scale = (float(scale[0]), float(scale[1]))
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str) -> "Surface":
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        xyz = [tifffile.imread(os.path.join(path, f"{c}.tif")).astype(np.float32) for c in "xyz"]
        pts = np.stack(xyz, axis=-1)
        mask_p = os.path.join(path, "mask.tif")
        if os.path.exists(mask_p):
            m = tifffile.imread(mask_p)
            if m.ndim == 3:
                m = m[..., 0] if m.shape[-1] < m.shape[0] else m[0]
            if m.shape == pts.shape[:2]:
                pts[m == 0] = np.nan
        sc = meta.get("scale", [1.0, 1.0])
        return cls(pts, (sc[0], sc[1]), meta)

    def save(self, path: str, **extra_meta):
        os.makedirs(path, exist_ok=True)
        disk = np.where(np.isnan(self.points), -1.0, self.points).astype(np.float32)
        for i, c in enumerate("xyz"):
            tifffile.imwrite(os.path.join(path, f"{c}.tif"), disk[..., i])
        v = self.points[self.valid]
        meta = dict(self.meta)
        meta.update({"type": "seg", "format": "tifxyz", "scale": list(self.scale),
                     "uuid": meta.get("uuid") or os.path.basename(os.path.normpath(path)) or str(_uuid.uuid4())})
        if len(v):
            meta["bbox"] = [v.min(0).tolist(), v.max(0).tolist()]
        meta.update(extra_meta)
        tmp = os.path.join(path, "meta.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, os.path.join(path, "meta.json"))  # atomic, like VC3D

    # ------------------------------------------------------------ geometry
    @property
    def shape(self):
        return self.points.shape[:2]

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.points[..., 0])

    def filled_mask(self) -> np.ndarray:
        """Vertices interpolated into holes (regularize.fill_holes, meta["filled"]), not traced: never evidence."""
        m = np.zeros(self.shape, bool)
        f = self.meta.get("filled")
        if f and tuple(f.get("grid", ())) == tuple(self.shape):
            m.flat[np.asarray(f["flat_index"], np.int64)] = True
        return m

    def normals(self, sigma: float = 0.0) -> np.ndarray:
        """Unit normals n = normalize(dP/du x dP/dv) (VC3D orientation).

        ``sigma`` = 0 (raw): central differences where both neighbours are valid, one-sided otherwise; NaN if
        a direction has no valid neighbour.  Compiled kernel, bit-identical to _normals_reference.
        ``sigma`` > 0: the same on the grid smoothed by a masked gaussian of ``sigma`` voxels (smoothed_points),
        so vertex jitter along the normal cannot tilt or reverse them; NaN at invalid vertices.
        """
        from . import _kernels as K

        if sigma > 0:
            ps = self.smoothed_points(sigma, centred=True)  # normals do not depend on the origin
            out = np.empty_like(ps)
            K.call(K.grid_normals, out.shape[0] * out.shape[1], ps, out)
            out[~self.valid] = np.nan
            return out
        out = np.empty_like(self.points)
        K.call(K.grid_normals, out.shape[0] * out.shape[1], self.points, out)
        return out

    def smoothed_points(self, sigma: float, centred: bool = False) -> np.ndarray:
        """Masked (normalised) gaussian of the vertex positions, ``sigma`` in voxels (sigma * scale grid cells
        per axis).  Holes do not pull positions: only valid vertices are averaged; defined wherever a valid
        vertex lies within ~3 sigma (so holes and borders get a smooth continuation), NaN beyond."""
        from scipy.ndimage import gaussian_filter

        ok = self.valid
        if not ok.any():
            return np.full(self.points.shape, np.nan, np.float32)
        c = self.points[ok].mean(0, dtype=np.float64)  # filter coordinates relative to the centre (float32 precision)
        sg = (max(sigma * self.scale[1], 1e-3), max(sigma * self.scale[0], 1e-3))
        w = gaussian_filter(ok.astype(np.float32), sg, truncate=3.0)
        out = np.stack([gaussian_filter(np.where(ok, self.points[..., k] - c[k], 0).astype(np.float32), sg,
                                        truncate=3.0) for k in range(3)], -1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out /= w[..., None]
        if not centred:  # centred: relative to the mean valid vertex
            out += c.astype(np.float32)
        out[w < 1e-5] = np.nan
        return out

    def _normals_reference(self) -> np.ndarray:
        du = _deriv(self.points, axis=1)
        dv = _deriv(self.points, axis=0)
        n = np.cross(du, dv)
        ln = np.linalg.norm(n, axis=2, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            n = n / ln
        n[~(ln[..., 0] > 0)] = np.nan
        return n.astype(np.float32)

    def resample(self, render_scale: float, crop=None, normals=None):
        """Bilinearly resample the grid to ``render_scale`` output pixels per voxel.

        Returns (points, normals) on the output grid; a pixel is invalid (NaN) unless all
        four surrounding vertices are valid, so the surface is never interpolated across holes.
        ``crop`` = (x0, y0, w, h) in output pixels; ``normals`` = self.normals() if already computed.
        """
        sx, sy = self.scale
        H, W = self.shape
        out_w = int(np.floor((W - 1) / sx * render_scale)) + 1
        out_h = int(np.floor((H - 1) / sy * render_scale)) + 1
        x0, y0, w, h = crop if crop else (0, 0, out_w, out_h)
        u = (np.arange(x0, x0 + w) / render_scale) * sx
        v = (np.arange(y0, y0 + h) / render_scale) * sy
        nrm = self.normals() if normals is None else normals
        return bilerp_grid(self.points, v, u), _normalize(bilerp_grid(nrm, v, u))

    def triangles(self):
        """Triangulate valid quads.  Returns (vertex_index_grid, verts (N,3), tris (T,3)).

        Quads with all 4 corners valid give 2 triangles; quads with exactly 3 give one.
        """
        ok = self.valid
        idx = np.full(ok.shape, -1, np.int64)
        idx[ok] = np.arange(ok.sum())
        a, b = idx[:-1, :-1], idx[:-1, 1:]
        c, d = idx[1:, :-1], idx[1:, 1:]
        tris = []
        full = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        tris.append(np.stack([a[full], c[full], b[full]], 1))
        tris.append(np.stack([b[full], c[full], d[full]], 1))
        cnt = (a >= 0).astype(int) + (b >= 0) + (c >= 0) + (d >= 0)
        three = cnt == 3
        for missing, tri in (((a < 0), (b, c, d)), ((b < 0), (a, c, d)), ((c < 0), (a, d, b)), ((d < 0), (a, c, b))):
            sel = three & missing
            tris.append(np.stack([t[sel] for t in tri], 1))
        T = np.concatenate(tris).astype(np.int64)
        return idx, self.points[ok].astype(np.float64), T

    def edge_lengths(self):
        """Lengths (voxels) of horizontal and vertical grid edges, NaN where invalid."""
        eu = np.linalg.norm(np.diff(self.points, axis=1), axis=2)
        ev = np.linalg.norm(np.diff(self.points, axis=0), axis=2)
        return eu, ev

    def area(self) -> float:
        """3D area in voxel^2 (sum over triangles(); compiled, bit-identical to _area_reference)."""
        from . import _kernels as K

        return float(0.5 * K.triangle_norms(self.points, self.valid).sum())

    def _area_reference(self) -> float:
        _, V, T = self.triangles()
        return float(0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1).sum())

    def to_obj(self, path: str):
        idx, V, T = self.triangles()
        H, W = self.shape
        rr, cc = np.nonzero(idx >= 0)
        with open(path, "w") as f:
            f.write("# unroller tifxyz export\n")
            np.savetxt(f, V, fmt="v %.4f %.4f %.4f")
            np.savetxt(f, np.c_[cc / max(W - 1, 1), 1 - rr / max(H - 1, 1)], fmt="vt %.7f %.7f")
            t = T + 1
            np.savetxt(f, np.c_[t[:, 0], t[:, 0], t[:, 1], t[:, 1], t[:, 2], t[:, 2]], fmt="f %d/%d %d/%d %d/%d")


def _deriv(p: np.ndarray, axis: int) -> np.ndarray:
    pf = np.moveaxis(p, axis, 0)
    fwd = np.full_like(pf, np.nan)
    bwd = np.full_like(pf, np.nan)
    fwd[:-1] = pf[1:] - pf[:-1]
    bwd[1:] = pf[1:] - pf[:-1]
    cen = (fwd + bwd) / 2
    out = np.where(np.isfinite(cen), cen, np.where(np.isfinite(fwd), fwd, bwd))
    out[~np.isfinite(pf)] = np.nan
    return np.moveaxis(out, 0, axis)


def _normalize(n):
    with np.errstate(invalid="ignore", divide="ignore"):
        return n / np.linalg.norm(n, axis=-1, keepdims=True)


def bilerp_grid(grid: np.ndarray, v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a (H, W, C) grid at the outer product of row coords v and col coords u.

    Any NaN corner (or out of range) yields NaN.
    """
    H, W = grid.shape[:2]
    vi = np.clip(np.floor(v).astype(np.int64), 0, max(H - 2, 0))
    ui = np.clip(np.floor(u).astype(np.int64), 0, max(W - 2, 0))
    tv = (v - vi)[:, None, None].astype(np.float32)
    tu = (u - ui)[None, :, None].astype(np.float32)
    g = lambda dv, du: grid[np.minimum(vi + dv, H - 1)][:, np.minimum(ui + du, W - 1)]
    out = (g(0, 0) * (1 - tv) * (1 - tu) + g(0, 1) * (1 - tv) * tu + g(1, 0) * tv * (1 - tu) + g(1, 1) * tv * tu)
    outside = (v < 0) | (v > H - 1)
    out[outside] = np.nan
    out[:, (u < 0) | (u > W - 1)] = np.nan
    return out
