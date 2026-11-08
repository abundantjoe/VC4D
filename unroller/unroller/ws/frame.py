"""Scroll frame: the umbilicus (scroll axis) and polar coordinates about it (PLAN section 1).  FROZEN after WP0.

``Umbilicus``: a polyline c(z) of axis points, given as ZYX rows in any order (the Scroll 1A file
``umbilicus-scroll1a_zyx.txt`` is unsorted); sorted by z internally and interpolated piecewise linearly.
Beyond the first / last control point the axis is extrapolated along a least-squares line through the
control points within ``extrap_vx`` of that end, and ``extrap(z)`` flags those heights (EXTRAP).

``Frame``: the umbilicus plus the seam direction, the chirality chi (0 = not decided yet; +-1 once decided,
PLAN section 1), the core radius and the axis re-fit shift.  theta(p) is the polar angle about c(p_z) measured
from the seam direction in the VC convention (``VC/core/src/Umbilicus.cpp``: atan2(det(seam, ray),
dot(seam, ray)), wrapped to [0, 2 pi)), in radians.  ``radius`` is the distance from c(p_z) in the plane
z = p_z; ``rhat`` / ``theta_hat`` are the in-plane unit vectors of increasing r / theta.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .types import SCHEMA_VERSION

TWO_PI = 2.0 * np.pi


def wrap(a):
    """Angle difference wrapped to [-pi, pi)."""
    return (np.asarray(a, np.float64) + np.pi) % TWO_PI - np.pi


class Umbilicus:
    def __init__(self, zyx: np.ndarray, extrap_vx: float = 500.0):   # any row order; sorted internally
        a = np.asarray(zyx, np.float64).reshape(-1, 3)
        a = a[np.all(np.isfinite(a), 1)]
        if len(a) == 0:
            raise ValueError("Umbilicus needs at least one finite control point")
        a = a[np.argsort(a[:, 0], kind="stable")]
        zu, inv = np.unique(a[:, 0], return_inverse=True)
        if len(zu) < len(a):  # repeated heights: average them (deterministic)
            cnt = np.bincount(inv)
            a = np.stack([zu, np.bincount(inv, a[:, 1]) / cnt, np.bincount(inv, a[:, 2]) / cnt], 1)
        self.zyx = a
        self.extrap_vx = float(extrap_vx)
        self._lines = [self._end_line(lo) for lo in (True, False)]

    def _end_line(self, lower: bool):
        """(slope_x, slope_y, x0, y0, z0) of the extrapolation line at one end."""
        z, y, x = self.zyx[:, 0], self.zyx[:, 1], self.zyx[:, 2]
        z0 = z[0] if lower else z[-1]
        xe, ye = (x[0], y[0]) if lower else (x[-1], y[-1])
        sel = np.abs(z - z0) <= self.extrap_vx
        if sel.sum() < 2 and len(z) >= 2:  # fewer than two points in the window: the end segment
            sel = np.zeros(len(z), bool)
            sel[[0, 1] if lower else [-2, -1]] = True
        if sel.sum() < 2:  # a single control point: a vertical axis
            return 0.0, 0.0, float(xe), float(ye), float(z0)
        A = np.stack([z[sel] - z0, np.ones(int(sel.sum()))], 1)
        sx = np.linalg.lstsq(A, x[sel], rcond=None)[0][0]
        sy = np.linalg.lstsq(A, y[sel], rcond=None)[0][0]
        # the line is anchored at the end control point itself, so the axis stays continuous there
        return float(sx), float(sy), float(xe), float(ye), float(z0)

    @classmethod
    def load(cls, path) -> "Umbilicus":
        """A VC umbilicus text file ("z, y, x" per line, comma or whitespace separated) or a ``Frame`` JSON."""
        path = str(path)
        if path.endswith(".json"):
            with open(path) as f:
                d = json.load(f)
            return cls(np.asarray(d["umbilicus_zyx"], float), float(d.get("extrap_vx", 500.0)))
        with open(path) as f:
            txt = f.read().replace(",", " ")
        return cls(np.loadtxt(txt.splitlines(), ndmin=2))

    def center(self, z) -> np.ndarray:
        """Axis point(s) (..., 3) xyz at height(s) z."""
        z = np.asarray(z, np.float64)
        zc, yc, xc = self.zyx[:, 0], self.zyx[:, 1], self.zyx[:, 2]
        if len(zc) == 1:
            x = np.full(z.shape, xc[0])
            y = np.full(z.shape, yc[0])
        else:
            x = np.interp(z, zc, xc)
            y = np.interp(z, zc, yc)
        for lower, (sx, sy, x0, y0, z0) in zip((True, False), self._lines):
            m = (z < zc[0]) if lower else (z > zc[-1])
            if np.any(m):
                x = np.where(m, x0 + sx * (z - z0), x)
                y = np.where(m, y0 + sy * (z - z0), y)
        return np.stack([x, y, z], -1)

    def tangent(self, z) -> np.ndarray:
        """Unit axis direction (..., 3) xyz (pointing to +z) at height(s) z: the slope of the polyline segment
        containing z (the extrapolation line beyond the ends)."""
        z = np.asarray(z, np.float64)
        zc, yc, xc = self.zyx[:, 0], self.zyx[:, 1], self.zyx[:, 2]
        if len(zc) >= 2:
            k = np.clip(np.searchsorted(zc, z, side="right") - 1, 0, len(zc) - 2)
            dz = zc[k + 1] - zc[k]
            sx = (xc[k + 1] - xc[k]) / dz
            sy = (yc[k + 1] - yc[k]) / dz
        else:
            sx = np.zeros(z.shape)
            sy = np.zeros(z.shape)
        for lower, (lx, ly, *_rest) in zip((True, False), self._lines):
            m = (z < zc[0]) if lower else (z > zc[-1])
            sx = np.where(m, lx, sx)
            sy = np.where(m, ly, sy)
        t = np.stack([sx, sy, np.ones(z.shape)], -1)
        return t / np.linalg.norm(t, axis=-1, keepdims=True)

    def extrap(self, z) -> np.ndarray:
        """True where z lies beyond the control points (the axis there is extrapolated: flag EXTRAP)."""
        z = np.asarray(z, np.float64)
        return (z < self.zyx[0, 0]) | (z > self.zyx[-1, 0])


@dataclass
class Frame:
    umb: Umbilicus; seam_xy: tuple = (1.0, 0.0); chi: int = 0; r_core: float = 50.0; delta_axis: float = 0.0

    def __post_init__(self):
        s = np.asarray(self.seam_xy, np.float64)
        nrm = float(np.hypot(s[0], s[1]))
        if not nrm > 0:
            raise ValueError("seam_xy must be a non-zero direction")
        self.seam_xy = (float(s[0] / nrm), float(s[1] / nrm))
        self.chi = int(self.chi)
        if self.chi not in (-1, 0, 1):
            raise ValueError("chi must be -1, 0 (undecided) or +1")

    @classmethod
    def from_truth(cls, ph, dz: float = 1.0) -> "Frame":
        """The true frame of a phantom (``Phantom`` or ``WholeScroll``): the axis is the image of the
        material axis, ``ph.to_world([0, 0, z])``, sampled every ``dz`` material slices (16 slices of margin
        beyond the volume; the rest is extrapolated); chi is measured (+1 for the phantoms here: theta
        increases with arc length, so H = t / 2 pi increases with chi * theta); r_core = the mean radius of
        the sheet over t in [3 pi, 5 pi] at mid-height, i.e. the innermost two wraps are CORE (where D2
        measured 76-84% accuracy on phantoms)."""
        Z = int(ph.size[0])
        mz = np.arange(-16.0, Z + 16.0 + 1e-9, float(dz))
        w = ph.to_world(np.stack([np.zeros_like(mz), np.zeros_like(mz), mz], 1))
        w = w[np.all(np.isfinite(w), 1)]
        umb = Umbilicus(w[:, ::-1])
        frame = cls(umb=umb, chi=1)
        tc = np.linspace(3 * np.pi, 5 * np.pi, 129)
        pc = ph.point(np.interp(tc, ph._tc, ph._sc), np.full_like(tc, Z / 2.0))
        rc = frame.radius(pc[np.all(np.isfinite(pc), 1)])
        frame.r_core = float(np.mean(rc)) if len(rc) else float(ph.r0)
        # chirality from geometry: sign of d theta / d s along the sheet at mid-height
        s = np.linspace(0.3, 0.9, 64) * float(ph.length)
        s = s[(s > 2.0) & (s < float(ph.length) - 2.0)]
        zc = np.full_like(s, Z / 2.0)
        p0 = ph.point(s, zc)
        p1 = ph.point(s + 1.0, zc)
        ok = np.all(np.isfinite(p0) & np.isfinite(p1), 1)
        dth = wrap(frame.theta(p1[ok]) - frame.theta(p0[ok]))
        frame.chi = 1 if np.median(dth) > 0 else -1
        return frame

    def _rel(self, p):
        p = np.asarray(p, np.float64)
        c = self.umb.center(p[..., 2])
        return p[..., 0] - c[..., 0], p[..., 1] - c[..., 1]

    def theta(self, p):
        """Polar angle (rad, [0, 2 pi)) of xyz points (..., 3) about c(p_z), from the seam direction."""
        rx, ry = self._rel(p)
        bx, by = self.seam_xy
        a = np.arctan2(bx * ry - by * rx, bx * rx + by * ry)
        return np.where(a < 0, a + TWO_PI, a)

    def radius(self, p):
        rx, ry = self._rel(p)
        return np.hypot(rx, ry)

    def rhat(self, p):
        rx, ry = self._rel(p)
        r = np.hypot(rx, ry)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.stack([rx / r, ry / r, np.zeros_like(r)], -1)
        return out

    def theta_hat(self, p):
        rh = self.rhat(p)
        return np.stack([-rh[..., 1], rh[..., 0], np.zeros_like(rh[..., 0])], -1)

    def wrap_turns(self, p, q) -> np.ndarray:   # chi * wrap(theta(q) - theta(p)) / 2pi
        if self.chi == 0:
            raise ValueError("chirality not decided (Frame.chi == 0)")
        return self.chi * wrap(self.theta(q) - self.theta(p)) / TWO_PI

    def save(self, path):
        """``frame.json`` (atomic write)."""
        d = {"schema_version": SCHEMA_VERSION, "umbilicus_zyx": self.umb.zyx.tolist(),
             "extrap_vx": self.umb.extrap_vx, "seam_xy": list(self.seam_xy), "chi": self.chi,
             "r_core": self.r_core, "delta_axis": self.delta_axis}
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path):
        with open(str(path)) as f:
            d = json.load(f)
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema version {d.get('schema_version')} != {SCHEMA_VERSION}")
        return cls(umb=Umbilicus(np.asarray(d["umbilicus_zyx"], float), d.get("extrap_vx", 500.0)),
                   seam_xy=tuple(d["seam_xy"]), chi=int(d["chi"]), r_core=float(d["r_core"]),
                   delta_axis=float(d.get("delta_axis", 0.0)))
