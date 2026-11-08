"""Synthetic rolled scrolls with exact ground truth, for validation at scale.

A phantom is a *generalised cylinder*: a planar spiral cross-section (Archimedean, with optional
pitch wobble and elliptical crushing) extruded along the scroll axis, then rigidly tilted and
shifted inside the volume.  Generalised cylinders are developable, so an exact isometric
unrolling exists and its coordinates are known for every voxel:

    s = arc length along the cross-section curve,   z = position along the scroll axis.

Density: Gaussian sheet profile across the sheet.  Ink: a thin denser layer on the outer
(+normal) face, painted from a text image indexed by (s, z) in voxels.

``two_layer=True`` models how papyrus was made: a recto layer (fibers along the roll, s) and a
verso layer (fibers along the axis, z) pressed together, with optional delamination patches where
the layers separate by up to ``delamination`` voxels.  Ink then sits on the recto (inner) face.

``papyrus=dict(...)`` switches the two-layer sheets to the measured material model (used by
``random_sector``; see ``_make_papyrus`` and ``PAPYRUS_DEFAULTS``): soft-edged laminae with
independently rough faces and fiber filaments reaching into the gaps, a 3D anisotropic fiber +
grain texture in material coordinates (recto fibers along s, verso fibers along z), spatially
correlated reconstruction noise and an explicit air-clip level of the standardised volumes.

Scope of the ink model (important): this is a *density* ink, i.e. it tests that geometry,
sampling, layer offsets and registration carry an ink signal to the right pixel.  It is NOT a
model of real carbon ink in Herculaneum CT, which is far subtler; nothing here may be used to
train or certify an ink detector for real scrolls.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .surface import Surface

ALPHABET = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
LATIN = "ABCDEFGHIKLMNOPQRSTVXYZ"


PAPYRUS_DEFAULTS = dict(
    fiber_len=6.0,      # fiber correlation length along the fiber direction (vx)
    fiber_wid=0.8,      # ... across fibers and through the thickness (vx)
    fiber_amp=0.12,     # relative std of the fiber texture
    grain_sig=0.7,      # isotropic grain correlation (vx)
    grain_amp=0.10,     # relative std of the grain
    grain_skew=0.0,     # lognormal skew of the grain (0 = gaussian; > 0 = sparse bright specks)
    edge_soft=0.6,      # half-width of the lamina edge transition (vx)
    rough_amp=0.8,      # std of each face's position (vx), independently per face
    rough_corr=3.0,     # in-plane correlation of the face roughness along the axis (vx)
    rough_aniso=1.0,    # ... times this along the roll (s)
    fray_amp=3.0,       # length of fiber filaments sticking out of the faces into the gaps (vx)
    fray_frac=0.08,     # fraction of the face area that carries filaments
    noise_corr=0.8,     # in-plane (xy) correlation (gaussian sigma, vx) of the reconstruction noise
    noise_corr_z=0.0,   # its correlation along z: slice-wise reconstruction makes slices independent
    psf_z_rel=1.0,      # blur along z relative to in-plane (real Scroll 1: sharper along z)
    debris_amp=0.0,     # loose low-density material in the gaps (relative to sheet density)
    debris_sig=3.0,     # its correlation length (vx)
    clip=None,          # air-clip level subtracted before clipping at 0 (None: 1.5 x background)
    quant=8,            # the standardised Scroll 1 zarr keeps 5 bits: values are multiples of 8 (low bits masked)
)


# random_sector priors: name -> uniform (lo, hi), 7.91 um voxels, drawn in this order (fitted hull)
from .sector_fit import HULL as _HULL  # noqa: E402

SECTOR_PRIORS = dict(_HULL)


def _gap_radius(g, r0, turns, seed):
    """Radius of the spiral as a function of the parameter t for a gap model (see Phantom): local band
    thickness T(t) and free gap G(t) are lognormal processes, r(t) = r(t - 2 pi) + T + G.  Optional keys:
    core_factor/core_wraps (gaps grow from core_factor x median over the first core_wraps wraps),
    fusions [(wrap, angle, extent_rad, n_wraps)] (n_wraps consecutive wraps in contact), corr_r (radius at
    which corr_vx is converted to an angle; default r0).  Returns (tgrid, r, T)."""
    from scipy.ndimage import gaussian_filter1d

    grng = np.random.default_rng(seed + 61)
    rmax_guess = r0 + (g["thick_median"] + g["gap_median"]) * 1.6 * turns
    n_turn = int(np.ceil(2 * np.pi * rmax_guess / 2.0))
    n_tot = int(np.ceil(turns * n_turn)) + 1
    corr = max(1.0, g["corr_vx"] / (2 * np.pi * max(g.get("corr_r", r0), 1.0)) * n_turn)

    def lognormal_process(median, sig):
        x = gaussian_filter1d(grng.standard_normal(n_tot), corr)
        x /= x.std() + 1e-9
        return median * np.exp(sig * x)

    Tt = lognormal_process(g["thick_median"], g["thick_sigma_ln"])
    Gt = np.maximum(lognormal_process(g["gap_median"], g["gap_sigma_ln"]), 0.3)
    if "core_factor" in g:  # tight core: gaps grow from core_factor x median to the median over core_wraps
        ramp = np.clip(np.arange(n_tot) / (n_turn * g.get("core_wraps", 3.0)), 0, 1)
        Gt = np.maximum(Gt * (g["core_factor"] + (1 - g["core_factor"]) * ramp), 0.3)
    tt_ = np.arange(n_tot) * (2 * np.pi / n_turn)
    for w0, a0, ext, nw in g.get("fusions", ()):  # nw consecutive wraps in contact over ext radians
        for j in range(1, int(nw)):
            c0 = 2 * np.pi * (w0 + j) + a0
            m = np.clip(np.minimum(tt_ - c0, c0 + ext - tt_) / 0.3 + 0.5, 0, 1)
            m = m * m * (3 - 2 * m)  # smooth ends over 0.3 rad: the wrap settles onto its neighbour gradually
            Gt = Gt * (1 - m) + 0.3 * m
    rr = np.empty(n_tot)
    rr[:n_turn] = r0 + (Tt[:n_turn] + Gt[:n_turn]) * np.arange(n_turn) / n_turn
    for i in range(n_turn, n_tot):
        rr[i] = rr[i - n_turn] + 0.5 * (Tt[i] + Tt[i - n_turn]) + Gt[i]
    return np.arange(n_tot) * (2 * np.pi / n_turn), rr, Tt


def _crush(xy, crush):
    """Crushed cross-section: beyond |x . d| > c along each direction d the section is compressed by k
    (0 < k <= 1), with a smooth (softplus, width w) but sharp crease.  The map is monotone along d, so
    wraps never cross; every wrap that crosses the crease line gets a kink."""
    out = np.array(xy, float)
    for cr in crush:
        d = np.array([np.cos(cr["angle"]), np.sin(cr["angle"])])
        x = out @ d
        u = (np.abs(x) - cr["c"]) / cr["w"]
        sp = cr["w"] * np.where(u > 30, u, np.log1p(np.exp(np.minimum(u, 30))))
        out = out - (np.sign(x) * (1 - cr["k"]) * sp)[:, None] * d[None]
    return out


def _periodic_noise(rng, shape, sig):
    """Unit-std gaussian random field on a periodic grid (anisotropic gaussian correlation)."""
    from scipy.ndimage import gaussian_filter

    g = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sig, mode="wrap")
    return (g / (g.std() + 1e-9)).astype(np.float32)


def _rot(ax, ay):
    cx, sx, cy, sy = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


class Phantom:
    _inv_iters = 8  # fixed-point iterations inverting the deformation field
    _inv_tol = None  # optional early stop (max update, vx)
    def __init__(self, size=(96, 224, 224), r0=20.0, pitch=12.0, turns=6.0, ellipticity=0.0,
                 ell_angle=0.0, pitch_wobble=0.0, wobble_freq=1.37, wobble_phase=0.0, tilt_deg=(0.0, 0.0),
                 center_offset=(0.0, 0.0), thickness=3.0, ink_offset=1.0, ink_gain=55.0, sheet=150.0,
                 background=20.0, noise=6.0, text="HERCVLANEVM PAPYRVS", seed=0,
                 two_layer=False, fiber_contrast=0.35, delamination=0.0, delam_fraction=0.2,
                 warp_amp=0.0, warp_scale=40.0, tear_fraction=0.0, cracks=0, kollesis_at=None,
                 kollesis_width=0.0, inclusions=0, psf_sigma=0.0, phase_alpha=0.0, ring_amp=0.0,
                 clip_background=False, ink_sigma=0.6, letter_height=None, column_width=None,
                 intercolumn=None, window=None, thickness_var=0.0, gap_model=None, anomalies=None,
                 density_var=0.0, seam=False, pivot_at_volume_center=False, papyrus=None, crush=None, twist=0.0):
        self.size = tuple(size)
        self.twist = float(twist)
        self.r0, self.pitch, self.turns = r0, pitch, turns
        self.thickness, self.ink_offset, self.ink_gain = thickness, ink_offset, ink_gain
        self.sheet, self.background, self.noise, self.seed = sheet, background, noise, seed
        Z, Y, X = self.size
        self.center = np.array([X / 2 + center_offset[0], Y / 2 + center_offset[1], Z / 2])
        self.R = _rot(np.radians(tilt_deg[0]), np.radians(tilt_deg[1]))
        # rigid placement pivot: material point pm sits at world point pw.  Default: the scroll axis at
        # mid-height.  For large-roll sectors the tilt must pivot about the volume centre, otherwise a
        # few degrees of tilt about a far-away axis shift the visible region by hundreds of voxels.
        if pivot_at_volume_center:
            self._pm = np.array([X / 2 - self.center[0], Y / 2 - self.center[1], Z / 2])
            self._pw = np.array([X / 2, Y / 2, Z / 2])
        else:
            self._pm = np.array([0.0, 0.0, Z / 2])
            self._pw = self.center.copy()
        self.params = dict(r0=r0, pitch=pitch, turns=turns, ellipticity=ellipticity, pitch_wobble=pitch_wobble,
                           tilt_deg=[float(v) for v in tilt_deg], thickness=thickness, ink_gain=ink_gain, noise=noise)
        # cross-section curve.  ``window`` = (centre angle, half angle): only the arc of every wrap that
        # can intersect the volume is built densely (large rolls); arc length still comes from the
        # full spiral, integrated on a coarse (~1 vx) curve.
        b = pitch / (2 * np.pi)
        tmax = 2 * np.pi * turns
        ca, sa = np.cos(ell_angle), np.sin(ell_angle)
        Re = np.array([[ca, -sa], [sa, ca]])
        E = Re @ np.diag([1 + ellipticity, 1 - ellipticity]) @ Re.T

        self._T_local = None
        if gap_model is None:
            def radius(t):
                return r0 + b * t + pitch_wobble * pitch * np.sin(wobble_freq * t + wobble_phase) / (2 * np.pi)
            rmax = r0 + b * tmax
        else:
            # local, heavy-tailed gaps and thicknesses (lognormal processes along the curve):
            # r(t) = r(t - 2 pi) + T(t) + G(t), so the gap at every angle is controlled directly and
            # gaps near zero make fused contacts (measured on PHerc. Paris 4: median gap ~7 vx,
            # p10-p90 ~2-29 vx; median band ~7 vx, p10-p90 ~4-20 vx at 7.91 um)
            g = gap_model
            tgrid, rr, Tt = _gap_radius(g, r0, turns, seed)
            tmax = min(tmax, tgrid[-1])
            self._T_local = (tgrid, Tt)

            def radius(t):
                return np.interp(t, tgrid, rr)
            rmax = rr.max()

        def curve(t):
            r = radius(t)
            xy = np.stack([r * np.cos(t), r * np.sin(t)], 1) @ E
            return _crush(xy, crush) if crush else xy

        tc = np.linspace(0, tmax, int(rmax * tmax / 1.0) + 2)
        sc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve(tc), axis=0), axis=1))]
        self._tc, self._sc = tc, sc
        if window is None:
            t = np.linspace(0, tmax, int(rmax * tmax / 0.2) + 2)
            spans = [(0.0, tmax)]
        else:
            phi, half = window
            if ellipticity:  # ``window`` is a polar angle of the elliptical section, not the spiral parameter
                dv = np.array([np.cos(phi), np.sin(phi)]) @ np.linalg.inv(E)
                phi = float(np.arctan2(dv[1], dv[0]))
                half = min(np.pi, half * (1 + ellipticity) / (1 - ellipticity))
            spans = []
            for k in range(int(np.floor((phi - half) / (2 * np.pi))) - 1, int(np.ceil(tmax / (2 * np.pi))) + 2):
                a, bb = max(phi - half + 2 * np.pi * k, 0.0), min(phi + half + 2 * np.pi * k, tmax)
                if bb > a:
                    spans.append((a, bb))
            t = np.concatenate([np.linspace(a, bb, int((r0 + b * bb) * (bb - a) / 0.2) + 2) for a, bb in spans])
        xy = curve(t)
        self._t = t
        self._s = np.interp(t, tc, sc)
        self._xy = xy
        tan = np.gradient(xy, axis=0)
        brk = np.r_[np.nonzero(np.diff(t) > 1e-3 + 2 * (t[1] - t[0]) if len(t) > 1 else [])[0]]
        tan[brk] = xy[brk] - xy[brk - 1]
        tan[np.minimum(brk + 1, len(t) - 1)] = xy[np.minimum(brk + 2, len(t) - 1)] - xy[np.minimum(brk + 1, len(t) - 1)]
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        self._n = np.stack([tan[:, 1], -tan[:, 0]], 1)  # outward for a counter-clockwise spiral
        self._tree = cKDTree(xy)
        self.length = float(sc[-1])
        # compact coordinate for feature images: only the windows are stored
        sa_ = np.interp([a for a, _ in spans], tc, sc)
        sb_ = np.interp([bb for _, bb in spans], tc, sc)
        self._win_a, self._win_b = sa_, sb_
        self._win_c = np.r_[0, np.cumsum(sb_ - sa_ + 8)][:-1]
        self._compact_len = int(np.ceil(self._win_c[-1] + sb_[-1] - sa_[-1])) + 2
        self.min_gap = self._min_gap()  # before deformation; compression can bring sheets into contact
        # real letters (2-4 mm = 250-500 vx at 7.91 um) are taller than a test crop: the ink image may be
        # taller than the volume, which then sees part of a line of text, like a real crop does
        ink_h = Z if letter_height is None else max(Z, int(np.ceil(2.6 * letter_height)))
        self.ink = self._ink_image(text, self._compact_len, ink_h, letter_height, column_width, intercolumn)
        self.ink_sigma = ink_sigma
        self.carb = dict(warp_amp=warp_amp, warp_scale=warp_scale, tear_fraction=tear_fraction, cracks=cracks,
                         kollesis_at=kollesis_at, kollesis_width=kollesis_width, inclusions=inclusions,
                         psf_sigma=psf_sigma, phase_alpha=phase_alpha, ring_amp=ring_amp,
                         clip_background=clip_background, letter_height=letter_height)
        self.params.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in self.carb.items()})
        self._make_warp(warp_amp, warp_scale)
        self._make_damage(tear_fraction, cracks)
        self.two_layer, self.fiber_contrast = two_layer, fiber_contrast
        self.thickness_var = thickness_var
        self.density_var = density_var
        self.seam = seam
        if density_var > 0:  # large-scale density heterogeneity along the sheet (charring, compression)
            from scipy.ndimage import gaussian_filter as _gf

            shp = self._feature_shape()
            dg = _gf(np.random.default_rng(seed + 67).standard_normal(shp, dtype=np.float32), (8.0, 15.0))
            self.dens_mod = np.clip(1 + density_var * dg / (dg.std() + 1e-9), 0.3, 2.0)
        self.anomalies = self._place_anomalies(anomalies or [])
        if thickness_var > 0:  # papyrus thickness is not constant (handmade, unevenly compressed)
            from scipy.ndimage import gaussian_filter

            shp = self._feature_shape()
            g = gaussian_filter(np.random.default_rng(seed + 53).standard_normal(shp, dtype=np.float32), 10.0)
            self.thick_mod = 1 + thickness_var * g / (g.std() + 1e-9)
        self.params.update(two_layer=two_layer, delamination=delamination)
        self.papyrus = None
        if papyrus is not None:
            if not two_layer:
                raise ValueError("the papyrus material model needs two_layer=True")
            self._make_papyrus(dict(PAPYRUS_DEFAULTS, **papyrus))
        if two_layer:
            self._make_layers(delamination, delam_fraction)

    def _make_layers(self, delamination, delam_fraction):
        """Two crossed fiber layers (recto on the inner face, verso outer) with delamination patches.

        Images indexed (z, s) like the ink: recto texture varies across rows (fibers run along s),
        verso texture varies across columns (fibers run along z), gap = local layer separation.
        """
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(self.seed + 17)
        Z, Wd = self._feature_shape()
        tex_r = gaussian_filter(rng.standard_normal(size=(Z, Wd), dtype=np.float32), (0.8, 25.0))
        tex_v = gaussian_filter(rng.standard_normal(size=(Z, Wd), dtype=np.float32), (25.0, 0.8))
        self.tex_r = tex_r / (tex_r.std() + 1e-9)
        self.tex_v = tex_v / (tex_v.std() + 1e-9)
        g = gaussian_filter(rng.standard_normal(size=(Z, Wd), dtype=np.float32), 12.0)
        g /= g.std() + 1e-9
        thr = np.quantile(g, 1 - delam_fraction) if delam_fraction > 0 else np.inf
        self.gap = delamination * np.clip((g - thr) / max(g.max() - thr, 1e-9) * 3, 0, 1)

    def _make_papyrus(self, pp):
        """Material model of carbonised papyrus (all features indexed in material coordinates).

        * texture: periodic 3D tiles (z, s, d) of fiber noise, elongated along s for the recto
          and along z for the verso, plus isotropic grain; different wraps see different s, so the
          texture is uncorrelated between wraps.
        * faces: every lamina face moves independently by a smooth random offset (``rough_amp``)
          and sparse filaments (``fray_frac`` of the face area, up to ``fray_amp`` vx long,
          elongated along the lamina's fiber direction) reach into the gap.
        """
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(self.seed + 83)
        self.papyrus = pp
        fl, fw = pp["fiber_len"], pp["fiber_wid"]
        shp = (64, 128, 16)

        def grain():
            g = _periodic_noise(rng, shp, pp["grain_sig"])
            if pp["grain_skew"] > 0:
                g = np.exp(pp["grain_skew"] * g)
                g = (g - g.mean()) / (g.std() + 1e-9)
            return g

        self.tex3_r = pp["fiber_amp"] * _periodic_noise(rng, shp, (fw, fl, fw)) + pp["grain_amp"] * grain()
        self.tex3_v = pp["fiber_amp"] * _periodic_noise(rng, shp, (fl, fw, fw)) + pp["grain_amp"] * grain()
        Z, Wd = self._feature_shape()
        rc = pp["rough_corr"]
        faces = []
        for k in range(4):  # recto inner, recto outer, verso inner, verso outer
            r = gaussian_filter(rng.standard_normal((Z, Wd)).astype(np.float32), (rc, rc * pp["rough_aniso"]))
            r *= pp["rough_amp"] / (r.std() + 1e-9)
            sig = (0.8, 3.0 * rc) if k < 2 else (3.0 * rc, 0.8)  # filaments follow the fibers
            f = gaussian_filter(rng.standard_normal((Z, Wd)).astype(np.float32), sig)
            f /= f.std() + 1e-9
            q = np.quantile(f, 1 - pp["fray_frac"]) if pp["fray_frac"] > 0 else np.inf
            fil = pp["fray_amp"] * np.clip((f - q) / max(f.max() - q, 1e-6) * 3, 0, 1)
            faces.append((r + fil).astype(np.float32))
        self.faces = faces

    def _tex3(self, T, s, z, d):
        from . import _kernels as K

        a = np.ascontiguousarray(np.asarray(z, np.float64).ravel())
        out = np.empty(a.size)
        K.trilinear_periodic(T, a, np.ascontiguousarray(np.asarray(s, np.float64).ravel()),
                             np.ascontiguousarray(np.asarray(d, np.float64).ravel()), out)
        return out.reshape(np.shape(s))

    def _papyrus_density(self, s, za, d, g, h):
        """Relative density of the two laminae at (s, z, d); g = delamination gap, h = lamina thickness."""
        pp = self.papyrus
        w = pp["edge_soft"]
        out = 0.0
        for k, (c, T) in enumerate(((-(h + g) / 2, self.tex3_r), ((h + g) / 2, self.tex3_v))):
            dl = d - c
            inner = self._img(self.faces[2 * k], s, za)
            outer = self._img(self.faces[2 * k + 1], s, za)
            lo, hi = -h / 2 - inner, h / 2 + outer
            occ = 0.25 * (1 + np.tanh((dl - lo) / w)) * (1 + np.tanh((hi - dl) / w))
            out = out + occ * (1 + self._tex3(T, s, za, dl))
        return out

    def _feature_shape(self):
        """(rows, columns) of the (z, s) feature images: only the rows the volume can see."""
        return min(self.ink.shape[0], self.size[0] + 32), self.ink.shape[1]

    def local_thickness(self, s):
        """Sheet thickness at arc length s (gap_model phantoms vary it along the curve)."""
        if self._T_local is None:
            return np.full(np.shape(s), self.thickness, float)
        t = np.interp(s, self._sc, self._tc)
        return np.interp(t, *self._T_local)

    def _place_anomalies(self, specs):
        """Foreign bodies found inside real rolls: insects (e.g. weevils), hairs, mineral grains.

        Each spec: {"kind": "insect"|"hair", "s": arc length, "z": height, "size": length in vx,
        "side": +1/-1 (which gap, outside/inside the wrap)}.  An insect is a 3-segment body (head,
        thorax, abdomen) denser than papyrus that pushes the neighbouring sheets apart (a smooth
        bulge added to the deformation field); a hair is a thin dense curved tube lying in the gap.
        """
        out = []
        rng = np.random.default_rng(self.seed + 71)
        for sp in specs:
            s0, z0, L, side = sp["s"], sp["z"], sp["size"], sp.get("side", 1)
            T = float(self.local_thickness(s0))
            n_world = self.point(s0, z0, 1.0) - self.point(s0, z0, 0.0)
            n_world /= np.linalg.norm(n_world)
            t_world = self.point(s0 + 1.0, z0) - self.point(s0, z0)
            t_world /= np.linalg.norm(t_world)
            if sp["kind"] == "insect":
                w = L / 4.0  # body width
                c = self.point(s0, z0, side * (T / 2 + w / 2))
                ax = rng.normal(size=3)
                ax -= (ax @ n_world) * n_world
                ax /= np.linalg.norm(ax)
                parts = [(c - 0.38 * L * ax, 0.13 * L), (c - 0.12 * L * ax, 0.16 * L), (c + 0.2 * L * ax, 0.24 * L)]
                out.append({"kind": "insect", "center": c, "normal": n_world, "parts": parts,
                            "bulge": side * w / 2, "width": w, "length": L, "density": sp.get("density", 1.6)})
            elif sp["kind"] == "hair":
                c = self.point(s0, z0, side * (T / 2 + sp.get("radius", 4.0) + 1))
                ang = rng.uniform(0, np.pi)
                up = self.point(s0, z0 + 1.0) - self.point(s0, z0)
                up /= np.linalg.norm(up)
                d = np.cos(ang) * t_world + np.sin(ang) * up
                pts = c + np.linspace(-L / 2, L / 2, 64)[:, None] * d + \
                    np.sin(np.linspace(0, np.pi, 64))[:, None] * rng.uniform(-6, 6) * n_world
                out.append({"kind": "hair", "points": pts, "radius": sp.get("radius", 4.0),
                            "density": sp.get("density", 1.2)})
        if out and self._D is None:
            self._D = np.zeros((3, 4, 4, 4))
            self._wstep = max(self.size) / 2.0
        return out

    def _bulge(self, u):
        """Displacement that pushes sheets away from insect bodies (smooth, odd across the body plane)."""
        out = np.zeros_like(u)
        for a in getattr(self, "anomalies", []):
            if a["kind"] != "insect":
                continue
            r = u - a["center"]
            dn = r @ a["normal"]
            lat2 = (r * r).sum(1) - dn * dn
            eps = max(2.0 * abs(a["bulge"]), 1.0)
            amp = abs(a["bulge"]) * np.tanh(dn / eps) * np.exp(-lat2 / (2 * (0.6 * a["length"]) ** 2)) \
                * np.exp(-dn * dn / (2 * (2 * eps) ** 2))
            out += amp[:, None] * a["normal"][None]
        return out

    def _make_warp(self, amp, scale):
        """Smooth random displacement field D (world voxels) on a coarse grid: bending, waviness,
        twist and local compression that vary along the scroll axis too (carbonised rolls are not
        extrusions).  max|grad D| ~ amp/scale is kept < 0.3 so the map stays a diffeomorphism."""
        from scipy.ndimage import gaussian_filter

        self.warp_amp = amp
        if amp <= 0:
            self._D = None
            return
        rng = np.random.default_rng(self.seed + 29)
        Z, Y, X = self.size
        self._wstep = scale / 2
        shape = tuple(int(np.ceil(n / self._wstep)) + 3 for n in (Z, Y, X))
        D = np.stack([gaussian_filter(rng.normal(size=shape), 1.0) for _ in range(3)])  # (3, gz, gy, gx) xyz
        D *= amp / (D.std() + 1e-9)
        self._D = D

    def _fine_field(self, h=4.0, margin=48.0):
        """The smooth displacement field sampled once (cubic) on an h-voxel grid over the volume;
        lookups are then compiled trilinear (interpolation error ~ h^2/8 |D''| ~ 0.02 vx here)."""
        from scipy.ndimage import map_coordinates

        Z, Y, X = self.size
        ax = [np.arange(-margin, n + margin + h, h) for n in (X, Y, Z)]
        gz, gy, gx = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
        u = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
        g = (u[:, ::-1] / self._wstep + 1).T
        F = np.stack([map_coordinates(self._D[k], g, order=3, mode="nearest").reshape(gz.shape) for k in range(3)])
        self._F = np.ascontiguousarray(F, dtype=np.float64)
        self._F_origin = np.array([-margin, -margin, -margin], float)
        self._F_inv = 1.0 / h

    def _disp(self, u):
        """D at undeformed world points u (N,3 xyz)."""
        from . import _kernels as K

        if getattr(self, "_F", None) is None:
            self._fine_field(h=max(4.0, (np.prod(self.size) / 1.0e7) ** (1 / 3)))  # bounded memory for big volumes
        u = np.ascontiguousarray(u, dtype=np.float64)
        D = np.empty((len(u), 3))
        K.trilinear_field(self._F, u, self._F_origin, self._F_inv, D)
        return D + self._bulge(u) if getattr(self, "anomalies", None) else D

    def _make_damage(self, tear_fraction, cracks):
        """Papyrus-absent mask (z, s): ragged tears/holes + thin transverse cracks."""
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(self.seed + 41)
        Z, Wd = self._feature_shape()
        keep = np.ones((Z, Wd), np.float32)
        if tear_fraction > 0:
            f = gaussian_filter(rng.standard_normal(size=(Z, Wd), dtype=np.float32), (6.0, 10.0)) + \
                0.35 * gaussian_filter(rng.standard_normal(size=(Z, Wd), dtype=np.float32), 1.5)
            keep[f > np.quantile(f, 1 - tear_fraction)] = 0
        for _ in range(int(cracks)):
            s0 = rng.uniform(0, Wd)
            zz = np.arange(Z)
            path = s0 + np.cumsum(rng.normal(0, 0.6, Z))  # mostly transverse (along the axis), wandering
            z0, z1 = sorted(rng.integers(0, Z, 2))
            for z in zz[z0:z1 + 1]:
                c = int(round(path[z]))
                keep[z, max(c - 1, 0):c + 1] = 0
        self.keep = keep

    def compact(self, s):
        """True arc length -> column of the (windowed) feature images."""
        s = np.asarray(s, float)
        k = np.clip(np.searchsorted(self._win_a, s, side="right") - 1, 0, len(self._win_a) - 1)
        return s - self._win_a[k] + self._win_c[k]

    def _img(self, img, s, z):
        from . import _kernels as K

        z = np.asarray(z, dtype=np.float64)
        c = np.ascontiguousarray(self.compact(s).ravel())
        out = np.empty(c.size)
        K.bilinear2d(img, np.ascontiguousarray(z.ravel()), c, out, True)
        return out.reshape(z.shape)

    # ------------------------------------------------------------ factory
    @classmethod
    def random(cls, seed: int, size=(64, 192, 192), two_layer: bool = True):
        """A random but physically sane scroll: wraps never intersect (min gap > thickness)."""
        rng = np.random.default_rng(seed)
        for _ in range(100):
            thickness = rng.uniform(3.0, 8.0) if two_layer else rng.uniform(2.0, 4.0)
            pitch = rng.uniform(thickness + 4.0, thickness + 10.0)
            r0 = rng.uniform(10, 25)
            ell = rng.uniform(0, 0.3)
            rmax = min(size[1], size[2]) / 2 - 10
            turns = max(1.5, (rmax / (1 + ell) - r0) / pitch)
            alphabet = ALPHABET if rng.random() < 0.5 else LATIN
            words = ["".join(rng.choice(list(alphabet), rng.integers(3, 9))) for _ in range(12)]
            p = cls(size=size, r0=r0, pitch=pitch, turns=turns, ellipticity=ell, ell_angle=rng.uniform(0, np.pi),
                    pitch_wobble=rng.uniform(0, 0.25), wobble_freq=rng.uniform(0.5, 2.5),
                    wobble_phase=rng.uniform(0, 2 * np.pi), tilt_deg=tuple(rng.uniform(-6, 6, 2)),
                    center_offset=tuple(rng.uniform(-4, 4, 2)), thickness=thickness,
                    ink_offset=thickness / 3, ink_gain=rng.uniform(20, 60), noise=rng.uniform(3, 10),
                    text=" ".join(words), seed=seed, two_layer=two_layer,
                    delamination=rng.uniform(0, 4) if two_layer else 0.0, delam_fraction=rng.uniform(0.05, 0.3))
            if p.min_gap > thickness + 2.0 and p._fits():
                return p
        raise RuntimeError("could not draw a valid phantom")

    @staticmethod
    def draw_params(rng, priors=None, particles="fitted", jitter=0.25) -> dict:
        """One draw of the random_sector parameters (see random_sector)."""
        pr = dict(SECTOR_PRIORS, **(priors or {}))
        q = {k: float(rng.uniform(*pr[k])) for k in SECTOR_PRIORS}  # fixed draw order
        if isinstance(particles, str) and particles == "fitted":
            from .sector_fit import PARTICLES

            particles = None if priors else PARTICLES
        if particles:
            par = particles[int(rng.integers(len(particles)))]
            for k in SECTOR_PRIORS:
                if k in par:
                    sd = float(np.std([pp[k] for pp in particles]))
                    q[k] = float(np.clip(par[k] + jitter * sd * rng.standard_normal(), *pr[k]))
        return q

    @classmethod
    def random_sector(cls, seed: int, size=(96, 192, 192), radius=(600.0, 3000.0), carbonized: bool = True,
                      anomalies: bool = False, letter_height="real", priors: dict | None = None,
                      particles="fitted", jitter: float = 0.25):
        """A window onto a LARGE roll, like any real crop of Scroll 1: the scroll axis lies far outside
        the volume, so the box sees many gently curved wraps.

        Parameters come from a population (ABC-SMC) fit to real Scroll 1 calibration cubes
        (scripts/fit_sector_priors.py, unroller/sector_fit.py): one of the fitted parameter vectors
        ("particles") is picked and every parameter is perturbed by a gaussian of ``jitter`` x the
        particles' std, clipped to the ``SECTOR_PRIORS`` hull.  ``particles=None`` (or any
        ``priors`` override without explicit particles) draws every parameter uniformly from
        ``SECTOR_PRIORS`` (name -> (lo, hi)) with ``priors`` overriding individual ranges.  Geometry: local band thickness and gap are lognormal
        processes along the curve (heavy-tailed gaps, so a fraction of contacts is fused); sheets are
        two crossed-fiber laminae with delamination, rough faces and fiber filaments, 3D fiber+grain
        texture, density heterogeneity and thin near-papyrus-density ink; deformation, cracks, mineral
        inclusions, CT blur/phase fringes/stitching seam, correlated noise and air clipping.
        ``anomalies`` adds foreign bodies (an insect such as a weevil, a hair) near the centre.
        """
        rng = np.random.default_rng(seed)
        q = cls.draw_params(rng, priors, particles, jitter)
        Z, Y, X = size
        if letter_height == "real":  # 2-4 mm letters at 7.91 um
            letter_height = int(rng.uniform(250, 500))
        gm = {k: q[k] for k in ("thick_median", "thick_sigma_ln", "gap_median", "gap_sigma_ln", "corr_vx")}
        P_ = 1.3 * (gm["thick_median"] + gm["gap_median"])
        Rc = rng.uniform(*radius)
        ang = rng.uniform(0, 2 * np.pi)
        half = 0.5 * np.hypot(X, Y) + 20
        e = q["ellipticity"]  # the section is stretched by 1 +- e: cover every radius the volume can see
        r0 = max(Rc / (1 + e) - half - 6 * P_, 10.0)
        turns = (Rc / (1 - e) - Rc / (1 + e) + 2 * half + 12 * P_) / P_ + 1
        off = (-Rc * np.cos(ang), -Rc * np.sin(ang))  # scroll axis at distance Rc from the volume centre
        win = (ang, min(np.pi, 1.3 * half / max(Rc / (1 + q["ellipticity"]) - half, 1.0)))
        alphabet = ALPHABET if rng.random() < 0.7 else LATIN
        words = ["".join(rng.choice(list(alphabet), rng.integers(3, 9))) for _ in range(12)]
        thick = gm["thick_median"]
        papyrus = {k: q[k] for k in PAPYRUS_DEFAULTS if k in q}
        papyrus["clip"] = q["background"] + q["clip_k"] * q["noise"]
        c = dict(warp_amp=q["warp_amp"], warp_scale=q["warp_scale"], tear_fraction=q["tear_fraction"],
                 cracks=int(q["cracks"]), thickness_var=q["thickness_var"], density_var=q["density_var"],
                 inclusions=int(q["inclusions"]), psf_sigma=q["psf_sigma"], phase_alpha=q["phase_alpha"],
                 ring_amp=0.0, seam=bool(q["seam"] < 0.3), clip_background=True,
                 delamination=q["delam_rel"] * thick, delam_fraction=q["delam_fraction"], papyrus=papyrus) \
            if carbonized else {}
        p = cls(size=size, r0=r0, pitch=P_, turns=turns, ellipticity=q["ellipticity"], ell_angle=rng.uniform(0, np.pi),
                tilt_deg=tuple(rng.uniform(-1, 1, 2) * q.get("tilt_max", 6.0)), center_offset=off, thickness=thick, two_layer=True,
                fiber_contrast=0.25, ink_gain=q["ink_rel"] * q["sheet"], ink_sigma=q["ink_sigma"],
                noise=q["noise"], background=q["background"], sheet=q["sheet"], text=" ".join(words), seed=seed,
                window=win, gap_model=gm, letter_height=letter_height, pivot_at_volume_center=True, **c)
        if anomalies:
            # place foreign bodies on the wrap through the volume centre
            sv, _, _ = p._locate(np.array([[X / 2, Y / 2, Z / 2]]))
            specs = [{"kind": "insect", "s": float(sv[0]) + rng.uniform(-30, 30), "z": Z / 2 + rng.uniform(-10, 10),
                      "size": rng.uniform(40, 120), "side": int(rng.choice([-1, 1])), "density": rng.uniform(1.3, 2.0)},
                     {"kind": "hair", "s": float(sv[0]) + rng.uniform(-60, 60), "z": Z / 2 + rng.uniform(-25, 25),
                      "size": rng.uniform(60, 160), "side": int(rng.choice([-1, 1])), "radius": rng.uniform(2.5, 5)}]
            p.anomalies = p._place_anomalies(specs)
        p.params.update(sector_radius=float(Rc), gap_model={k: float(v) for k, v in gm.items()}, anomalies=anomalies,
                        sector_draw=q)
        return p

    def _min_gap(self):
        """Smallest centre-to-centre distance between adjacent wraps, probed along the normals."""
        gaps = []
        sub = slice(None, None, max(1, len(self._xy) // 20000))
        xy, nrm, ss = self._xy[sub], self._n[sub], self._s[sub]
        for sgn in (1, -1):
            probe = xy[None] + sgn * np.linspace(1, 2 * self.pitch, 40)[:, None, None] * nrm[None]
            dd, ii = self._tree.query(probe.reshape(-1, 2))
            ii = ii.reshape(probe.shape[:2])
            dd = dd.reshape(probe.shape[:2])
            other = (np.abs(self._s[ii] - ss[None]) > 3 * self.pitch) & (dd < 0.3)
            first = np.where(other.any(0), np.argmax(other, 0), -1)
            gaps.append(np.linspace(1, 2 * self.pitch, 40)[first[first >= 0]])
        g = np.concatenate(gaps)
        return float(g.min()) if g.size else float(self.pitch)

    def _fits(self):
        Z, Y, X = self.size
        for z in (0.0, Z - 1.0):
            w = self.to_world(np.c_[self._xy, np.full(len(self._xy), z)])
            if w[:, 0].min() < 3 or w[:, 1].min() < 3 or w[:, 0].max() > X - 4 or w[:, 1].max() > Y - 4:
                return False
        return True

    @staticmethod
    def _font(size):
        from PIL import ImageFont

        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/freefont/FreeSans.ttf"):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        raise RuntimeError("no font with Greek glyphs found (install DejaVu Sans)")

    @classmethod
    def _ink_image(cls, text, width, height, letter_height=None, column_width=None, intercolumn=None, as_uint8=False):
        """Text image indexed (z, s).  Without a layout: lines across the whole strip.  With
        ``letter_height`` (voxels) text is set in columns of ``column_width`` separated by
        ``intercolumn`` blank papyrus, like a bookroll."""
        from PIL import Image, ImageDraw

        img = Image.new("L", (width, height), 0)
        lh = int(letter_height or max(10, height // 5))
        font = cls._font(int(round(lh * 1.35)))  # cap height ~ 0.73 em for DejaVu Sans
        # guard: glyphs must be real (the PIL default font draws every Greek letter as one box)
        a = Image.new("L", (64, 64))
        b = Image.new("L", (64, 64))
        ImageDraw.Draw(a).text((4, 4), "Ψ", font=cls._font(40), fill=255)
        ImageDraw.Draw(b).text((4, 4), "Ω", font=cls._font(40), fill=255)
        assert np.asarray(a).tobytes() != np.asarray(b).tobytes(), "font lacks Greek glyphs"
        line_h = int(round(lh * 1.7))
        if line_h // 2 + line_h > height:
            raise ValueError(f"letter height {lh} vx does not fit a volume {height} vx tall: no text would be drawn")
        cw = column_width or min(width, 2048)  # PIL cannot draw arbitrarily long lines
        gap = intercolumn if intercolumn is not None else 0
        x0 = 8 if column_width is None else gap // 2
        k = 0
        while x0 < width:
            y = max(line_h // 2, height // 8 if letter_height is None else line_h // 2)
            while y + line_h <= height:
                line = (text + "  ") * (cw // max(lh, 1) + 1)
                # clip each line to the column width
                tmp = Image.new("L", (cw, line_h), 0)
                ImageDraw.Draw(tmp).text((0 if column_width else (k % 2) * 17, 0), line, fill=255, font=font)
                img.paste(tmp, (x0, y), tmp)
                y += line_h
                k += 1
            x0 += cw + gap
        if as_uint8:  # 4x less memory for whole-scroll strips
            return np.asarray(img, np.uint8)
        return np.asarray(img, np.float32) / np.float32(255.0)

    # ------------------------------------------------------------ coordinates
    def to_world(self, m):
        """material (mx, my, z_axis) -> world voxel (x, y, z): rigid placement, then deformation."""
        m = np.asarray(m, float)
        if self.twist:
            m = self._twist(m, 1.0)
        u = (m - self._pm) @ self.R.T + self._pw
        if self._D is None:
            return u
        flat = u.reshape(-1, 3)
        return (flat + self._disp(flat)).reshape(u.shape)

    def to_material(self, w):
        w = np.asarray(w, float)
        if self._D is not None:  # invert x = u + D(u) by fixed-point iteration (contraction: |grad D| < 1)
            flat = w.reshape(-1, 3)
            u = flat.copy()
            for _ in range(self._inv_iters):
                u_new = flat - self._disp(u)
                if self._inv_tol is not None and np.abs(u_new - u).max() < self._inv_tol:
                    u = u_new
                    break
                u = u_new
            w = u.reshape(w.shape)
        m = (w - self._pw) @ self.R + self._pm
        return self._twist(m, -1.0) if self.twist else m

    def _twist(self, m, sign):
        """Rotate the cross-section about the scroll axis by twist (deg / vx) x (z - Z/2)."""
        a = sign * np.radians(self.twist) * (m[..., 2] - self.size[0] / 2)
        c, s_ = np.cos(a), np.sin(a)
        out = m.copy()
        out[..., 0] = c * m[..., 0] - s_ * m[..., 1]
        out[..., 1] = s_ * m[..., 0] + c * m[..., 1]
        return out

    def _locate(self, w):
        """world points -> (s, z_axis, signed normal distance d)."""
        m = self.to_material(w.reshape(-1, 3))
        ok = np.all(np.isfinite(m), 1)
        s = np.full(len(m), np.nan)
        d = np.full(len(m), np.nan)
        _, i = self._tree.query(m[ok, :2], workers=-1)
        rel = m[ok, :2] - self._xy[i]
        tan = np.stack([-self._n[i, 1], self._n[i, 0]], 1)
        along = (rel * tan).sum(1)  # sub-sample correction
        s[ok] = self._s[i] + along
        d[ok] = (rel * self._n[i]).sum(1)
        shp = w.shape[:-1]
        return s.reshape(shp), m[:, 2].reshape(shp), d.reshape(shp)

    def ink_depth(self, gap=0.0, s=None):
        """Signed normal position of the ink: the inner (recto) face for two-layer sheets."""
        if not self.two_layer:
            return self.ink_offset
        T = self.thickness if s is None else self.local_thickness(s)
        return -(T / 2) - np.asarray(gap) / 2

    def unrolled_coords(self, xyz):
        s, z, _ = self._locate(np.asarray(xyz, float))
        return np.stack([s, z], -1)

    def sheet_distance(self, xyz):
        return self._locate(np.asarray(xyz, float))[2]

    def point(self, s, z, offset=0.0):
        """World position of material (s, z), displaced ``offset`` along the outward normal."""
        s, z = np.broadcast_arrays(np.asarray(s, float), np.asarray(z, float))
        x = np.interp(s, self._s, self._xy[:, 0])  # valid inside the built windows only
        y = np.interp(s, self._s, self._xy[:, 1])
        nx = np.interp(s, self._s, self._n[:, 0])
        ny = np.interp(s, self._s, self._n[:, 1])
        nn = np.hypot(nx, ny)
        m = np.stack([x + offset * nx / nn, y + offset * ny / nn, z], -1)
        return self.to_world(m)

    def ink_at(self, s, z):
        from . import _kernels as K

        z = np.asarray(z, dtype=np.float64)
        c = np.ascontiguousarray(self.compact(s).ravel())
        out = np.empty(c.size)
        K.bilinear2d(self.ink, np.ascontiguousarray(np.broadcast_to(z, np.shape(s)).ravel()), c, out, False)
        return out.reshape(np.shape(s))

    # ------------------------------------------------------------ outputs
    def _density(self, w: np.ndarray) -> np.ndarray:
        """Material density (before imaging physics) at world points ``w`` (..., 3) xyz."""
        sig = self.thickness / 2.355
        kol = self.carb["kollesis_at"] or []
        kw = self.carb["kollesis_width"]
        s, za, d = self._locate(w)
        inside = (s > 0) & (s < self.length)
        keep = self._img(self.keep, s, za)
        if not self.two_layer:
            sheet = np.exp(-0.5 * (d / sig) ** 2) * inside
            inkw = np.exp(-0.5 * ((d - self.ink_offset) / self.ink_sigma) ** 2) * inside
            v = self.sheet * sheet * keep + self.ink_gain * self.ink_at(s, za) * inkw * keep
        else:
            # recto (inner, d<0) and verso (outer, d>0), each half the thickness, pushed apart by the gap
            g = self._img(self.gap, s, za)
            h = self.local_thickness(s) / 2 * (self._img(self.thick_mod, s, za) if self.thickness_var > 0 else 1.0)
            sl = h / 2.355
            dr, dv = d + h / 2 + g / 2, d - h / 2 - g / 2
            fc = self.fiber_contrast
            if self.papyrus is not None:
                rec, ver = self._papyrus_density(s, za, d, g, h), 0.0
            else:
                rec = np.exp(-0.5 * (dr / sl) ** 2) * (1 + fc * self._img(self.tex_r, s, za))
                ver = np.exp(-0.5 * (dv / sl) ** 2) * (1 + fc * self._img(self.tex_v, s, za))
            extra = 0
            for s_k in kol:  # kollesis: the next sheet overlaps on the outer face, fibers discontinuous
                band = (np.abs(s - s_k) < kw / 2) & inside
                if band.any():
                    de = d - self.thickness
                    extra = extra + band * (np.exp(-0.5 * ((de + h / 2) / sl) ** 2) * (1 + fc * self._img(self.tex_v, s + 977, za))
                                            + np.exp(-0.5 * ((de - h / 2) / sl) ** 2) * (1 + fc * self._img(self.tex_r, s + 977, za)))
            inkw = np.exp(-0.5 * ((d - self.ink_depth(g, s)) / self.ink_sigma) ** 2)
            dm = self._img(self.dens_mod, s, za) if self.density_var > 0 else 1.0
            v = (self.sheet * dm * (rec + ver + extra) * keep + self.ink_gain * self.ink_at(s, za) * inkw * keep) * inside
        return v

    def volume(self) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, laplace

        Z, Y, X = self.size
        rng = np.random.default_rng(self.seed + 1_000_003)
        vol = np.empty(self.size, np.float32)
        yy, xx = np.mgrid[0:Y, 0:X].astype(float)
        for z in range(Z):
            vol[z] = self._density(np.stack([xx, yy, np.full_like(xx, z)], -1))
        # foreign bodies
        zz3, yy3, xx3 = None, None, None
        for a in self.anomalies:
            pts_list = [(c, r) for c, r in a["parts"]] if a["kind"] == "insect" else \
                [(p, a["radius"]) for p in a["points"]]
            for c, r in pts_list:
                lo = np.maximum(np.floor(c[::-1] - r - 2).astype(int), 0)
                hi = np.minimum(np.ceil(c[::-1] + r + 3).astype(int), [Z, Y, X])
                if np.any(hi <= lo):
                    continue
                zz3, yy3, xx3 = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                dd = np.sqrt((xx3 - c[0]) ** 2 + (yy3 - c[1]) ** 2 + (zz3 - c[2]) ** 2)
                body = np.clip(r + 0.5 - dd, 0, 1)
                if a["kind"] == "insect":  # chitin shell denser than the interior
                    body = body * (0.6 + 0.4 * np.exp(-0.5 * ((dd - r + 1.5) / 1.2) ** 2))
                sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                np.maximum(sub, a["density"] * self.sheet * body, out=sub)
        if self.papyrus is not None and self.papyrus["debris_amp"] > 0:  # loose material in the gaps
            db = gaussian_filter(rng.standard_normal(vol.shape, dtype=np.float32), self.papyrus["debris_sig"])
            db = np.maximum(db / (db.std() + 1e-9), 0) * (self.papyrus["debris_amp"] * self.sheet)
            vol += db * np.clip(1 - vol / self.sheet, 0, 1)
        # dense mineral / sediment inclusions
        for _ in range(int(self.carb["inclusions"])):
            c = rng.uniform([0, 0, 0], [Z, Y, X])
            r = rng.uniform(0.8, 2.5)
            lo = np.maximum(np.floor(c - r - 1).astype(int), 0)
            hi = np.minimum(np.ceil(c + r + 2).astype(int), [Z, Y, X])
            zz, yy2, xx2 = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            m = (zz - c[0]) ** 2 + (yy2 - c[1]) ** 2 + (xx2 - c[2]) ** 2 <= r * r
            vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][m] += 1.6 * self.sheet
        # imaging physics: system blur, phase-contrast edge enhancement, rings, noise, standardisation
        if self.carb["psf_sigma"] > 0:
            ps = self.carb["psf_sigma"]
            vol = gaussian_filter(vol, ps if self.papyrus is None else (ps * self.papyrus["psf_z_rel"], ps, ps))
        if self.carb["phase_alpha"] > 0:
            vol = vol - self.carb["phase_alpha"] * laplace(gaussian_filter(vol, 0.7))
        if self.carb["ring_amp"] > 0:  # rings about the rotation axis (here: volume centre), constant along z
            rr = np.hypot(yy - Y / 2, xx - X / 2)
            prof = gaussian_filter(rng.normal(size=int(rr.max()) + 2), 0.8)
            vol += (self.carb["ring_amp"] * prof[np.rint(rr).astype(int)])[None]
        if self.seam:  # slab stitching seam: sub-voxel shift + intensity step above a z plane
            from scipy.ndimage import shift as _shift

            zs = Z // 2 + int(rng.integers(-Z // 4, Z // 4 + 1))
            vol[zs:] = _shift(vol[zs:], (0, rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)), order=1) * rng.uniform(0.93, 1.07)
        if self.papyrus is not None and self.papyrus["noise_corr"] > 0:  # reconstruction noise is correlated
            nc = self.papyrus["noise_corr"]
            nz = gaussian_filter(rng.standard_normal(vol.shape, dtype=np.float32), (self.papyrus["noise_corr_z"], nc, nc))
            vol += self.background + nz * (self.noise / (nz.std() + 1e-9))
        else:
            vol += self.background + rng.normal(0, self.noise, vol.shape).astype(np.float32)
        if self.carb["clip_background"]:  # standardised volumes clip air to 0
            clip = None if self.papyrus is None else self.papyrus["clip"]
            vol -= 1.5 * self.background if clip is None else clip
        out = np.clip(np.rint(vol), 0, 255).astype(np.uint8)
        q = 0 if self.papyrus is None else int(self.papyrus["quant"] or 0)
        if q > 1:
            out = (out // q * q).astype(np.uint8)
        return out

    def surface(self, s0=None, s1=None, z0=4, z1=None, step=2.0, distort=0.3) -> Surface:
        """Ground-truth tifxyz patch on the sheet centre.

        ``distort`` > 0 spaces the columns non-uniformly in arc length (up to +-distort relative),
        a deliberately non-isometric grid that the flattener must correct. 0 = isometric grid.
        """
        s0 = 0.2 * self.length if s0 is None else s0
        s1 = 0.9 * self.length if s1 is None else s1
        z1 = self.size[0] - 5 if z1 is None else z1
        zs = np.arange(z0, z1 + 1e-9, step)
        n = int((s1 - s0) / step)
        k = np.arange(n + 1) / n
        warp = k + distort * np.sin(2 * np.pi * 3 * k) / (2 * np.pi * 3)  # monotone for distort < 1
        ss = s0 + (s1 - s0) * warp
        pts = self.point(ss[None, :], zs[:, None])
        return Surface(pts, (1 / step, 1 / step), {"source": "phantom", "phantom": self.params})


def SpiralScroll(size=(96, 224, 224), **kw):
    """The fixed reference phantom (perfectly circular, axis-aligned Archimedean spiral)."""
    return Phantom(size=size, **kw)
