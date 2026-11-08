"""Whole-scroll synthetic test bed with exact ground truth (every wrap of a roll, umbilicus near the centre).

``WholeScroll`` is a full roll built from the realistic sector model (``Phantom.random_sector``'s fitted
particles: laminae, rough faces, fiber filaments, 3D texture, fused contacts, delamination, tilt, anomalies,
anisotropic blur, slice-independent noise, air clip, 5-bit quantisation) plus whole-scroll features that
matter for winding errors:

  * variable wrap spacing: a tight core whose gaps grow over the first wraps (``core_factor``,
    ``core_wraps``) - ASSUMED, there are no real labels of the core of Scroll 1 in this repository;
  * a crushed cross-section: global ellipticity (fitted) plus crease planes beyond which the section is
    compressed (``crush``; every wrap crossing a crease gets a kink) - ASSUMED, not measured;
  * long fused stretches where 2-3 consecutive wraps touch (``fusions``);
  * tears (fitted) and large missing pieces (``missing_fraction``);
  * the outermost wrap ends at a random angle (fractional turns); the innermost starts at the umbilicus;
  * a mild twist of the cross-section along the axis (``twist`` deg/vx).

Sizes (Z, Y, X), wraps: small (128, 256, 256) 8, medium (256, 640, 640) 20, large (512, 1536, 1536) 40.
To fit a whole roll into a test-size box the gap median of the drawn particle is compressed by
``spec["gap_scale"]``; sheet thickness keeps its real value (``spec["scale"]`` = 1) unless gaps at 0.15x
are not enough.  Textures, edges and noise keep their real voxel scale.  Test-size rolls have much
smaller radii than Scroll 1 (tens to hundreds of vx instead of thousands), so sheets curve more.

The CT is generated tile by tile on demand: ``ws.volume_obj()`` is a ``Volume`` (``sample``, ``levels[0]
.read``) backed by 128^3 chunks rendered with a 16 vx halo, so blur, phase fringes, debris and noise are
continuous across chunk borders (noise is a hash of the global voxel coordinate).  Chunks live in a bounded
RAM LRU cache and, optionally, a bounded disk cache.  ``ws.volume()`` assembles the full array (small/medium).

Truth API (world xyz in voxels, any leading shape):
  sheet_coords(xyz)  -> dict(w, theta, z, depth, s): winding number (0 = innermost), unwrapped spiral
                        angle (rad), axis coordinate, signed distance from the sheet centre (+ = outward),
                        arc length from the umbilicus
  winding_number(xyz), true_unrolled(xyz) -> (s, z) exact isometric layout
  true_surface(w_range, step) -> tifxyz Surface on the sheet centre for wraps [w0, w1)
  ink_at(s, z) -> ink coverage in [0, 1] (Greek text at real letter size, on the recto)
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numba as nb
import numpy as np

from .phantom import PAPYRUS_DEFAULTS, Phantom, _crush, _gap_radius
from .surface import Surface
from .volume import Volume

SIZES = {"small": ((128, 256, 256), 8), "medium": ((256, 640, 640), 20), "large": ((512, 1536, 1536), 40)}
FEATURE_PERIOD = 16384  # feature images repeat after this arc length (bounded memory for large rolls)


@nb.njit(parallel=True, cache=True)
def _hash_normal(z0, y0, x0, nz, ny, nx, seed, out):
    """Standard normal noise that is a pure function of the global voxel (z, y, x) and the seed."""
    for i in nb.prange(nz):
        for j in range(ny):
            for k in range(nx):
                h = np.uint64((z0 + i) * 73856093) ^ np.uint64((y0 + j) * 19349663) ^ np.uint64((x0 + k) * 83492791)
                h = h ^ np.uint64(seed)
                a = h + np.uint64(0x9E3779B97F4A7C15)
                a = (a ^ (a >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                a = (a ^ (a >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                a = a ^ (a >> np.uint64(31))
                b = a + np.uint64(0x9E3779B97F4A7C15)
                b = (b ^ (b >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                b = (b ^ (b >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                b = b ^ (b >> np.uint64(31))
                u1 = (np.float64(a >> np.uint64(11)) + 0.5) / 9007199254740992.0
                u2 = np.float64(b >> np.uint64(11)) / 9007199254740992.0
                out[i, j, k] = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)


@nb.njit(parallel=True, cache=True)
def _bulge_kernel(u, C, Nrm, B, L, out):
    """Phantom._bulge, compiled, with compact support (zero beyond 4 sigma laterally / 8 eps along the normal):
    smooth displacement pushing sheets away from insect bodies."""
    for i in nb.prange(u.shape[0]):
        ox = 0.0
        oy = 0.0
        oz = 0.0
        for k in range(C.shape[0]):
            rx, ry, rz = u[i, 0] - C[k, 0], u[i, 1] - C[k, 1], u[i, 2] - C[k, 2]
            dn = rx * Nrm[k, 0] + ry * Nrm[k, 1] + rz * Nrm[k, 2]
            lat2 = rx * rx + ry * ry + rz * rz - dn * dn
            eps = max(2.0 * abs(B[k]), 1.0)
            sl = 0.6 * L[k]
            if lat2 > 16 * sl * sl or abs(dn) > 8 * eps:  # compact support (4 sigma): < 4e-3 vx neglected
                continue
            amp = abs(B[k]) * np.tanh(dn / eps) * np.exp(-lat2 / (2 * sl * sl)) * np.exp(-dn * dn / (2 * (2 * eps) ** 2))
            ox += amp * Nrm[k, 0]
            oy += amp * Nrm[k, 1]
            oz += amp * Nrm[k, 2]
        out[i, 0] = ox
        out[i, 1] = oy
        out[i, 2] = oz


def _noise_block(z0, y0, x0, shape, seed):
    out = np.empty(shape, np.float32)
    _hash_normal(z0, y0, x0, shape[0], shape[1], shape[2], seed, out)
    return out


def _filtered_std(sig):
    """Std of unit white noise after a gaussian filter of sigmas ``sig`` (measured once on a big block)."""
    from scipy.ndimage import gaussian_filter

    b = gaussian_filter(_noise_block(0, 0, 0, (64, 64, 64), 12345), sig)
    return float(b[16:-16, 16:-16, 16:-16].std())


class WholeScroll(Phantom):
    _inv_iters = 24  # invert the deformation until the update is < 1e-3 vx
    _inv_tol = 1e-3  # 8 nm at 7.91 um; CT and truth API share the same inverse
    def __init__(self, size="small", seed=0, wraps=None, chunk=None, halo=16, cache_gb=1.0, disk_cache=None,
                 disk_cache_gb=2.0, anomalies=True, **spec_override):
        self._ctor = dict(size=size, seed=seed, wraps=wraps, chunk=chunk, halo=halo, anomalies=anomalies, **spec_override)
        shape, nw = SIZES[size] if isinstance(size, str) else (tuple(size), wraps)
        wraps = int(wraps or nw)
        Z, Y, X = shape
        rng = np.random.default_rng(seed)
        q, gate = self._choose_particle(rng)  # one fitted sector particle: material, texture and imaging
        spec = dict(  # whole-scroll features (ASSUMED where no real statistic exists; see module notes)
            core_factor=rng.uniform(0.3, 0.6), core_wraps=rng.uniform(2.0, 4.0),
            crush=[dict(angle=rng.uniform(0, np.pi), c=rng.uniform(0.45, 0.75), k=rng.uniform(0.5, 0.85),
                        w=rng.uniform(2.0, 5.0))] + ([dict(angle=rng.uniform(0, np.pi), c=rng.uniform(0.6, 0.85),
                                                          k=rng.uniform(0.6, 0.9), w=rng.uniform(2.0, 5.0))]
                                                     if rng.random() < 0.3 else []),
            n_fusions=int(rng.integers(2, 5)), twist=rng.uniform(-0.03, 0.03), tilt=tuple(rng.uniform(-3, 3, 2)),
            missing_fraction=rng.uniform(0.01, 0.04), outer_end=rng.uniform(0.05, 0.95), ell_angle=rng.uniform(0, np.pi))
        spec["fusions"] = [(int(rng.integers(1, max(2, wraps - 3))), rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 2.0),
                            int(rng.integers(2, 4))) for _ in range(spec["n_fusions"])]
        spec.update(spec_override)
        e = q["ellipticity"]
        margin = 8 + 2 * q["warp_amp"] + np.tan(np.radians(max(map(abs, spec["tilt"])))) * Z / 2
        rbox = 0.5 * min(X, Y) - margin
        turns = wraps + spec["outer_end"]
        r0 = 0.04 * rbox + 4
        crush = [dict(c, c=c["c"] * rbox) for c in spec["crush"]] or None

        def extent(ft, fg):
            g = self._gap_model(q, ft, fg, spec, rbox)
            tg, rr, _ = _gap_radius(g, r0, turns, seed)
            tt = np.linspace(0, 2 * np.pi * turns, 20000)
            r = np.interp(tt, tg, rr)
            ca, sa = np.cos(spec["ell_angle"]), np.sin(spec["ell_angle"])
            Re = np.array([[ca, -sa], [sa, ca]])
            E = Re @ np.diag([1 + e, 1 - e]) @ Re.T
            xy = np.stack([r * np.cos(tt), r * np.sin(tt)], 1) @ E
            return np.abs(_crush(xy, crush) if crush else xy).max()

        # fit the roll into the box: compress the GAPS first (sheets keep their real thickness: face roughness,
        # filaments and laminae are in absolute voxels); only if gaps at 0.15x are not enough, thin the sheets
        def bisect(fun):
            lo, hi = 0.02, 1.5
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if fun(mid) <= rbox else (lo, mid)
            return lo
        ft, fg = 1.0, 0.15
        if extent(1.0, 0.15) <= rbox:
            fg = bisect(lambda x: extent(1.0, x))
        else:
            ft = bisect(lambda x: extent(x, 0.15))
        f = ft
        gm = self._gap_model(q, ft, fg, spec, rbox)
        self.spec = dict(spec, particle_gate=gate, scale=ft, gap_scale=fg, wraps=wraps, turns=turns, r0=r0, rbox=rbox, size=list(shape), seed=seed,
                         sector_params=q)
        papyrus = {k: q[k] for k in PAPYRUS_DEFAULTS if k in q}
        papyrus["clip"] = q["background"] + q["clip_k"] * q["noise"]
        words = ["".join(rng.choice(list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"), rng.integers(3, 9))) for _ in range(10)]
        super().__init__(
            size=shape, r0=r0, pitch=ft * q["thick_median"] + fg * q["gap_median"], turns=turns, ellipticity=e,
            ell_angle=spec["ell_angle"], tilt_deg=spec["tilt"], center_offset=(0.0, 0.0), thickness=f * q["thick_median"],
            two_layer=True, fiber_contrast=0.25, ink_gain=q["ink_rel"] * q["sheet"], ink_sigma=q["ink_sigma"],
            noise=q["noise"], background=q["background"], sheet=q["sheet"], text=" ".join(words), seed=seed,
            gap_model=gm, letter_height=int(rng.uniform(250, 500)), crush=crush, twist=spec["twist"],
            warp_amp=q["warp_amp"], warp_scale=q["warp_scale"], tear_fraction=q["tear_fraction"], cracks=0,
            thickness_var=q["thickness_var"], density_var=q["density_var"], inclusions=0, psf_sigma=q["psf_sigma"],
            phase_alpha=q["phase_alpha"], ring_amp=0.0, seam=False, clip_background=True,
            delamination=q["delam_rel"] * f * q["thick_median"], delam_fraction=q["delam_fraction"], papyrus=papyrus)
        # exact arc length from the dense curve (the coarse table under-resolves sharp creases)
        ds = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self._xy, axis=0), axis=1))]
        self._s, self._tc, self._sc, self.length = ds, self._t, ds, float(ds[-1])
        self._blend_period_seam()
        # real-size letters are taller than a test volume: put the densest band of text into its z window
        prof = self.ink.mean(1, dtype=np.float64)
        win = np.convolve(prof, np.ones(Z), "valid")
        self._ink_z0 = int(np.argmax(win)) if len(win) else 0
        self._missing_pieces(spec["missing_fraction"], rng)
        # foreign bodies spread over the roll, inclusions at the fitted rate, a stitching seam
        nA = max(1, int(round(np.prod(shape) / 2e8)))  # foreign bodies are rare in real rolls
        if anomalies:
            specs = []
            for kind in ("insect", "hair") * nA:
                s0 = rng.uniform(0.15, 0.95) * self.length
                d = {"kind": kind, "s": s0, "z": rng.uniform(0.2, 0.8) * Z, "side": int(rng.choice([-1, 1]))}
                d.update({"size": rng.uniform(40, 120), "density": rng.uniform(1.3, 2.0)} if kind == "insect"
                         else {"size": rng.uniform(60, 160), "radius": rng.uniform(2.5, 5)})
                specs.append(d)
            self.anomalies = self._place_anomalies(specs)
        n_inc = rng.poisson(q["inclusions"] / (96 * 192 * 192) * np.prod(shape))
        self._inc_c = rng.uniform([0, 0, 0], shape, (n_inc, 3))  # zyx
        self._inc_r = rng.uniform(0.8, 2.5, n_inc)
        self._seam = (int(rng.integers(Z // 4, 3 * Z // 4)), rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8),
                      rng.uniform(0.93, 1.07)) if q["seam"] < 0.3 else None
        self._seeds = (int(rng.integers(1, 2**62)), int(rng.integers(1, 2**62)))
        pp = self.papyrus
        self._db_std = _filtered_std(pp["debris_sig"]) if pp["debris_amp"] > 0 else 1.0
        self._nz_std = _filtered_std((pp["noise_corr_z"], pp["noise_corr"], pp["noise_corr"]))
        self.chunk = int(chunk or (256 if np.prod(shape) > 2e8 else 128))  # big tiles: less halo overhead
        self.halo = int(halo)
        self._cache = _ChunkCache(int(cache_gb * 2**30), disk_cache, int(disk_cache_gb * 2**30))
        self.params.update(whole_scroll=True, spec={k: v for k, v in self.spec.items() if k != "sector_params"})

    # ------------------------------------------------------------ construction helpers
    @staticmethod
    def _choose_particle(rng, tries=12, cv_range=(0.09, 0.18)):
        """Draw fitted sector particles until one renders sheets with a real in-sheet CV (real Scroll 1 IQR
        0.09-0.18, all regions).  The fitted particle SET matches real statistics when pooled, but a few
        particles alone are outliers (e.g. coarse face roughness on thin laminae -> beaded sheets); a whole
        scroll is a single particle, so each candidate is checked on a small sector rendering."""
        from .realism import sheet_stats

        best = None
        for k in range(tries):
            q = Phantom.draw_params(rng)
            v = Phantom.random_sector(int(rng.integers(1, 2**31)), size=(48, 128, 128), particles=[q], jitter=0.0).volume()
            cv = float(np.median(sheet_stats(v, n_probes=120, seed=k, margin_xy=24, margin_z=8)["insheet_cv"]))
            if cv_range[0] <= cv <= cv_range[1]:
                return q, {"tries": k + 1, "insheet_cv": cv}
            if best is None or abs(cv - np.mean(cv_range)) < abs(best[1] - np.mean(cv_range)):
                best = (q, cv)
        return best[0], {"tries": tries, "insheet_cv": best[1], "accepted": False}


    @staticmethod
    def _gap_model(q, ft, fg, spec, rbox):
        return dict(thick_median=ft * q["thick_median"], thick_sigma_ln=q["thick_sigma_ln"], gap_median=fg * q["gap_median"],
                    gap_sigma_ln=q["gap_sigma_ln"], corr_vx=q["corr_vx"], corr_r=0.5 * rbox,
                    core_factor=spec["core_factor"], core_wraps=spec["core_wraps"], fusions=spec["fusions"])

    def _feature_shape(self):
        return min(self.ink.shape[0], self.size[0] + 32), min(self.ink.shape[1], FEATURE_PERIOD)

    def _img(self, img, s, z):
        from . import _kernels as K

        z = np.asarray(z, dtype=np.float64)
        c = np.asarray(s, np.float64)
        if img.shape[1] < self.ink.shape[1]:  # periodic feature image
            c = np.mod(c, img.shape[1])
        out = np.empty(c.size)
        K.bilinear2d(img, np.ascontiguousarray(z.ravel()), np.ascontiguousarray(c.ravel()), out, True)
        return out.reshape(z.shape)

    @classmethod
    def _ink_image(cls, *a, **k):
        return Phantom._ink_image.__func__(cls, *a, as_uint8=True, **k)

    def ink_at(self, s, z):
        """Ink coverage in [0, 1] at arc length s and axis coordinate z."""
        return super().ink_at(s, np.asarray(z, float) + getattr(self, "_ink_z0", 0)) / 255.0

    def _blend_period_seam(self, m=128):
        """Feature images that repeat (large rolls) are cross-faded over m columns at the period seam."""
        if self.ink.shape[1] <= FEATURE_PERIOD:
            return
        a = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, m)))[None].astype(np.float32)
        names = ["gap", "tex_r", "tex_v"] + (["thick_mod"] if self.thickness_var > 0 else []) + \
            (["dens_mod"] if self.density_var > 0 else [])
        imgs = [getattr(self, n) for n in names] + list(self.faces)
        for f in imgs:
            donor = f[:, m - 1::-1]  # mirror of the first m columns: continuous with column 0
            f[:, -m:] = (1 - a) * f[:, -m:] + a * donor

    def _missing_pieces(self, frac, rng):
        """Large missing pieces of papyrus (blobs ~20 x 40 vx) on top of the fitted tears."""
        from scipy.ndimage import gaussian_filter

        if frac <= 0:
            return
        g = gaussian_filter(rng.standard_normal(self.keep.shape).astype(np.float32), (20.0, 40.0), mode="wrap")
        self.keep = self.keep * (g < np.quantile(g, 1 - frac))

    def _bulge(self, u):
        """Insect bulges (Phantom._bulge, compiled), skipping bodies whose support (4 sigma) cannot reach this
        batch of points (part of the model: to_world and to_material use the same field, truth stays exact)."""
        out = np.zeros_like(u)
        B = getattr(self, "_bulge_arrays", None)
        if B is None or B[0] is not getattr(self, "anomalies", None):
            ins = [a for a in getattr(self, "anomalies", []) if a["kind"] == "insect"]
            B = (getattr(self, "anomalies", None), np.array([a["center"] for a in ins], float).reshape(-1, 3),
                 np.array([a["normal"] for a in ins], float).reshape(-1, 3), np.array([a["bulge"] for a in ins], float),
                 np.array([a["length"] for a in ins], float))
            B = B + (np.hypot(4 * 0.6 * B[4], 8 * np.maximum(2 * np.abs(B[3]), 1.0)),)
            self._bulge_arrays = B
        _, C, Nrm, Bu, L, R = B
        if len(C) == 0:
            return out
        lo, hi = u.min(0), u.max(0)
        near = np.all((C + R[:, None] >= lo) & (C - R[:, None] <= hi), 1)
        if near.any():
            _bulge_kernel(np.ascontiguousarray(u, np.float64), C[near], Nrm[near], Bu[near], L[near], out)
        return out

    def _inverse_field(self):
        """Inverse displacement on the forward field's grid (fixed point solved once at every node), so
        to_material starts within ~0.01 vx of the solution and refines in 1-3 iterations."""
        from . import _kernels as K

        if getattr(self, "_F", None) is None:
            self._disp(np.zeros((1, 3)))  # builds the forward grid
        F = self._F
        _, gz, gy, gx = F.shape
        inv = 1.0 / self._F_inv
        o = self._F_origin
        Finv = np.empty_like(F)
        for k0 in range(0, gz, 8):  # bounded memory
            zz, yy, xx = np.meshgrid(np.arange(k0, min(k0 + 8, gz)), np.arange(gy), np.arange(gx), indexing="ij")
            x = np.stack([o[0] + xx.ravel() * inv, o[1] + yy.ravel() * inv, o[2] + zz.ravel() * inv], 1)
            u = x.copy()
            for _ in range(self._inv_iters):
                u_new = x - self._disp(u)
                done = np.abs(u_new - u).max() < self._inv_tol
                u = u_new
                if done:
                    break
            Finv[:, k0:k0 + zz.shape[0]] = (x - u).T.reshape(3, zz.shape[0], gy, gx)
        self._Finv = np.ascontiguousarray(Finv)

    def to_material(self, w):
        from . import _kernels as K

        w = np.asarray(w, float)
        if self._D is None:
            return super().to_material(w)
        if getattr(self, "_Finv", None) is None:
            self._inverse_field()
        flat = np.ascontiguousarray(w.reshape(-1, 3))
        Di = np.empty((len(flat), 3))
        K.trilinear_field(self._Finv, flat, self._F_origin, self._F_inv, Di)
        u = flat - Di
        for _ in range(self._inv_iters):  # refine: same fixed point and tolerance as Phantom.to_material
            u_new = flat - self._disp(u)
            done = np.abs(u_new - u).max() < self._inv_tol
            u = u_new
            if done:
                break
        m = (u.reshape(w.shape) - self._pw) @ self.R + self._pm
        return self._twist(m, -1.0) if self.twist else m

    # ------------------------------------------------------------ truth
    def sheet_coords(self, xyz) -> dict:
        xyz = np.asarray(xyz, float)
        s, z, d = self._locate(xyz)
        t = np.interp(s, self._sc, self._tc)
        return {"w": np.floor(t / (2 * np.pi)), "theta": t, "z": z, "depth": d, "s": s}

    def winding_number(self, xyz):
        return self.sheet_coords(xyz)["w"]

    def true_unrolled(self, xyz):
        c = self.sheet_coords(xyz)
        return np.stack([c["s"], c["z"]], -1)

    def wrap_s_range(self, w0, w1=None):
        """Arc-length interval of wraps [w0, w1)."""
        w1 = w0 + 1 if w1 is None else w1
        tmax = self._tc[-1]
        t = np.clip([2 * np.pi * w0, 2 * np.pi * w1], 0, tmax)
        return tuple(np.interp(t, self._tc, self._sc))

    def present(self, s, z):
        """Papyrus present (not torn / missing) at material (s, z)."""
        s = np.asarray(s, float)
        return (self._img(self.keep, s, z) > 0.5) & (s > 0) & (s < self.length)

    def true_surface(self, w_range=(0, 1), step=2.0, z_range=None, damage=True) -> Surface:
        """Sheet-centre tifxyz Surface of wraps [w0, w1), isometric grid of spacing ``step`` (vx);
        rows follow the scroll axis.  ``damage``: NaN where the papyrus is torn or missing."""
        s0, s1 = self.wrap_s_range(*w_range)
        z0, z1 = z_range or (0.0, self.size[0] - 1.0)
        ss = np.arange(s0 + step / 2, s1, step)  # half-step inset: no vertex sits on a wrap boundary
        zs = np.arange(z0, z1 + 1e-9, step)
        pts = self.point(ss[None, :], zs[:, None])
        if damage:
            pts[~self.present(ss[None, :] + 0 * zs[:, None], zs[:, None] + 0 * ss[None, :])] = np.nan
        return Surface(pts, (1 / step, 1 / step), {"source": "wholescroll", "w_range": list(w_range),
                                                    "s0": float(s0), "z0": float(z0), "step": step})

    # ------------------------------------------------------------ CT, tile by tile
    def render_block(self, z0, z1, y0, y1, x0, x1) -> np.ndarray:
        """Standardised uint8 CT of the box [z0,z1) x [y0,y1) x [x0,x1) (any box; halo handled here)."""
        from scipy.ndimage import gaussian_filter, laplace
        from scipy.ndimage import shift as _shift

        H = self.halo
        Z0, Y0, X0 = z0 - H, y0 - H, x0 - H
        sh = (z1 - z0 + 2 * H, y1 - y0 + 2 * H, x1 - x0 + 2 * H)
        vol = np.empty(sh, np.float32)
        yy, xx = np.mgrid[Y0:Y0 + sh[1], X0:X0 + sh[2]].astype(float)
        for i in range(sh[0]):
            vol[i] = self._density(np.stack([xx, yy, np.full_like(xx, Z0 + i)], -1))
        lo_b = np.array([Z0, Y0, X0])
        hi_b = lo_b + sh
        for a in self.anomalies:  # foreign bodies (same model as Phantom.volume)
            parts = a["parts"] if a["kind"] == "insect" else [(p, a["radius"]) for p in a["points"]]
            for c, r in parts:
                lo = np.maximum(np.floor(c[::-1] - r - 2).astype(int), lo_b)
                hi = np.minimum(np.ceil(c[::-1] + r + 3).astype(int), hi_b)
                if np.any(hi <= lo):
                    continue
                zz3, yy3, xx3 = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                dd = np.sqrt((xx3 - c[0]) ** 2 + (yy3 - c[1]) ** 2 + (zz3 - c[2]) ** 2)
                body = np.clip(r + 0.5 - dd, 0, 1)
                if a["kind"] == "insect":
                    body = body * (0.6 + 0.4 * np.exp(-0.5 * ((dd - r + 1.5) / 1.2) ** 2))
                sl = tuple(slice(l_ - b_, h_ - b_) for l_, h_, b_ in zip(lo, hi, lo_b))
                np.maximum(vol[sl], a["density"] * self.sheet * body, out=vol[sl])
        pp = self.papyrus
        if pp["debris_amp"] > 0:
            R = int(np.ceil(4 * pp["debris_sig"])) + 1  # its own margin: exact across tiles
            db = gaussian_filter(_noise_block(Z0 - R, Y0 - R, X0 - R, tuple(n + 2 * R for n in sh), self._seeds[0]),
                                 pp["debris_sig"])[R:-R, R:-R, R:-R]
            db = np.maximum(db / self._db_std, 0) * (pp["debris_amp"] * self.sheet)
            vol += db * np.clip(1 - vol / self.sheet, 0, 1)
        sel = np.all((self._inc_c + self._inc_r[:, None] + 2 >= lo_b) & (self._inc_c - self._inc_r[:, None] - 2 < hi_b), 1)
        for c, r in zip(self._inc_c[sel], self._inc_r[sel]):
            lo = np.maximum(np.floor(c - r - 1).astype(int), lo_b)
            hi = np.minimum(np.ceil(c + r + 2).astype(int), hi_b)
            if np.any(hi <= lo):
                continue
            zz, yy2, xx2 = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            m = (zz - c[0]) ** 2 + (yy2 - c[1]) ** 2 + (xx2 - c[2]) ** 2 <= r * r
            vol[tuple(slice(l_ - b_, h_ - b_) for l_, h_, b_ in zip(lo, hi, lo_b))][m] += 1.6 * self.sheet
        ps = self.carb["psf_sigma"]
        if ps > 0:
            vol = gaussian_filter(vol, (ps * pp["psf_z_rel"], ps, ps))
        if self.carb["phase_alpha"] > 0:
            vol = vol - self.carb["phase_alpha"] * laplace(gaussian_filter(vol, 0.7))
        if self._seam is not None and Z0 + sh[0] > self._seam[0]:
            k = max(self._seam[0] - Z0, 0)
            vol[k:] = _shift(vol[k:], (0, self._seam[1], self._seam[2]), order=1) * self._seam[3]
        nz = gaussian_filter(_noise_block(Z0, Y0, X0, sh, self._seeds[1]), (pp["noise_corr_z"], pp["noise_corr"], pp["noise_corr"]))
        vol += self.background + nz * (self.noise / self._nz_std)
        vol -= pp["clip"]
        out = np.clip(np.rint(vol[H:-H, H:-H, H:-H]), 0, 255).astype(np.uint8)
        q = int(pp["quant"] or 0)
        return (out // q * q).astype(np.uint8) if q > 1 else out

    def chunk_array(self, key):
        """Chunk (iz, iy, ix): the box [i C, (i+1) C + 1) clipped to the volume (one voxel of overlap)."""
        a = self._cache.get(key)
        if a is None:
            C = self.chunk
            lo = [k * C for k in key]
            hi = [min((k + 1) * C + 1, n) for k, n in zip(key, self.size)]
            a = self.render_block(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
            self._cache.put(key, a)
        return a

    def volume(self) -> np.ndarray:
        """The whole volume in memory (small / medium only)."""
        if np.prod(self.size) > 4e8:
            raise MemoryError("use volume_obj() (tile-wise) for large whole scrolls")
        return self.volume_obj().levels[0].read(0, self.size[0], 0, self.size[1], 0, self.size[2])

    def volume_obj(self) -> Volume:
        return _WholeScrollVolume(self)


def _write_chunks(args):
    ctor, path, keys, threads = args
    import numba
    import numcodecs

    # several workers share the cores; never more than numba allows (NUMBA_NUM_THREADS may be below the core count)
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))
    ws = WholeScroll(cache_gb=0.25, **ctor)
    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=0)
    C = ws.chunk
    for key in keys:
        a = ws.chunk_array(key)[:C, :C, :C]
        blk = np.zeros((C, C, C), np.uint8)
        blk[:a.shape[0], :a.shape[1], :a.shape[2]] = a
        d = os.path.join(path, "0", str(key[0]), str(key[1]))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, str(key[2])), "wb") as f:
            f.write(codec.encode(blk))
        ws._cache = _ChunkCache(ws._cache.ram_bytes)  # nothing reused across chunks: keep memory flat
    return len(keys)


def write_zarr(ws: WholeScroll, path: str, workers: int = 4) -> str:
    """Render the whole volume tile by tile into an OME-Zarr (level 0, chunks of ``ws.chunk``^3, blosc zstd),
    in ``workers`` spawned processes that rebuild the same (deterministic) scroll.  Readable by Volume(path)."""
    import json
    import multiprocessing as mp

    import numcodecs

    C = ws.chunk
    os.makedirs(os.path.join(path, "0"), exist_ok=True)
    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=0)
    with open(os.path.join(path, "0", ".zarray"), "w") as f:
        json.dump({"chunks": [C] * 3, "compressor": codec.get_config(), "dimension_separator": "/", "dtype": "|u1",
                   "fill_value": 0, "filters": None, "order": "C", "shape": list(ws.size), "zarr_format": 2}, f)
    with open(os.path.join(path, ".zattrs"), "w") as f:
        json.dump({"multiscales": [{"version": "0.4", "datasets": [{"path": "0"}]}], "wholescroll": ws.params["spec"]},
                  f, default=float)
    with open(os.path.join(path, ".zgroup"), "w") as f:
        json.dump({"zarr_format": 2}, f)
    keys = [(i, j, k) for i in range(-(-ws.size[0] // C)) for j in range(-(-ws.size[1] // C)) for k in range(-(-ws.size[2] // C))]
    threads = max(1, (os.cpu_count() or 1) // max(workers, 1))
    jobs = [(ws._ctor, path, keys[i::workers], threads) for i in range(workers)]
    if workers <= 1:
        _write_chunks(jobs[0])
    else:
        with mp.get_context("spawn").Pool(workers) as pl:
            pl.map(_write_chunks, jobs)
    return path


class _ChunkCache:
    """RAM LRU (bounded bytes) + optional disk cache (bounded bytes, oldest files evicted)."""

    def __init__(self, ram_bytes, disk_dir=None, disk_bytes=2 * 2**30):
        self.ram, self.ram_bytes, self.used = OrderedDict(), ram_bytes, 0
        self.disk, self.disk_bytes = disk_dir, disk_bytes
        if disk_dir:
            os.makedirs(disk_dir, exist_ok=True)

    def get(self, key):
        a = self.ram.get(key)
        if a is not None:
            self.ram.move_to_end(key)
            return a
        if self.disk:
            p = os.path.join(self.disk, "%d_%d_%d.npz" % key)
            if os.path.exists(p):
                a = np.load(p)["a"]
                self._ram_put(key, a)
                return a
        return None

    def _ram_put(self, key, a):
        self.ram[key] = a
        self.used += a.nbytes
        while self.used > self.ram_bytes and len(self.ram) > 1:
            _, b = self.ram.popitem(last=False)
            self.used -= b.nbytes

    def put(self, key, a):
        self._ram_put(key, a)
        if self.disk:
            np.savez_compressed(os.path.join(self.disk, "%d_%d_%d.npz" % key), a=a)
            files = [os.path.join(self.disk, f) for f in os.listdir(self.disk) if f.endswith(".npz")]
            sizes = {f: os.path.getsize(f) for f in files}
            total = sum(sizes.values())
            for f in sorted(files, key=os.path.getmtime):
                if total <= self.disk_bytes:
                    break
                total -= sizes[f]
                os.remove(f)


class _TileLevel:
    def __init__(self, ws: WholeScroll):
        self.ws = ws
        self.shape = tuple(ws.size)
        self.chunks = (ws.chunk,) * 3
        self.dtype = np.dtype(np.uint8)

    def sample_points(self, p, f: float = 1.0, order: int = 1):
        from . import _kernels as K

        if f != 1.0:
            raise ValueError("a WholeScroll volume has a single level")
        q = np.ascontiguousarray(p, dtype=np.float64)
        out = np.zeros(len(q), np.float32)
        ok = np.all(np.isfinite(q), 1)
        out[~ok] = np.nan
        C = self.ws.chunk
        base = np.floor(q[ok]).astype(np.int64)[:, ::-1]  # zyx
        inside = np.all((base >= 0) & (base < np.array(self.shape)), 1)
        idx = np.flatnonzero(ok)[inside]
        key = base[inside] // C
        if len(idx) == 0:
            return out
        code = (key[:, 0] * 4096 + key[:, 1]) * 4096 + key[:, 2]
        order_ = np.argsort(code, kind="stable")
        code_s = code[order_]
        cuts = np.flatnonzero(np.diff(code_s)) + 1
        for grp in np.split(order_, cuts):
            k = tuple(int(v) for v in key[grp[0]])
            a = self.ws.chunk_array(k)
            loc = q[idx[grp]] - np.array([k[2] * C, k[1] * C, k[0] * C], float)
            res = np.empty(len(grp), np.float32)
            (K.nearest_dense if order == 0 else K.trilinear_dense)(a, np.ascontiguousarray(loc), 1.0, res)
            out[idx[grp]] = res
        return out

    def read(self, z0, z1, y0, y1, x0, x1):
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        C = self.ws.chunk
        lo = np.maximum([z0, y0, x0], 0)
        hi = np.minimum([z1, y1, x1], self.shape)
        if np.any(hi <= lo):
            return out
        for kz in range(lo[0] // C, (hi[0] - 1) // C + 1):
            for ky in range(lo[1] // C, (hi[1] - 1) // C + 1):
                for kx in range(lo[2] // C, (hi[2] - 1) // C + 1):
                    a = self.ws.chunk_array((kz, ky, kx))
                    c0 = np.array([kz, ky, kx]) * C
                    a0 = np.maximum(lo, c0)
                    a1 = np.minimum(hi, c0 + C)
                    src = tuple(slice(u - c, v - c) for u, v, c in zip(a0, a1, c0))
                    dst = tuple(slice(u - b, v - b) for u, v, b in zip(a0, a1, (z0, y0, x0)))
                    out[dst] = a[src]
        return out

    def gather(self, zyx):
        zyx = np.asarray(zyx, np.int64)
        return self.sample_points(zyx[:, ::-1].astype(float), order=0)


class _WholeScrollVolume(Volume):
    """Volume interface (``sample``, ``levels[0].read``) over a WholeScroll, generated on demand."""

    def __init__(self, ws: WholeScroll):
        self.path = None
        self.voxel_size = 7.91
        self.ws = ws
        self.levels = [_TileLevel(ws)]
