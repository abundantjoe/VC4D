import numpy as np
import pytest

from unroller import Surface, Volume
from unroller.flatten import flatten
from unroller.metrics import procrustes_rms, stack_report
from unroller.phantom import Phantom, SpiralScroll
from unroller.render import layer_offsets, render
from unroller.resample import flat_surface, obj_to_surface
from unroller.volume import write_ome_zarr


@pytest.fixture(scope="module")
def phantom(tmp_path_factory):
    P = SpiralScroll(size=(48, 224, 224))
    d = tmp_path_factory.mktemp("ph")
    write_ome_zarr(str(d / "v.zarr"), P.volume(), levels=2)
    return P, Volume(str(d / "v.zarr"))


def test_volume_roundtrip(tmp_path):
    a = np.random.default_rng(0).integers(0, 255, (40, 70, 90), dtype=np.uint8)
    write_ome_zarr(str(tmp_path / "a.zarr"), a, chunks=(16, 32, 32))
    v = Volume(str(tmp_path / "a.zarr"))
    z, y, x = 7, 33, 61
    assert v.sample(np.array([[x, y, z]], float), order=0)[0] == a[z, y, x]
    assert np.array_equal(v.levels[0].read(0, 40, 0, 70, 0, 90), a)
    # trilinear midpoint
    m = v.sample(np.array([[x + 0.5, y, z]]))[0]
    assert m == pytest.approx((float(a[z, y, x]) + a[z, y, x + 1]) / 2, abs=1e-3)


def test_tifxyz_roundtrip_invalid_sentinel(tmp_path):
    p = np.random.rand(5, 6, 3).astype(np.float32) * 100
    p[2, 3] = -1
    s = Surface(p, (0.5, 0.5))
    assert not s.valid[2, 3]
    s.save(str(tmp_path / "seg"))
    t = Surface.load(str(tmp_path / "seg"))
    assert t.scale == (0.5, 0.5) and not t.valid[2, 3]
    np.testing.assert_allclose(t.points[s.valid], p[s.valid])


def test_normals_orientation():
    v, u = np.mgrid[0:4, 0:5].astype(float)
    s = Surface(np.stack([u, v, np.zeros_like(u)], -1))
    np.testing.assert_allclose(s.normals()[1, 1], [0, 0, 1], atol=1e-6)  # du x dv, as VC3D


def test_layer_offsets_symmetric():
    np.testing.assert_allclose(layer_offsets(5, 2.0), [-4, -2, 0, 2, 4])


def test_flatten_recovers_arclength(phantom):
    P, _ = phantom
    S = P.surface()
    uv = flatten(S)
    r = procrustes_rms(uv.reshape(-1, 2), P.unrolled_coords(S.points).reshape(-1, 2))
    assert r["rms"] < 0.1


def _flips(S, uv):
    from unroller.flatten import _clean, triangle_singular_values
    idx, V, T = S.triangles()
    T = _clean(V, T)
    det = triangle_singular_values(V, T, uv[idx >= 0])[1]
    return int(min((det < 0).sum(), (det > 0).sum()))


@pytest.mark.parametrize("make", [lambda: SpiralScroll(), lambda: Phantom.random(1, two_layer=False),
                                  lambda: Phantom.random(2, two_layer=True)])
def test_flatten_multires_recovers_truth(make):
    # exactly developable sheets with a deliberately non-isometric grid (distort=0.3): the coarse-to-fine
    # solver alone must land on the true unrolling (the old position upsampling stopped at 0.60 vx rms)
    P = make()
    S = P.surface(step=1.0)
    info = {}
    uv = flatten(S, info=info)
    assert len(info["levels"]) > 1 and info["converged"]
    r = procrustes_rms(uv.reshape(-1, 2), P.unrolled_coords(S.points).reshape(-1, 2))
    assert r["rms"] < 0.005
    assert _flips(S, uv) == 0


def test_rotation_prolongation_is_exact_on_developable_sheet(phantom):
    from unroller.flatten import _Arap, _cell_maps, _clean, _prolong, grid_init
    P, _ = phantom
    S = P.surface(step=1.0)
    C = Surface(S.points[::2, ::2], (0.5, 0.5))
    gc = flatten(C, multires=False, tol=1e-9, iterations=3000)  # solved coarse level (aligned: a rigid motion is harmless)
    idx, V, T = S.triangles()
    T = _clean(V, T)
    rr, cc = np.nonzero(idx >= 0)
    A = _Arap(V, T, rr, cc)
    u = A.solve(_prolong(_cell_maps(C.points, gc), 2, rr, cc, T, A.x, A.y), 0.0)
    e_grid = A.energy(grid_init(idx, V, T, S.scale)[A.order])
    assert A.energy(u) < 1e-9 * e_grid  # the fine start is already the developed sheet


def test_flatten_converges_on_curved_sheet():
    # doubly curved (non-developable) sheet, irregular grid: multires + Anderson must agree with a
    # run to full convergence (measured: energy gap 1.2e-5, layout max 0.003 vx)
    from unroller.flatten import _Arap, _clean, grid_init
    rng = np.random.default_rng(0)
    v, u = np.mgrid[0:60, 0:90].astype(float)
    u += 0.3 * np.sin(u / 7.0)
    x, y = 2.0 * u, 2.0 * v
    pts = np.stack([x, y, ((x - 90) ** 2 + (y - 60) ** 2) / 400.0 + rng.normal(0, 0.05, x.shape)], -1)
    S = Surface(pts, (0.5, 0.5))
    info = {}
    uv = flatten(S, coarse_vertices=1000, info=info)
    idx, V, T = S.triangles()
    T = _clean(V, T)
    rr, cc = np.nonzero(idx >= 0)
    A = _Arap(V, T, rr, cc)
    ref, e_ref, _, _ = A.run(grid_init(idx, V, T, S.scale), 2000, 1e-9)
    assert info["converged"] and len(info["levels"]) > 1
    assert info["energy"] <= e_ref * (1 + 1e-4)
    assert procrustes_rms(uv[idx >= 0], ref)["max"] < 0.01


def _curved_sheet(h, w):
    v, u = np.mgrid[0:h, 0:w].astype(float)
    x, y = 2.0 * u, 2.0 * v
    return Surface(np.stack([x, y, 20 * np.sin(x / 150.0) + (y - h) ** 2 / (40.0 * h)], -1).astype(np.float32), (0.5, 0.5))


def test_flatten_multigrid_level_matches_direct():
    S = _curved_sheet(120, 240)
    info = {}
    uv = flatten(S, coarse_vertices=1000, direct_vertices=5000, info=info)
    assert info["levels"][-1]["cg_iterations"] and info["converged"]
    ref = flatten(S, coarse_vertices=1000)
    assert procrustes_rms(uv.reshape(-1, 2), ref.reshape(-1, 2))["max"] < 0.01


def test_flatten_memory_bounded():
    # levels above direct_vertices hold no factorisation: peak RSS growth stays O(n)
    # (measured on 320k vertices: 1.9 KB/vertex multigrid vs 3.2 KB/vertex factorised)
    import os
    import subprocess
    import sys
    code = (
        "import resource, numpy as np, tests.test_unroller as t\n"
        "from unroller.flatten import flatten\n"
        "S = t._curved_sheet(400, 800)\n"
        "r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "flatten(S, direct_vertices=30000)\n"
        "print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r0) / S.valid.sum())\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
    assert float(out.stdout.split()[-1]) < 2.5  # KB per vertex


def test_flatten_irregular_grid_no_folds():
    # tracer-like grid: +-0.45 vx tangential jitter.  Before the weight clamp and the grid-fold frames,
    # ARAP went indefinite (E < 0) and folded 351 triangles, 0.60 vx rms from the truth
    from unroller.metrics import flatten_report
    P = SpiralScroll()
    t = P.unrolled_coords(P.surface(step=1.0).points)
    rng = np.random.default_rng(0)
    S = Surface(P.point(t[..., 0] + rng.uniform(-0.45, 0.45, t.shape[:2]),
                        t[..., 1] + rng.uniform(-0.45, 0.45, t.shape[:2])), (1.0, 1.0))
    info = {}
    uv = flatten(S, info=info)
    # this grid contracts slowly (a tol=1e-8 run needs ~2900 iterations; at the 200-iteration cap the
    # layout is ~0.009 vx from converged): the stop must say so rather than claim convergence
    assert info["energy"] >= 0
    assert info["converged"] == (info["remaining"] < 1e-4)
    assert flatten_report(S, uv)["flipped_triangles"] == 0
    assert procrustes_rms(uv.reshape(-1, 2), P.unrolled_coords(S.points).reshape(-1, 2))["rms"] < 0.02


def test_flatten_stop_is_not_premature_on_holed_grid():
    # the old stop (plain step < tol) quit here after 7 iterations, 9.0 tol from the converged layout;
    # the rate-aware stop reaches 0.96 tol (measured against a tol=1e-9 run; tol=1e-7 is enough here)
    P = Phantom.random(17, two_layer=False)
    pts = P.surface(step=1.0).points.copy()
    H, W = pts.shape[:2]
    yy, xx = np.mgrid[:H, :W]
    rng = np.random.default_rng(17)
    for _ in range(6):
        r, c, rad = rng.integers(5, H - 5), rng.integers(5, W - 5), rng.uniform(3, 6)
        pts[(yy - r) ** 2 + (xx - c) ** 2 < rad * rad] = np.nan
    S = Surface(pts, (1.0, 1.0))
    info = {}
    uv = flatten(S, info=info)
    ref = flatten(S, tol=1e-7, iterations=3000)
    assert info["converged"] and info["remaining"] < 1e-4
    assert procrustes_rms(uv.reshape(-1, 2), ref.reshape(-1, 2))["max"] < 2e-4


def test_flatten_report_shows_flips_at_mirrored_triangles():
    # a plane grid whose columns 30/31 cross over in one row (a tracer crossover): the true unrolling
    # (x, y) inverts those cells' triangles in their own frames; flatten_report judges them in the
    # mirrored frames but must still expose them, and a layout that ignores the fold must count as flipped
    from unroller.metrics import flatten_report
    rr, cc = np.mgrid[0:50, 0:60].astype(float)
    P = np.stack([cc, rr, np.zeros_like(cc)], -1)
    P[25, 30, 0], P[25, 31, 0] = 30.55, 30.45
    S = Surface(P.astype(np.float32), (1.0, 1.0))
    truth = flatten_report(S, P[..., :2].astype(np.float64))
    assert truth["mirrored_triangles"] > 0 and truth["flipped_triangles"] == 0
    assert truth["mirrored_flipped_own_frame"] == truth["mirrored_triangles"]
    assert truth["flipped_triangles_own_frame"] == truth["mirrored_triangles"]
    naive = flatten_report(S, np.stack([cc, rr], -1))  # the grid layout: ignores the crossover
    assert naive["mirrored_flipped"] == naive["mirrored_triangles"] == naive["flipped_triangles"]
    assert naive["flipped_triangles_own_frame"] == 0
    rep = flatten_report(S, flatten(S))
    assert rep["flipped_triangles"] == 0 and rep["mirrored_triangles"] == truth["mirrored_triangles"]


def test_nested_dissection_is_a_permutation():
    from unroller.flatten import _nested_dissection
    rr, cc = np.nonzero(np.random.default_rng(1).random((40, 70)) > 0.3)
    assert np.array_equal(np.sort(_nested_dissection(rr, cc, leaf=8)), np.arange(len(rr)))


def test_render_ink_and_centering(phantom):
    P, V = phantom
    S = P.surface()
    F = flat_surface(S, flatten(S), out_scale=1.0)
    st = render(F, V, n_layers=9)
    assert abs(stack_report(st)["peak_offset_vx"]) < 0.5


def test_stack_report_of_an_empty_render_has_no_peak():
    """A degenerate trace renders nothing: no peak (None, never a number a check could pass), and no crash
    (nanargmax used to raise on an all-NaN profile)."""
    import warnings
    nan = np.full((7, 4, 5), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the guard itself must not warn either
        rep = stack_report(nan)
    assert rep["peak_layer"] is None and rep["peak_offset_vx"] is None
    assert rep["layers"] == 7 and rep["valid_fraction"] == 0.0 and np.isnan(rep["profile"]).all()
    prof = np.full((7, 1, 1), np.nan)  # render_stream's (layers, 1, 1) mean profile with a single finite layer:
    prof[3] = 5.0                      # still a peak (the guard fires only when nothing at all was rendered)
    assert stack_report(prof)["peak_layer"] == 3
    st = np.zeros((5, 3, 3))
    st[:] = np.array([1.0, 2.0, 4.0, 3.0, 1.0])[:, None, None]  # unchanged for a normal stack: parabolic sub-layer peak
    assert stack_report(st)["peak_layer"] == 2 and abs(stack_report(st)["peak_offset_vx"] - 1 / 6) < 1e-12


def test_obj_import_recovers_isometric_grid(tmp_path, phantom):
    P, _ = phantom
    S = P.surface(distort=0.0)  # arc-length grid, step 2
    S.to_obj(str(tmp_path / "m.obj"))
    T = obj_to_surface(str(tmp_path / "m.obj"), step=1.0)
    eu, ev = T.edge_lengths()
    assert np.nanmedian(eu) == pytest.approx(1.0, abs=0.02)
    assert np.nanmedian(ev) == pytest.approx(1.0, abs=0.02)
    assert T.area() == pytest.approx(S.area(), rel=0.02)


def test_render_stream_matches_in_memory(tmp_path, phantom):
    import tifffile

    from unroller.render import render_stream, zarr_layers_to_tifs

    P, V = phantom
    S = P.surface()
    ref = render(S, V, n_layers=5)
    render_stream(S, V, str(tmp_path / "r.zarr"), n_layers=5, tile=64)
    zarr_layers_to_tifs(str(tmp_path / "r.zarr"), str(tmp_path / "layers"))
    got = np.stack([tifffile.imread(str(tmp_path / "layers" / f"{i:02d}.tif")) for i in range(5)])
    assert np.array_equal(got, np.clip(np.rint(np.nan_to_num(ref)), 0, 255).astype(np.uint8))


def test_refine_recovers_known_offset(phantom):
    from unroller.refine import refine

    P, V = phantom
    S = P.surface()
    shifted = Surface(S.points + 1.5 * S.normals(), S.scale)
    R, rep = refine(shifted, V, iterations=3)
    d = P.sheet_distance(R.points)
    assert np.sqrt(np.nanmean(d[R.valid] ** 2)) < 0.4


def test_trace_stays_on_sheet(phantom):
    from unroller.trace import trace

    P, V = phantom
    seed = P.point(0.5 * P.length, 24.0)
    T = trace(V, seed, step=3.0, generations=15, umbilicus=P.center, bounds=((2, 2, 2), (221, 221, 45)))
    s = P.unrolled_coords(T.points)[..., 0]
    assert T.valid.sum() > 100
    assert np.nansum(np.abs(np.diff(s, axis=1)) > 12) == 0  # no wrap jumps
    assert np.sqrt(np.nanmean(P.sheet_distance(T.points)[T.valid] ** 2)) < 1.0


def test_fiber_profile_finds_recto():
    from unroller.metrics import fiber_profile

    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter

    horiz = gaussian_filter(rng.normal(size=(64, 64)), (0.8, 20))  # varies along rows: recto-like
    vert = gaussian_filter(rng.normal(size=(64, 64)), (20, 0.8))
    stack = np.stack([horiz] * 3 + [vert] * 3) * 50 + 100
    fp = fiber_profile(stack, 1.0, tile=None)
    assert fp["recto_offset_vx"] < 0 < fp["verso_offset_vx"]


def _two_layer_stack(rng, H=96, W=256):
    """Own sheet (recto at -1, verso at +1, isotropic letters/noise on top), dark gaps at +-3, and at the stack
    ends the faint but purely anisotropic facing layers of the neighbouring wraps: inner wrap's verso at -4,
    outer wrap's recto at +4 (the reverse order).  Rows = scroll axis; ink side = -1."""
    from scipy.ndimage import gaussian_filter

    def tex(sig):
        t = gaussian_filter(rng.normal(size=(H, W)), sig)
        return t / t.std()
    rec, ver, iso = tex((0.8, 20)), tex((20, 0.8)), tex(1.0)
    w_r = np.array([0.0, 0.2, 0.7, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    w_v = w_r[::-1]
    mean = np.array([60, 20, 100, 150, 160, 150, 100, 20, 60], float)
    stack = np.stack([mean[i] + 30 * (w_r[i] * rec + w_v[i] * ver) + 25 * iso * (mean[i] > 80) for i in range(9)])
    stack[0] += 15 * ver
    stack[8] += 15 * rec
    return stack + rng.normal(0, 2, stack.shape)


def test_fiber_profile_ignores_neighbouring_wraps():
    from unroller.metrics import fiber_profile

    stack = _two_layer_stack(np.random.default_rng(4))
    whole = fiber_profile(stack, 1.0, tile=None, band_frac=None)
    assert whole["ink_side"] == 1  # the whole-stack extremes are the neighbours' layers: the call inverts
    fp = fiber_profile(stack, 1.0, tile=None)
    assert fp["recto_offset_vx"] == -1 and fp["verso_offset_vx"] == 1 and fp["ink_side"] == -1
    assert fp["sheet_band_vx"] == [-2.0, 2.0]


def test_fiber_profile_partial_coverage_and_axis():
    """A flattened segment covers a slanted part of its bounding image (often < 50%): still a call.
    With the image transposed the call only survives if the scroll axis is passed."""
    from unroller.metrics import fiber_profile

    stack = _two_layer_stack(np.random.default_rng(5))
    H, W = stack.shape[1:]
    yy, xx = np.mgrid[0:H, 0:W]
    stack[:, np.abs(xx - 1.6 * yy - 50) > 45] = np.nan
    assert 0.25 < np.isfinite(stack[0]).mean() < 0.4
    assert fiber_profile(stack, 1.0, tile=None)["ink_side"] == -1
    tr = stack.transpose(0, 2, 1)
    assert fiber_profile(tr, 1.0, tile=None, axis=(0.0, 1.0))["ink_side"] == -1
    assert fiber_profile(tr, 1.0, tile=None)["ink_side"] == 1


def test_fiber_ink_side_on_phantom_ground_truth_surface():
    """Phantom seed 1020 (bench render): the inner wrap's verso used to win the argmin (call +1, ink is at -1)."""
    from unroller.metrics import axis_direction, fiber_profile
    from unroller.volume import ArrayVolume

    P = Phantom.random(1020)
    S = P.surface(distort=0.0, step=1.0)
    n = 4 * int(np.ceil(P.thickness)) + 1
    st = render(S, ArrayVolume(P.volume()), n_layers=n, step=0.5)
    truth = int(np.sign(P.ink_depth()))
    ax = axis_direction(S, up=P.R[:, 2])
    fp = fiber_profile(st, 0.5, tile=None, axis=ax)
    assert fp["ink_side"] == truth and fp["recto_offset_vx"] < 0 < fp["verso_offset_vx"]
    assert fiber_profile(st[::-1], 0.5, tile=None, axis=ax)["ink_side"] == -truth  # normal flipped
    yy, xx = np.mgrid[0:st.shape[1], 0:st.shape[2]]
    st[:, np.abs(xx - 8 * yy - 100) > 120] = np.nan  # ~40% coverage
    assert fiber_profile(st, 0.5, tile=None)["ink_side"] == truth


def test_windowed_phantom_geometry_matches_full():
    from unroller.phantom import Phantom

    kw = dict(size=(32, 96, 96), r0=150.0, pitch=10.0, turns=8, thickness=4.0, center_offset=(-200.0, 0.0))
    full = Phantom(**kw)
    win = Phantom(window=(0.0, 0.6), **kw)
    yy, xx = np.mgrid[0:96, 0:96].astype(float)
    w = np.stack([xx, yy, np.full_like(xx, 16)], -1)
    s1, _, d1 = full._locate(w)
    s2, _, d2 = win._locate(w)
    on = np.abs(d1) < 4  # mid-gap points are equidistant from two wraps (ties), not scored
    assert np.nanmax(np.abs(d1 - d2)[on]) < 0.05
    assert np.nanmax(np.abs(s1 - s2)[on]) < 0.5  # arc length from the coarse integration


def test_greek_font_is_real():
    from unroller.phantom import Phantom

    a = Phantom._ink_image("Ψ", 64, 128, letter_height=30, column_width=64, intercolumn=0)
    b = Phantom._ink_image("Ω", 64, 128, letter_height=30, column_width=64, intercolumn=0)
    assert a.any() and not np.array_equal(a, b)
    with pytest.raises(ValueError):  # letters taller than the volume must not silently vanish
        Phantom._ink_image("Ψ", 64, 64, letter_height=30, column_width=64, intercolumn=0)


def test_fiber_orientation_accuracy_on_lic_ground_truth():
    """Wavy fibers of known direction (line-integral convolution): median error < 1.5 deg, p95 < 6 deg."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    from unroller.fibers import orientation_field

    rng = np.random.default_rng(3)
    H, W = 128, 192
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    g = gaussian_filter(rng.normal(size=(H, W)), 20)
    ang = np.radians(90) + np.radians(25) * g / g.std()
    noise = gaussian_filter(rng.normal(size=(H, W)), 0.7)
    acc = noise.copy()
    for sgn in (1, -1):
        x, y = xx.copy(), yy.copy()
        for _ in range(24):
            a = map_coordinates(ang, [y, x], order=1, mode="nearest")
            x, y = x + sgn * np.cos(a), y + sgn * np.sin(a)
            acc += map_coordinates(noise, [y, x], order=1, mode="nearest")
    img = acc + 0.15 * acc.std() * rng.normal(size=(H, W))
    _, _, fdir = orientation_field(img, fiber_width=0.7)
    est = np.arctan2(fdir[..., 1], fdir[..., 0])
    d = np.degrees(np.abs(np.angle(np.exp(2j * (est - ang)))) / 2)[20:-20, 20:-20]
    assert np.median(d) < 1.5 and np.percentile(d, 95) < 6.0


def _slabs(period=16, thick=8, shape=(64, 128, 128), level=200):
    """Planar sheets normal to x: ``thick`` vx of papyrus every ``period`` vx, air = 0."""
    x = np.arange(shape[2])
    row = np.where((x % period) < thick, level, 0).astype(np.uint8)
    return np.broadcast_to(row, shape).copy()


def test_realism_sheet_stats_on_known_slabs():
    from unroller.realism import sheet_stats

    s = sheet_stats(_slabs(), n_probes=60, seed=1)
    assert len(s["band_vx"]) > 30 and len(s["gap_vx"]) > 30
    assert abs(np.median(s["band_vx"]) - 8) <= 1 and abs(np.median(s["gap_vx"]) - 8) <= 1
    assert s["fused_fraction"][0] == 0.0
    assert np.median(s["core_to_max"]) > 0.95  # flat-topped slab: no edge brightening
    assert np.median(s["insheet_cv"]) < 0.02  # homogeneous papyrus
    assert np.median(s["corr_ax_vx"]) == 16.0 and np.median(s["corr_circ_vx"]) == 16.0  # no texture: capped


def test_realism_sheet_stats_detect_fused_contacts_and_texture():
    from scipy.ndimage import gaussian_filter

    from unroller.realism import sheet_stats

    a = _slabs().astype(float)
    rng = np.random.default_rng(0)
    tex = gaussian_filter(rng.standard_normal(a.shape), 1.5)
    a *= 1 + 0.3 * tex / tex.std()  # textured sheets
    a[:, 64:] = np.maximum(a[:, 64:], 120)  # half of the volume: gaps filled with dense material (fused)
    a = np.clip(a, 0, 255).astype(np.uint8)
    s = sheet_stats(a, n_probes=80, seed=2)
    assert 0.3 < s["fused_fraction"][0] < 0.8
    assert np.median(s["insheet_cv"]) > 0.1
    assert 1.0 < np.median(s["corr_ax_vx"]) < 6.0


def test_realism_slice_stats_spectrum_and_coherence():
    from unroller.realism import PSD_BANDS, slice_stats

    s = slice_stats(_slabs(period=16))  # fundamental at 1/16 = 0.0625 cycles/vx -> band 1
    assert PSD_BANDS[1][0] <= 1 / 16 < PSD_BANDS[1][1]
    assert np.argmax([np.median(s[f"psd_b{i}"]) for i in range(4)]) == 1
    assert np.median(s["coherence"]) > 0.95
    rng = np.random.default_rng(0)
    noise = np.clip(128 + 40 * rng.standard_normal((32, 96, 96)), 0, 255).astype(np.uint8)
    assert np.median(slice_stats(noise)["coherence"]) < 0.6


def test_random_sector_is_quantised_like_the_standardised_scroll():
    from unroller.phantom import Phantom

    v = Phantom.random_sector(3, size=(32, 96, 96)).volume()
    assert np.all(v % 8 == 0) and v.max() <= 248  # the Scroll 1 zarr keeps 5 bits
    assert 0.2 < (v == 0).mean() < 0.8


def test_realism_anisotropy_stats_see_slice_independent_noise():
    from scipy.ndimage import gaussian_filter

    from unroller.realism import anisotropy_stats

    rng = np.random.default_rng(0)
    n = gaussian_filter(rng.standard_normal((48, 64, 64)), (0, 1.2, 1.2))  # correlated in-plane only
    a = np.clip(180 + 20 * n / n.std(), 0, 255).astype(np.uint8)
    s = anisotropy_stats(a)
    assert len(s["hp_corr_z1"]) == 2
    assert abs(np.median(s["hp_corr_z1"])) < 0.1 and np.median(s["hp_corr_xy1"]) > 0.6
    assert np.median(s["hp_corr_xy2"]) < np.median(s["hp_corr_xy1"])


def test_realism_geometry_stats_flat_vs_bent_sheets():
    from unroller.realism import geometry_stats

    flat = _slabs(shape=(80, 128, 128))
    g = geometry_stats(flat, n_probes=60, seed=0)
    assert np.median(g["turn_z32_deg"]) < 3 and np.median(g["turn_s32_deg"]) < 3
    assert g["kink_frac"][0] == 0 and g["lamina_split_frac"][0] == 0
    assert abs(g["bands_per_100vx"][0] - 100 / 16) < 1.0
    assert np.median(g["normal_z_abs"]) < 0.05
    # the same volume with z and x swapped (sheets across the axis) fails the orientation statistic
    gs = geometry_stats(np.ascontiguousarray(flat.transpose(2, 1, 0)), n_probes=40, seed=0)
    assert np.median(gs["normal_z_abs"]) > 0.95
    # sheets bent along z: x shift = 10 sin(2 pi z / 64) -> normals turn by up to ~45 deg over 32 vx
    z, x = np.mgrid[0:80, 0:128]
    xs = x - 10 * np.sin(2 * np.pi * z / 64)
    bent = np.broadcast_to(np.where((np.floor(xs) % 16) < 8, 200, 0)[:, None, :], (80, 128, 128)).astype(np.uint8)
    gb = geometry_stats(np.ascontiguousarray(bent), n_probes=60, seed=0)
    assert np.median(gb["turn_z32_deg"]) > 15
    # a two-lamina band (8 vx papyrus, 2 vx dark split in the middle) is detected as split
    x1 = np.arange(128)
    row = np.where((x1 % 20) < 10, 200, 0)
    row[(x1 % 20 >= 4) & (x1 % 20 < 6)] = 140  # dip above the air threshold, < 80 % of the laminae
    split = np.broadcast_to(row.astype(np.uint8), (64, 128, 128)).copy()
    assert geometry_stats(split, n_probes=60, seed=0)["lamina_split_frac"][0] > 0.5


def test_windowed_elliptical_phantom_matches_full():
    """Windowed and full geometry agree for an elliptical section (small roll; see also the large-roll test)."""
    from unroller.phantom import Phantom

    phi = np.arctan2(140.0, 150.0)  # polar angle of the volume centre (radius 205)
    kw = dict(size=(32, 96, 96), r0=150.0, pitch=10.0, turns=8, thickness=4.0, center_offset=(-150.0, -140.0),
              ellipticity=0.2, ell_angle=phi - np.pi / 4)  # squash oriented for the largest angular shift
    full = Phantom(**kw)
    win = Phantom(window=(phi, 1.3 * 68 / 205), **kw)  # random_sector's rule: 1.3 x the visible half-angle
    yy, xx = np.mgrid[0:96, 0:96].astype(float)
    w = np.stack([xx, yy, np.full_like(xx, 16)], -1)
    s1, _, d1 = full._locate(w)
    s2, _, d2 = win._locate(w)
    on = np.abs(d1) < 4
    assert np.nanmax(np.abs(d1 - d2)[on]) < 0.05
    assert np.nanmax(np.abs(s1 - s2)[on]) < 0.5


def _window_end_fraction(P, z=16):
    Y, X = P.size[1:]
    yy, xx = np.mgrid[0:Y:4, 0:X:4]
    w = np.stack([xx, yy, np.full_like(xx, z)], -1).reshape(-1, 3).astype(float)
    s, _, _ = P._locate(w)
    k = np.clip(np.searchsorted(P._win_a, s, side="right") - 1, 0, len(P._win_a) - 1)
    return float(np.mean(np.minimum(s - P._win_a[k], P._win_b[k] - s) < 3))


def test_large_roll_window_follows_the_elliptical_section():
    """A sector of a large, slightly elliptical roll (random_sector's geometry): no voxel may be left next to the
    end of a built arc (the old window, centred on the spiral parameter, left ~20 % of such volumes without sheets)."""
    from unroller.phantom import Phantom

    Rc, e, ang, P_ = 2000.0, 0.1, 0.8, 25.0
    half = 0.5 * np.hypot(96, 96) + 20
    P = Phantom(size=(32, 96, 96), r0=Rc / (1 + e) - half - 6 * P_, pitch=P_,
                turns=(Rc / (1 - e) - Rc / (1 + e) + 2 * half + 12 * P_) / P_ + 1, thickness=8.0,
                ellipticity=e, ell_angle=ang - np.pi / 4, center_offset=(-Rc * np.cos(ang), -Rc * np.sin(ang)),
                window=(ang, 1.3 * half / (Rc / (1 + e) - half)), pivot_at_volume_center=True)
    assert _window_end_fraction(P) < 0.005
