"""Speed-ups must not change a single bit: the chunked (OME-Zarr) sampling path is checked against the
dense reference kernels under eviction pressure, and the lean storage helpers are round-tripped."""
import os

import numpy as np
import pytest

from unroller.volume import ArrayVolume, Volume, write_ome_zarr


@pytest.fixture(scope="module")
def vols(tmp_path_factory):
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (96, 160, 128), dtype=np.uint8)
    a[:32, :32, :32] = 0  # absent chunks -> fill value path
    d = tmp_path_factory.mktemp("v")
    write_ome_zarr(str(d / "v.zarr"), a, chunks=(32, 32, 32), levels=2)
    # a tiny decoded pool (16 slots of 32^3) for 3*5*4 = 60 chunks forces eviction and splitting
    return a, ArrayVolume(a), Volume(str(d / "v.zarr"), decoded_cache_gb=16 * 32**3 / 2**30)


def _points(rng, n, shape):
    Z, Y, X = shape
    p = rng.uniform([-3, -3, -3], [X + 3, Y + 3, Z + 3], (n, 3))
    p[::97] = np.nan
    p[1::89, 0] = np.inf
    p[2::83] = np.round(p[2::83])  # exact voxel and chunk-boundary hits
    p[3::79, 2] = 31.0 + rng.uniform(0, 1e-9, len(p[3::79]))
    return p


def test_chunked_sampling_bit_identical_to_dense(vols):
    a, A, Z = vols
    rng = np.random.default_rng(0)
    for order in (1, 0):
        for p in (_points(rng, 50_000, a.shape),                       # random: thrashes the 16-slot pool
                  np.sort(_points(rng, 50_000, a.shape), axis=0)):     # coherent
            ref = A.sample(p, order=order)
            got = Z.sample(p, order=order)
            assert np.array_equal(ref, got, equal_nan=True)
            assert np.array_equal(ref, Z.sample(p, order=order), equal_nan=True)  # warm pool
    assert Z.levels[0].decoded > 16  # eviction happened


def test_level1_sampling_matches_dense_level(vols):
    a, _, Z = vols
    rng = np.random.default_rng(1)
    p = _points(rng, 20_000, a.shape)
    ref = ArrayVolume(Z.levels[1].read(0, 48, 0, 80, 0, 64)).sample(p * 0.5)
    assert np.array_equal(ref, Z.sample(p, level=1), equal_nan=True)


def test_sample_layers_fused_equals_per_layer(vols):
    a, A, Z = vols
    rng = np.random.default_rng(2)
    p = _points(rng, 3000, a.shape).astype(np.float32)
    n = rng.normal(size=(3000, 3)).astype(np.float32)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    offs = (np.arange(21) - 10) * 1.0
    for V in (A, Z):
        ref = np.stack([V.sample(p + o * n) for o in offs])
        assert np.array_equal(ref, V.sample_layers(p, n, offs), equal_nan=True)


def test_render_identical_with_fused_layers(vols):
    from unroller.render import render
    from unroller.surface import NORMAL_SIGMA_VX, Surface

    a, A, Z = vols
    yy, xx = np.mgrid[0:60, 0:50].astype(np.float32)
    pts = np.stack([xx * 2 + 10, yy * 2 + 20, 40 + 10 * np.sin(xx / 7)], -1)
    pts[5:9, 5:9] = np.nan
    S = Surface(pts, (0.5, 0.5))
    pp, nn = S.resample(1.0, normals=S.normals(NORMAL_SIGMA_VX))  # render's default normals
    offs = (np.arange(9) - 4.0)
    ref = np.stack([A.sample(pp + o * nn) for o in offs])
    for V in (A, Z):
        assert np.array_equal(ref, render(S, V, n_layers=9, tile_rows=37), equal_nan=True)


def test_store_roundtrip_and_report(tmp_path):
    from unroller import store

    rng = np.random.default_rng(0)
    arrs = {"u8": (rng.normal(100, 10, (40, 50, 60))).clip(0, 255).astype(np.uint8),
            "f32": rng.normal(size=(33, 17, 3)).astype(np.float32),
            "nan": np.where(rng.random((20, 20)) < 0.3, np.nan, rng.random((20, 20))).astype(np.float32),
            "i64": np.arange(1000, dtype=np.int64).reshape(10, 100)}
    for k, x in arrs.items():
        for fmt in ("zarr", "npz"):
            path = str(tmp_path / f"{k}.{fmt}")
            rep = store.save(path, x, chunks=(16, 16, 16)[:x.ndim])
            assert rep["raw_bytes"] == x.nbytes and rep["stored_bytes"] > 0
            y = store.load(path)
            assert y.dtype == x.dtype and y.shape == x.shape
            assert np.array_equal(x, y, equal_nan=True)
            if fmt == "zarr" and x.ndim == 3:
                assert np.array_equal(store.load(path, (slice(3, 30), slice(5, 7), slice(None))),
                                      x[3:30, 5:7], equal_nan=True)
    # zarr output is a valid OME-Zarr-free zarr v2 array that Volume can sample
    store.save(str(tmp_path / "vol.zarr"), arrs["u8"], chunks=(16, 16, 16))
    V = Volume(str(tmp_path / "vol.zarr"))
    p = rng.uniform(0, 39, (1000, 3))
    assert np.array_equal(V.sample(p), ArrayVolume(arrs["u8"]).sample(p))


def test_budget_evicts_least_recently_used(tmp_path):
    from unroller.store import WorkBudget

    root = tmp_path / "work"
    root.mkdir()
    for i in range(6):
        d = root / f"item{i}"
        d.mkdir()
        (d / "blob").write_bytes(b"x" * 1000)
        os.utime(d / "blob", (1000 + i, 1000 + i))
    B = WorkBudget(str(root), max_bytes=3500)
    B.touch(str(root / "item0"))  # item0 used most recently -> survives
    removed = B.enforce()
    left = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
    assert left == ["item0", "item4", "item5"]
    assert sorted(os.path.basename(r) for r in removed) == ["item1", "item2", "item3"]
    assert B.usage() <= 3500
    B.protect(str(root / "item4"))
    B.max_bytes = 0
    B.enforce()
    assert [p.name for p in root.iterdir() if not p.name.startswith(".")] == ["item4"]


def test_compiled_normals_bit_identical():
    from unroller.surface import Surface

    rng = np.random.default_rng(0)
    for i in range(40):
        H, W = rng.integers(1, 50, 2)
        p = rng.normal(0, 50, (H, W, 3)).astype(np.float32) * rng.choice([1e-3, 1, 1e4])
        p[rng.random((H, W)) < rng.uniform(0, 0.6)] = np.nan
        if i % 5 == 0:
            p[:] = p[0:1, 0:1]  # degenerate: zero-length normals -> NaN
        S = Surface(p)
        S.points[rng.random((H, W, 3)) < 0.05] = np.nan  # component-wise NaN (not produced by __init__)
        ref, got = S._normals_reference(), S.normals()
        nan = np.isnan(ref)
        assert np.array_equal(nan, np.isnan(got))
        assert np.array_equal(ref[~nan].view(np.uint32), got[~nan].view(np.uint32))  # bits, incl. -0.0


def _snap_cases(rng, shape, m=400):
    Z, Y, X = shape
    p = rng.uniform([1, 1, 1], [X - 2, Y - 2, Z - 2], (m, 3))
    p[:20] = rng.uniform([-4, -4, -4], [X + 4, Y + 4, Z + 4], (20, 3))  # edge of / outside the volume
    p[20:30, 2] = 0.0
    n = rng.normal(size=(m, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[30:40] = np.nan  # NaN normals (unfilled holes in the normal field)
    p[40:45] = np.nan
    return p, n


def _assert_snap_identical(V, p, n):
    import unroller.trace as tr
    from unroller.snap_fast import snap_ms

    opts = ({}, dict(basin=12.0), dict(basin=3.0, iters=50, tol=1e-3, strict=False), dict(iters=50, tol=1e-3))
    for radius, sigma in ((1.5, 3.5), (2.0, 1.0), (5.0, 4.0), (1.0, 1.5)):
        for clip in (None, 140.25):
            for max_width in (None, 6.0):
                for mc, opt in zip((10.0, -np.inf, 40.0, 25.0), opts):
                    kw = dict(min_contrast=mc, clip=clip, max_width=max_width, **opt)
                    a = tr._snap_ms(V, p, n, radius, sigma, return_width=True, **kw)
                    b = snap_ms(V, p, n, radius, sigma, return_width=True, **kw)
                    assert np.array_equal(a[1], b[1])                          # ok flags
                    assert np.array_equal(a[0], b[0], equal_nan=True)          # positions, bit for bit
                    assert np.array_equal(a[2], b[2], equal_nan=True)          # widths
                    a2 = tr._snap_ms(V, p, n, radius, sigma, **kw)
                    b2 = snap_ms(V, p, n, radius, sigma, **kw)
                    assert len(b2) == 2 and all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a2, b2))


def test_snap_fast_identical_on_phantoms():
    from unroller.phantom import Phantom

    for seed in (1, 2):
        vol = Phantom.random(seed).volume()
        rng = np.random.default_rng(seed)
        p, n = _snap_cases(rng, vol.shape)
        _assert_snap_identical(ArrayVolume(vol), p, n)


def test_snap_fast_identical_on_real_scroll1(tmp_path):
    """64^3 crop of PHerc. Paris 4 (Scroll 1, 54 keV 7.91 um, zyx origin stored in the file), both as an
    in-memory volume and as an OME-Zarr (chunked sampling path)."""
    d = np.load(os.path.join(os.path.dirname(__file__), "data", "scroll1_cube64.npz"))
    a = d["a"]
    write_ome_zarr(str(tmp_path / "s1.zarr"), a, chunks=(32, 32, 32))
    rng = np.random.default_rng(7)
    p, n = _snap_cases(rng, a.shape)
    for V in (ArrayVolume(a), Volume(str(tmp_path / "s1.zarr"))):
        _assert_snap_identical(V, p, n)


def test_trace_with_snap_fast_identical(monkeypatch):
    """The tracer runs the compiled snap; with the numpy reference patched in, the surface is the same."""
    import unroller.snap_fast as sf
    import unroller.trace as tr
    from unroller.phantom import Phantom

    P = Phantom.random(1)
    V = ArrayVolume(P.volume())
    seed = P.point(0.5 * P.length, 32)
    kw = dict(step=3.0, generations=15, umbilicus=P.center, up=P.R[:, 2], bounds=((2, 2, 2), (189, 189, 61)))
    for opt in ({}, tr.LEGACY, dict(ms_iters=50, ms_tol=1e-3, relax_soft=True)):
        got = tr.trace(V, seed, **kw, **opt)
        with monkeypatch.context() as m:
            m.setattr(sf, "snap_ms", tr._snap_ms)
            ref = tr.trace(V, seed, **kw, **opt)
        assert np.array_equal(ref.points, got.points, equal_nan=True)


def _peak_offsets_reference(s, vol, radius=3.0, step=0.5, sigma_vx=8.0, level=0):
    """refine.peak_offsets as of dfd7356 (per-offset sampling and smoothing, numpy peak pick)."""
    from unroller.refine import _nan_gauss

    n = s.normals()
    offs = np.arange(-radius, radius + 1e-9, step)
    sig = sigma_vx * np.sqrt(s.scale[0] * s.scale[1])
    prof = np.stack([_nan_gauss(vol.sample(s.points + o * n, level=level), sig) for o in offs])
    i = np.argmax(np.nan_to_num(prof, nan=-np.inf), axis=0)
    i = np.clip(i, 1, len(offs) - 2)
    a = np.take_along_axis(prof, (i - 1)[None], 0)[0]
    b = np.take_along_axis(prof, i[None], 0)[0]
    c = np.take_along_axis(prof, (i + 1)[None], 0)[0]
    den = a - 2 * b + c
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den < 0, 0.5 * (a - c) / den, np.nan)
    out = offs[i] + frac * step
    edge = (np.argmax(np.nan_to_num(prof, nan=-np.inf), axis=0) == 0) | (
        np.argmax(np.nan_to_num(prof, nan=-np.inf), axis=0) == len(offs) - 1)
    out[edge | ~s.valid] = np.nan
    return out


def test_refine_identical_to_reference(tmp_path):
    from unroller.phantom import Phantom
    from unroller.refine import peak_offsets

    P = Phantom.random(2)
    vol = P.volume()
    write_ome_zarr(str(tmp_path / "p.zarr"), vol, chunks=(32, 32, 32))
    S = P.surface(step=2.0)
    S.points[10:14, 5:30] = np.nan  # holes
    S.points[:, :3] += 40.0  # part of the surface leaves the volume
    from unroller.surface import Surface

    S = Surface(S.points, S.scale)
    for V in (ArrayVolume(vol), Volume(str(tmp_path / "p.zarr"), decoded_cache_gb=24 * 32**3 / 2**30)):
        for kw in (dict(), dict(radius=2.0, sigma_vx=3.0), dict(radius=4.0, step=0.25, sigma_vx=1.0)):
            ref = _peak_offsets_reference(S, V, **kw)
            got = peak_offsets(S, V, **kw)
            assert np.array_equal(ref, got, equal_nan=True)
            assert np.isfinite(got).sum() > 100


def test_compiled_area_bit_identical():
    from unroller.surface import Surface

    rng = np.random.default_rng(5)
    for i in range(30):
        H, W = rng.integers(1, 70, 2)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        p = np.stack([xx * 3.1, yy * 2.7, 5 * np.sin(xx / 4)], -1) + rng.normal(0, 0.3, (H, W, 3)).astype(np.float32)
        p[rng.random((H, W)) < rng.uniform(0, 0.5)] = np.nan
        S = Surface(p * rng.choice([1.0, 1000.0]))
        assert S.area() == S._area_reference()


def test_banded_rasterizer_identical():
    from unroller import _kernels as K

    rng = np.random.default_rng(11)
    for trial in range(12):
        nT = int(rng.integers(1, 3000))
        H, W = int(rng.integers(1, 120)), int(rng.integers(1, 120))
        g = rng.uniform(-10, max(H, W) + 10, (nT, 3, 2))
        if trial % 3 == 0:  # many exact ties: grid-aligned shared edges and duplicated triangles
            g = np.round(g)
            g[nT // 2:] = g[:nT - nT // 2]
        g[::17, 2] = g[::17, 1]  # degenerate
        V = rng.normal(size=(50, 3))
        T = rng.integers(0, 50, (nT, 3))
        ref_o = np.full((H, W, 3), np.nan, np.float32)
        ref_b = np.full((H, W), -np.inf, np.float32)
        K.rasterize_tris(g, T, V, ref_o, ref_b)
        for band in (1, 3, 8, 1000):
            o = np.full((H, W, 3), np.nan, np.float32)
            b = np.full((H, W), -np.inf, np.float32)
            K.rasterize_tris_banded(g, T, V, o, b, band)
            assert np.array_equal(ref_o, o, equal_nan=True) and np.array_equal(ref_b, b)
