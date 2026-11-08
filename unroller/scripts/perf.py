"""Pipeline speed harness: fixed workloads, wall time per stage, plus output fingerprints so that
every optimisation can be checked for identical (or numerically equivalent) results.

usage: python scripts/perf.py OUT.json [--remote] [--lean]

  --remote  also benchmark fetching Scroll 1 chunks over HTTP (MB/s, chunks/s; serial vs parallel)
  --lean    also measure lossless compression ratios on real Scroll 1 data (256^3 cube, 65-layer stack)
"""
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

from unroller.flatten import flatten
from unroller.phantom import Phantom
from unroller.render import render
from unroller.resample import flat_surface
from unroller.trace import trace
from unroller.volume import ArrayVolume, Volume, write_ome_zarr

CACHE = os.path.join(tempfile.gettempdir(), "unroller_perf_phantom.npz")
SCROLL1 = ("https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/"
           "54keV_7.91um_Scroll1A.zarr")


def fp(a):
    a = np.nan_to_num(np.asarray(a, np.float64), nan=-1)
    return {"sum": float(a.sum()), "abs": float(np.abs(a).sum()), "md5_r3": hashlib.md5(np.round(a, 3).tobytes()).hexdigest()[:10]}


def timeit(f, reps=1):
    best = np.inf
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best, out


def main(path):
    res = {}
    P = Phantom.random(1)
    if os.path.exists(CACHE):
        vol = np.load(CACHE)["a"]
    else:
        vol = P.volume()
        np.savez_compressed(CACHE, a=vol)
    V = ArrayVolume(vol)
    d = tempfile.mkdtemp()
    write_ome_zarr(f"{d}/v.zarr", vol, chunks=(64, 64, 64))
    Z = Volume(f"{d}/v.zarr")
    rng = np.random.default_rng(0)
    pts = rng.uniform([2, 2, 2], [189, 189, 61], (2_000_000, 3))
    V.sample(pts)  # warm: the first call loads the compiled kernel from the numba cache (~0.4 s)
    t, a = timeit(lambda: V.sample(pts), 3)
    res["sample_array_Mpts_per_s"] = len(pts) / t / 1e6
    res["fp_sample_array"] = fp(a)
    Z.sample(pts[:1000])  # warm metadata
    t, b = timeit(lambda: Z.sample(pts), 2)
    res["sample_zarr_Mpts_per_s"] = len(pts) / t / 1e6
    res["sample_zarr_equals_array"] = bool(np.allclose(a, b, atol=1e-4))
    res["sample_zarr_bitexact"] = bool(np.array_equal(a, b, equal_nan=True))
    # coherent access: a dense raster sweep through the volume (render / trace-like locality)
    yy, xx = np.mgrid[2:189:0.0935, 2:189:0.0935][:, :2000, :1000]
    coh = np.stack([xx.ravel(), yy.ravel(), 2 + 59 * (xx.ravel() - 2) / 187], 1)
    Z.sample(coh)
    t, c = timeit(lambda: Z.sample(coh), 3)
    res["sample_zarr_coherent_Mpts_per_s"] = len(coh) / t / 1e6
    res["fp_sample_coherent"] = fp(c)
    res["sample_zarr_coherent_bitexact"] = bool(np.array_equal(V.sample(coh), c, equal_nan=True))
    # cold random access over many small chunks (fetch + decode + sample; 32^3 chunks -> 72 chunks)
    write_ome_zarr(f"{d}/v32.zarr", vol, chunks=(32, 32, 32))
    t, c = timeit(lambda: Volume(f"{d}/v32.zarr").sample(pts), 3)
    res["sample_zarr_cold_random_Mpts_per_s"] = len(pts) / t / 1e6
    res["sample_zarr_cold_bitexact"] = bool(np.array_equal(a, c, equal_nan=True))
    seed = P.point(0.5 * P.length, 32)
    t, T = timeit(lambda: trace(V, seed, step=3.0, generations=40, umbilicus=P.center, up=P.R[:, 2],
                                bounds=((2, 2, 2), (189, 189, 61))))
    res["trace_s"] = t  # first call: includes loading kernels from the numba cache
    t, T2 = timeit(lambda: trace(V, seed, step=3.0, generations=40, umbilicus=P.center, up=P.R[:, 2],
                                 bounds=((2, 2, 2), (189, 189, 61))), 3)
    res["trace_warm_s"] = t
    res["trace_repeat_identical"] = bool(np.array_equal(T.points, T2.points, equal_nan=True))
    res["trace_vertices"] = int(T.valid.sum())
    res["fp_trace"] = fp(T.points)
    res.update(snap_bench(V, P, seed, T))
    S = P.surface(step=1.0)
    t, uv = timeit(lambda: flatten(S))
    res["flatten_s"] = t
    res["flatten_vertices"] = int(S.valid.sum())
    res["fp_flatten"] = fp(uv)
    t, F = timeit(lambda: flat_surface(S, uv, out_scale=1.0))
    res["resample_s"] = t
    res["resample_pixels"] = int(np.prod(F.shape))
    res["fp_resample"] = fp(F.points)
    render(F, V, n_layers=21)  # warm: first call loads the compiled kernel from the numba cache
    t, st = timeit(lambda: render(F, V, n_layers=21), 3)
    res["render_s"] = t
    res["render_Mvox_per_s"] = st.size / t / 1e6
    res["fp_render"] = fp(st)
    t, st = timeit(lambda: render(F, Z, n_layers=21), 3)
    res["render_zarr_Mvox_per_s"] = st.size / t / 1e6
    res["fp_render_zarr"] = fp(st)
    from unroller.render import render_stream
    import shutil

    import numcodecs

    def rs():
        shutil.rmtree(f"{d}/rs.zarr", ignore_errors=True)
        return render_stream(F, Z, f"{d}/rs.zarr", n_layers=65, tile=128)
    rs()
    t, rep = timeit(rs, 3)
    res["render_stream_65_s"] = t
    res["render_stream_Mvox_per_s"] = rep["uncompressed_bytes"] / t / 1e6
    res["render_stream_compression_ratio"] = rep["uncompressed_bytes"] / max(rep["bytes_written"], 1)
    m = json.load(open(f"{d}/rs.zarr/.zarray"))
    codec = numcodecs.get_codec(m["compressor"])
    h = hashlib.md5()
    for dp, _, fns in sorted(os.walk(f"{d}/rs.zarr")):
        for fn in sorted(fns):
            if not fn.startswith("."):
                h.update(os.path.relpath(os.path.join(dp, fn), d).encode())
                h.update(codec.decode(open(os.path.join(dp, fn), "rb").read()))
    res["fp_render_stream_md5"] = h.hexdigest()[:10]
    if "--remote" in sys.argv:
        res["remote"] = remote_bench()
    if "--lean" in sys.argv:
        res["lean"] = lean_bench(d)
    print(json.dumps(res, indent=1))
    json.dump(res, open(path, "w"), indent=1)


def snap_bench(V, P, seed, T_ref):
    """The numpy reference trace._snap_ms vs the compiled unroller.snap_fast.snap_ms that trace() runs: the exact
    calls a 40-generation trace makes (recorded), and a full trace with the reference patched in (must give the
    same surface as T_ref)."""
    import unroller.snap_fast as sf
    import unroller.trace as tr

    fast, orig, calls = sf.snap_ms, tr._snap_ms, []

    def rec(vol, p, n, radius, sigma, **kw):
        calls.append((p.copy(), n.copy(), radius, sigma, kw))
        return fast(vol, p, n, radius, sigma, **kw)
    kw = dict(step=3.0, generations=40, umbilicus=P.center, up=P.R[:, 2], bounds=((2, 2, 2), (189, 189, 61)))
    sf.snap_ms = rec
    try:
        tr.trace(V, seed, **kw)
    finally:
        sf.snap_ms = fast
    out = {"snap_calls_in_trace": len(calls), "snap_median_points": float(np.median([len(c[0]) for c in calls]))}
    for f, name in ((orig, "orig"), (fast, "fast")):
        f(V, *calls[0][:4], **calls[0][4])  # warm
        t = time.perf_counter()
        for _ in range(3):
            res = [f(V, p, n, r, s, **dict(k, return_width=True)) for p, n, r, s, k in calls]
        out[f"snap_{name}_calls_per_s"] = 3 * len(calls) / (time.perf_counter() - t)
        if name == "orig":
            ref = res
    out["snap_fast_bitexact"] = all(np.array_equal(x, y, equal_nan=True) for a, b in zip(ref, res) for x, y in zip(a, b))
    rng = np.random.default_rng(0)  # a large batch (a long trace front)
    p = rng.uniform([20, 20, 10], [170, 170, 50], (5000, 3))
    n = rng.normal(size=(5000, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    k = dict(calls[-1][4])
    for f, name in ((orig, "orig"), (fast, "fast")):
        t, _ = timeit(lambda: f(V, p, n, calls[-1][2], calls[-1][3], **k), 3)
        out[f"snap_5000pts_{name}_ms"] = t * 1e3
    sf.snap_ms = orig
    try:
        t, T = timeit(lambda: tr.trace(V, seed, **kw), 3)
    finally:
        sf.snap_ms = fast
    out["trace_snapref_warm_s"] = t
    out["fp_trace_snapref"] = fp(T.points)
    out["trace_snapref_identical"] = bool(np.array_equal(T.points, T_ref.points, equal_nan=True))
    return out


def remote_bench(n=48):
    """Fetch n distinct level-0 chunks (128^3, ~0.5-1.5 MB compressed each) of Scroll 1 three ways:
    a fresh connection per request (no reuse), serial with keep-alive, and the parallel keep-alive pool
    used by Volume.prefetch.  Disjoint chunk sets so no run is served from a cache warmed by another."""
    import requests

    from unroller.volume import _Store

    out = {}
    base = (40, 28, 30)  # around the middle of the scroll (z, y, x chunk indices)
    keys = [(base[0] + i // 16, base[1] + (i // 4) % 4, base[2] + i % 4) for i in range(3 * n)]
    sets = {"new_connection_serial": keys[:n], "keepalive_serial": keys[n:2 * n], "parallel_pool": keys[2 * n:]}
    for name, ks in sets.items():
        st = _Store(SCROLL1 + "/0")
        t = time.perf_counter()
        if name == "new_connection_serial":
            nbytes = sum(len(requests.get(f"{SCROLL1}/0/{'/'.join(map(str, k))}", timeout=60).content) for k in ks)
        elif name == "keepalive_serial":
            nbytes = sum(len(st.get("/".join(map(str, k))) or b"") for k in ks)
        else:
            from concurrent.futures import ThreadPoolExecutor

            from unroller.volume import IO_WORKERS

            with ThreadPoolExecutor(IO_WORKERS) as ex:
                nbytes = sum(len(b or b"") for b in ex.map(lambda k: st.get("/".join(map(str, k))), ks))
        dt = time.perf_counter() - t
        out[name] = {"chunks": len(ks), "MB": nbytes / 1e6, "s": dt, "MB_per_s": nbytes / 1e6 / dt,
                     "chunks_per_s": len(ks) / dt}
    # end to end: cold sampling of a 1024 x 1024 plane on level 0 (fetch + decode + sample)
    V = Volume(SCROLL1)
    yy, xx = np.mgrid[0:1024, 0:1024]
    c = np.array([4100.0, 4000.0, 7300.0])
    pts = np.stack([c[0] + xx.ravel() * 0.95 + yy.ravel() * 0.3, c[1] + yy.ravel(), c[2] + xx.ravel() * 0.2], 1)
    t = time.perf_counter()
    s = V.sample(pts)
    dt = time.perf_counter() - t
    out["cold_plane_1M_pts"] = {"s": dt, "Mpts_per_s": len(pts) / dt / 1e6, **V.stats(), "fp": fp(s)}
    raw = [b for b in V.levels[0]._raw.values() if b]
    out["ram_chunk_cache_ratio"] = len(raw) * int(np.prod(V.levels[0].chunks)) / max(sum(map(len, raw)), 1)
    t = time.perf_counter()
    V.sample(pts)
    out["warm_plane_Mpts_per_s"] = len(pts) / (time.perf_counter() - t) / 1e6
    out["trace_readahead"] = remote_trace_bench()
    return out


def remote_trace_bench():
    """Cold trace (40 generations, step 3) on Scroll 1 over HTTP with and without chunk readahead.
    Run order alternates (A: off then on, B: on then off) so a server-side cache warmed by the first
    run cannot favour one setting; the traced surfaces must be identical."""
    seeds = {"A": ([4737, 2251, 5292], [3840.0, 2257.0, 5300]), "B": ([2747, 2263, 8296], [3840.0, 2257.0, 8300])}
    res = {}
    for name, (seed, umb) in seeds.items():
        for ra in ((False, True) if name == "A" else (True, False)):
            V = Volume(SCROLL1, readahead=ra)
            t = time.perf_counter()
            T = trace(V, np.array(seed, float), step=3.0, generations=40, umbilicus=umb)
            dt = time.perf_counter() - t
            s = V.stats()
            res[f"{name}_readahead_{ra}"] = {"s": dt, "vertices": int(T.valid.sum()), "fp": fp(T.points)["md5_r3"],
                                             "http_requests": s["http_requests"], "MB": s["bytes_fetched"] / 1e6}
    return res


def lean_bench(d):
    """Lossless compression ratios on real Scroll 1 data."""
    from unroller import store
    from unroller.render import render
    from unroller.surface import Surface

    V = Volume(SCROLL1)
    z0, y0, x0 = 5120, 3584, 3840
    cube = V.levels[0].read(z0, z0 + 256, y0, y0 + 256, x0, x0 + 256)
    out = {"cube_origin_zyx": [z0, y0, x0], "cube_md5": hashlib.md5(cube.tobytes()).hexdigest()[:10]}
    for name in ("cube256.zarr", "cube256.npz"):
        r = store.save(f"{d}/{name}", cube, chunks=(128, 128, 128))
        assert np.array_equal(store.load(f"{d}/{name}"), cube)
        out[name] = {k: r[k] for k in ("raw_bytes", "stored_bytes", "ratio", "seconds")}
    # the in-RAM compressed chunk cache: compressed bytes held vs decoded size of the same chunks
    L = V.levels[0]
    raw = sum(len(b) for b in L._raw.values() if b)
    dec = sum(1 for b in L._raw.values() if b) * int(np.prod(L.chunks))
    out["ram_chunk_cache"] = {"chunks": len(L._raw), "compressed_bytes": raw, "decoded_bytes": dec,
                              "ratio": dec / max(raw, 1)}
    # a 65-layer stack rendered from a plane through the cube (uint8 like render_stream writes)
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float32)
    P = np.stack([x0 + 28 + xx, y0 + 28 + yy, np.full_like(xx, z0 + 128) + 0.15 * xx], -1)
    S = Surface(P, (1.0, 1.0))
    st = np.clip(np.rint(np.nan_to_num(render(S, V, n_layers=65))), 0, 255).astype(np.uint8)
    for name in ("stack65.zarr", "stack65.npz"):
        r = store.save(f"{d}/{name}", st, chunks=(65, 128, 128))
        assert np.array_equal(store.load(f"{d}/{name}"), st)
        out[name] = {k: r[k] for k in ("raw_bytes", "stored_bytes", "ratio", "seconds")}
    import tifffile

    os.makedirs(f"{d}/tifs", exist_ok=True)
    for i, layer in enumerate(st):
        tifffile.imwrite(f"{d}/tifs/{i:02d}.tif", layer, compression="zlib")
    out["stack65_tif_zlib"] = store.size_report(f"{d}/tifs", st.nbytes)
    return out


if __name__ == "__main__":
    main(sys.argv[1])
