"""Render benchmark on public Scroll 1 (PHercParis4) segment 20230504093154, standardized 7.91um OME-Zarr.
unroller render (remote cold / disk-cache warm / in-memory kernel) vs torch grid_sample and scipy map_coordinates
on the identical sample positions.  usage: python -I bench_render.py DATA OUT.json X0 Y0 W H NLAYERS"""
import sys, time, json, os
# requires: pip install -e .
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]
X0, Y0, W, H, NL = map(int, sys.argv[3:8])
from unroller.surface import Surface
from unroller.volume import Volume, ArrayVolume
from unroller.render import render, render_positions
import numba
URL = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr"
R = {"conditions": {"crop_xywh": [X0, Y0, W, H], "layers": NL, "numba_threads": numba.get_num_threads(),
                    "load_avg_start": os.getloadavg()}}
s = Surface.load(D + "/seg")
crop = (X0, Y0, W, H)
t = time.perf_counter(); pts, nrm, offs = render_positions(s, NL, crop=crop); R["positions_s"] = time.perf_counter() - t
ok = np.isfinite(pts[..., 0]); R["valid_frac"] = float(ok.mean())
cache = D + "/zcache"
# 1) cold remote render (chunks fetched over HTTPS into an empty disk cache)
v = Volume(URL, disk_cache=cache)
t = time.perf_counter(); a_cold = render(s, v, NL, crop=crop); R["unroller_render_remote_cold_s"] = time.perf_counter() - t
R["remote_stats"] = v.stats()
# 2) warm: fresh Volume, chunks from the local disk cache (decode + sample, no network)
v2 = Volume(URL, disk_cache=cache)
t = time.perf_counter(); a_warm = render(s, v2, NL, crop=crop); R["unroller_render_diskcache_s"] = time.perf_counter() - t
t = time.perf_counter(); a_warm2 = render(s, v2, NL, crop=crop); R["unroller_render_ramcache_s"] = time.perf_counter() - t
R["cold_eq_warm"] = bool(np.array_equal(a_cold, a_warm, equal_nan=True) and np.array_equal(a_warm, a_warm2, equal_nan=True))
# dense block covering all sample positions
q = pts[ok][:, None, :] + offs[None, :, None] * nrm[ok][:, None, :]
q = q.reshape(-1, 3)
lo = np.floor(q.min(0)).astype(int) - 2; hi = np.ceil(q.max(0)).astype(int) + 3  # xyz
t = time.perf_counter(); blk = v2.levels[0].read(lo[2], hi[2], lo[1], hi[1], lo[0], hi[0]); R["block_read_s"] = time.perf_counter() - t
R["block_zyx"] = list(blk.shape)
P = pts[ok] - lo; N = nrm[ok]
npts = len(P) * NL; R["n_samples"] = int(npts)
av = ArrayVolume(blk)
def tm(f, rep=3):
    f(); b = []
    for _ in range(rep):
        t = time.perf_counter(); r = f(); b.append(time.perf_counter() - t)
    return r, min(b)
ref = a_warm[:, ok]  # (NL, n) unroller remote render values at valid pixels
for th in (1, 2):
    numba.set_num_threads(th)
    u, tu = tm(lambda: av.sample_layers(P, N, offs))
    R[f"unroller_inmem_t{th}_s"] = tu; R[f"unroller_inmem_t{th}_Mpts_s"] = npts / tu / 1e6
R["unroller_inmem_eq_remote"] = bool(np.array_equal(u, ref))
# baseline positions materialised (zyx, as both libraries need); time of materialisation reported separately
t = time.perf_counter()
Q = (P[None, :, :] + offs[:, None, None] * N[None, :, :]).reshape(-1, 3)
R["materialise_positions_s"] = time.perf_counter() - t
import scipy.ndimage as ndi
bf = blk.astype(np.float32)
zyx = np.ascontiguousarray(Q[:, ::-1].T)
sp, ts = tm(lambda: ndi.map_coordinates(bf, zyx, order=1, mode="constant", cval=0.0, prefilter=False), rep=1)
R["scipy_map_coordinates_s"] = ts; R["scipy_Mpts_s"] = npts / ts / 1e6
sp = sp.reshape(NL, -1)
R["scipy_vs_unroller_maxabs"] = float(np.abs(sp - u).max()); R["scipy_vs_unroller_meanabs"] = float(np.abs(sp - u).mean())
import torch
vol_t = torch.from_numpy(bf)[None, None]
Dz, Hy, Wx = bf.shape
g = np.empty_like(Q, dtype=np.float32)
g[:, 0] = 2 * Q[:, 0] / (Wx - 1) - 1; g[:, 1] = 2 * Q[:, 1] / (Hy - 1) - 1; g[:, 2] = 2 * Q[:, 2] / (Dz - 1) - 1
gt = torch.from_numpy(g).view(1, -1, 1, 1, 3)
for th in (1, 2):
    torch.set_num_threads(th)
    with torch.no_grad():
        tg, tt = tm(lambda: torch.nn.functional.grid_sample(vol_t, gt, mode="bilinear", padding_mode="zeros", align_corners=True), rep=2)
    R[f"torch_grid_sample_t{th}_s"] = tt; R[f"torch_t{th}_Mpts_s"] = npts / tt / 1e6
tg = tg.numpy().reshape(NL, -1)
R["torch_vs_unroller_maxabs"] = float(np.abs(tg - u).max()); R["torch_vs_unroller_meanabs"] = float(np.abs(tg - u).mean())
R["torch_vs_unroller_frac_gt_0.5"] = float((np.abs(tg - u) > 0.5).mean())
np.save(D + "/render_crop.npy", a_warm[NL // 2])
R["load_avg_end"] = os.getloadavg()
json.dump(R, open(OUT, "w"), indent=1); print(json.dumps(R, indent=1))
