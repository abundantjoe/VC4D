"""unroller ARAP flatten vs libigl ARAP (community C++ library) vs the original VC UV map, on a crop of public
Scroll 1 segment 20230504093154 (tifxyz grid sampled on the official UV pixel grid).
usage: python -I bench_flatten.py DATA OUT.json X0 Y0 N"""
import sys, time, json, os
# requires: pip install -e .  and  pip install libigl==2.6.3
import numpy as np
from unroller.surface import Surface
from unroller.flatten import flatten, _clean, _align_to
from unroller.metrics import flatten_report, procrustes_rms
import igl
D, OUT = sys.argv[1], sys.argv[2]; X0, Y0, N = map(int, sys.argv[3:6])
full = Surface.load(D + "/seg_uv")
s = Surface(full.points[Y0:Y0 + N, X0:X0 + N])
R = {"crop_x0_y0_n": [X0, Y0, N], "load_avg_start": os.getloadavg()}
idx, V, T = s.triangles(); T = _clean(V, T)
R["vertices"] = int((idx >= 0).sum()); R["triangles"] = int(len(T))
rr, cc = np.nonzero(idx >= 0)
def summ(rep):
    k = ["triangles", "flipped_triangles", "area_ratio_2d_3d", "abs_log_area_distortion", "anisotropy_s1_over_s2", "l2_stretch"]
    return {x: rep[x] for x in k}
def grid_of(uvv):
    g = np.full(idx.shape + (2,), np.nan); g[rr, cc] = uvv; return g
# original VC UV map = the grid itself (col, row) in UV pixels; also its scale-normalised version
uv_vc = np.stack([cc, rr], 1).astype(np.float64)
rep = flatten_report(s, grid_of(uv_vc)); R["vc_uv"] = summ(rep)
sc = 1 / np.sqrt(rep["area_ratio_2d_3d"]); R["vc_uv_scale_normalised"] = summ(flatten_report(s, grid_of(uv_vc * sc)))
# unroller (defaults)
info = {}
t = time.perf_counter(); g_u = flatten(s, info=info); R["unroller_flatten_s"] = time.perf_counter() - t
R["unroller_info"] = {k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in info.items()}
R["unroller"] = summ(flatten_report(s, g_u))
uv_u = g_u[rr, cc].astype(np.float64)
# libigl ARAP (elements energy, one pinned vertex), init = scale-normalised VC UV map, as for a re-flatten
Vv = V[idx[rr, cc]].astype(np.float64) if V.shape[0] != len(rr) else V.astype(np.float64)
F = np.ascontiguousarray(T, dtype=np.int64)
b = np.array([0], np.int32); bc = (uv_vc[[0]] * sc)
U = uv_vc * sc
tot = 0.0; R["libigl"] = {}
data = igl.ARAPData(); data.energy = igl.ARAP_ENERGY_TYPE_ELEMENTS
t = time.perf_counter(); igl.arap_precomputation(np.ascontiguousarray(V, np.float64), F, 2, b, data)
R["libigl_precompute_s"] = time.perf_counter() - t; tot = R["libigl_precompute_s"]
done = 0
for it in (10, 40, 150):
    data.max_iter = it - done
    t = time.perf_counter(); U = igl.arap_solve(bc, data, U); dt = time.perf_counter() - t; tot += dt; done = it
    g = grid_of(U)
    R["libigl"][f"iter{it}"] = {"cum_s": tot, **summ(flatten_report(s, g)),
                                 "rms_vs_unroller_vx": float(np.sqrt(np.mean(np.sum((_align_to(U, uv_u) - uv_u) ** 2, 1))))}
R["load_avg_end"] = os.getloadavg()
json.dump(R, open(OUT, "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))
