"""40-generation unroller trace on public Scroll 1 (standardized zarr), seeded on a vertex of segment 20230504093154.
Reports cold (network) / warm timings for the default (C2) tracer and LEGACY, and how much of the trace lies on
the official segment (distance to nearest official vertex)."""
import sys, time, json, os
# requires: pip install -e .
import numpy as np
from unroller.surface import Surface
from unroller.volume import Volume
from unroller import trace as TR
from scipy.spatial import cKDTree
D, OUT = sys.argv[1], sys.argv[2]
URL = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr"
s = Surface.load(D + "/seg")
P = s.points[2024:2536, 384:896].reshape(-1, 3); P = P[np.isfinite(P[:, 0])]
tree = cKDTree(P)
R = {"load_avg_start": os.getloadavg()}
for k, (r, c) in enumerate([(2280, 640), (2150, 500)]):
    seed = s.points[r, c].astype(float)
    for name, kw in (("C2", {}), ("LEGACY", TR.LEGACY)):
        v = Volume(URL, disk_cache=D + "/zcache_trace")
        ts = []
        for rep in range(3):
            t = time.perf_counter(); out = TR.trace(v, seed, generations=40, **kw); ts.append(time.perf_counter() - t)
        surf = out[0] if isinstance(out, tuple) else out
        Q = surf.points[np.isfinite(surf.points[..., 0])]
        d, _ = tree.query(Q)
        R[f"seed{k}_{name}"] = {"times_s(first=cold+JIT,then warm)": ts, "vertices": int(len(Q)),
                                "frac_within_2vx_of_official": float((d < 2).mean()), "median_dist_vx": float(np.median(d)),
                                "bytes_fetched": v.stats()["bytes_fetched"]}
        print(k, name, R[f"seed{k}_{name}"], flush=True)
R["load_avg_end"] = os.getloadavg()
json.dump(R, open(OUT, "w"), indent=1)
