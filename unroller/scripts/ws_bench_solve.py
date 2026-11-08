"""WP3 benchmarks and formal gate measurements (results/ws_solve/gates.md).

usage: python scripts/ws_bench_solve.py gates OUT.json          # G6, G7, G8 (BK = scipy), G10 on seeds 1000+
       python scripts/ws_bench_solve.py scale N OUT.json [--obs K] [--slabs S] [--backend bk] [--cold]
Run single-threaded for the 1-CPU acceptance:  OMP_NUM_THREADS=1 taskset -c 0 python ...
"""
import json
import resource
import sys
import time

import numpy as np

import unroller.ws.solve as W
from unroller.ws.confidence import validate
from unroller.ws.solve import accuracy, sheetlets


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def gates(out):
    res = {}
    # G6: confidence proxy vs exact c*
    tr, e, bad, xyz = sheetlets(20000, 0.05, seed=1000, scale=10, return_xyz=True)
    x, info = W.solve(e, 20000, init="irls", backend="bk")
    t = time.perf_counter()
    v = validate(x, e, xyz, n_sample=1000, blocks=(256, 512, 1024), seed=1000, backend="bk")
    res["G6"] = {k: v[k] for k in v if not hasattr(v[k], "__len__")}
    res["G6"]["certified"] = info["certificate"]
    res["G6"]["wall_s"] = time.perf_counter() - t
    cs, p = v["cstar"], v["proxy"]
    res["G6"]["frac_cstar_zero"] = float(np.mean(cs < 1e-9))
    res["G6"]["frac_proxy_gt_2cstar_given_cstar_pos"] = float(np.mean(p[cs > 1e-9] > 2 * cs[cs > 1e-9] + 1e-6))
    res["G6"]["pass"] = bool(v["spearman"] >= 0.7 and v["frac_proxy_gt_2cstar"] <= 0.05)
    print("G6", res["G6"], flush=True)
    # G7: chirality
    def build(chi):
        _, ee, _ = sheetlets(20000, 0.05, seed=1001, chi=chi, scale=10)
        return ee, 20000
    chi, ratio, inf = W.choose_chirality(build, backend="bk")
    res["G7"] = {"chi": chi, "total_ratio": ratio, "informative_ratio": inf.get("informative_ratio"),
                 "informative_edges": inf.get("informative_edges"), "decided": inf["decided"],
                 "pass_registered(total ratio>=1.5)": bool(chi == 1 and ratio >= 1.5),
                 "pass_informative": bool(chi == 1 and inf["decided"])}
    print("G7", res["G7"], flush=True)
    # G8 + G10: BK vs scipy energies; accuracy cold / stitched / polished on 1e5-node graphs
    res["G10"] = []
    for seed, pw in ((1002, 0.03), (1003, 0.05), (1004, 0.10)):
        tr, e, bad, xyz = sheetlets(100_000, pw, seed=seed, scale=10, return_xyz=True)
        n = len(tr)
        r = {"seed": seed, "p_wrong": pw, "edges": len(e), "wrong_frac": float(bad.mean())}
        xc, ic = W.solve(e, n, init="tree", backend="bk")
        r.update({"cold_E": ic["energy"], "cold_cert": ic["certificate"], "cold_iters": ic["iters"],
                  "cold_s": ic["wall_s"], "acc_tree_init": accuracy(W.tree_init(W.Problem(e, n)), tr),
                  "acc_cold": accuracy(xc, tr), "truth_E": W.energy(tr, e)})
        if seed == 1003:
            xs_, is_ = W.solve(e, n, init="tree", backend="scipy")
            r["G8_scipy_equal_bk"] = bool(is_["energy_int"] == ic["energy_int"])
            r["scipy_s"] = is_["wall_s"]
        slab = np.minimum((xyz[:, 2] / (xyz[:, 2].max() / 4)).astype(int), 3)
        sol = W.solve_slabs(e, n, slab, backend="bk")
        m = sol.meta
        Pp = W.Problem(e, n)
        # stitched accuracy needs the stitched x: re-derive from meta is not stored, recompute quickly
        r.update({"stitched_E": m["stitched_energy"], "polished_E": m["energy"], "polished_cert": m["certificate"],
                  "polish_iters": m["polish_iters"], "slab_s": m["slab_s"], "block_s": m["block_s"],
                  "polish_s": m["polish_s"], "acc_polished": accuracy(sol.x, tr),
                  "E_equal_cold": bool(abs(m["energy"] - ic["energy"]) < 1e-6 * max(1, ic["energy"]))})
        res["G10"].append(r)
        print("G10", r, flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)


def scale(N, out, obs=1.0, slabs=8, backend="bk", cold=False):
    t = time.perf_counter()
    tr, e, bad, xyz = sheetlets(N, 0.05, seed=1005, scale=10, obs=obs, return_xyz=True)
    r = {"nodes": N, "edges": len(e), "wrong_frac": float(bad.mean()), "gen_s": time.perf_counter() - t,
         "backend": backend}
    print(r, flush=True)
    slab = np.minimum((xyz[:, 2] / (xyz[:, 2].max() / slabs)).astype(int), slabs - 1)
    sol = W.solve_slabs(e, N, slab, backend=backend, slab_init="tree",
                        log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
    m = sol.meta
    r.update({k: m[k] for k in ("stitched_energy", "energy", "certificate", "stopped_by", "polish_iters", "slab_s",
                                "block_s", "polish_s", "wall_s", "blocks", "block_edges", "slab_all_certified",
                                "scale", "quant_err", "quant_bound", "max_node_cap")})
    r["acc_polished"] = accuracy(sol.x, tr)
    r["truth_E"] = W.energy(tr, e)
    r["max_rss_gb"] = rss_gb()
    print(r, flush=True)
    if cold:
        x, ic = W.solve(e, N, init="tree", backend=backend, log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
        r.update({"cold_s": ic["wall_s"], "cold_iters": ic["iters"], "cold_cert": ic["certificate"],
                  "cold_E": ic["energy"], "acc_cold": accuracy(x, tr), "max_rss_gb": rss_gb()})
        print(r, flush=True)
    json.dump(r, open(out, "w"), indent=1, default=float)


def round2(what, seed, out):
    """G6b (tie-flagged confidence, 1e5-node graph) or G7b (informative-edge chirality) for one seed."""
    if what == "G6b":
        tr, e, bad, xyz = sheetlets(100_000, 0.05, seed=seed, scale=10, return_xyz=True)
        x, info = W.solve(e, len(tr), init="irls", backend="bk")
        t = time.perf_counter()
        v = validate(x, e, xyz, n_sample=1000, blocks=(256, 512, 1024), seed=seed, backend="bk", exclude_ties=True)
        r = {k: v[k] for k in v if not hasattr(v[k], "__len__")}
        cs, p = v["cstar"], v["proxy"]
        r.update({"seed": seed, "certified": info["certificate"], "wall_s": time.perf_counter() - t,
                  "frac_cstar_zero": float(np.mean(cs < 1e-9)),
                  "pass": bool(v["spearman"] >= 0.7 and v["frac_proxy_gt_2cstar"] <= 0.05)})
    else:
        pw = 0.05 if seed <= 2102 else 0.10

        def build(chi):
            _, ee, _ = sheetlets(20000, pw, seed=seed, chi=chi, scale=10)
            return ee, 20000
        chi, ratio, inf = W.choose_chirality(build, backend="bk")
        r = {"seed": seed, "p_wrong": pw, "chi": chi, "total_ratio": ratio, "informative_ratio": inf.get("informative_ratio"),
             "informative_edges": inf.get("informative_edges"), "decided": inf["decided"],
             "pass": bool(chi == 1 and inf.get("informative_ratio", 0) >= 1.5 and inf["decided"])}
    print(what, r, flush=True)
    json.dump(r, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "round2":
        round2(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == "gates":
        gates(sys.argv[2])
    else:
        a = sys.argv
        scale(int(float(a[2])), a[3], obs=float(a[a.index("--obs") + 1]) if "--obs" in a else 1.0,
              slabs=int(a[a.index("--slabs") + 1]) if "--slabs" in a else 8,
              backend=a[a.index("--backend") + 1] if "--backend" in a else "bk", cold="--cold" in a)
