"""Label-transfer evaluation on PUBLIC data.

    python eval/eval_labels.py --scratch <dir> --out results/labels_eval.json

(1) Registration residuals of the geometry-only transfer for every window of the 5 public scrolls (from the
    validation block written by the transfer, in coarse-scan voxels).
(2) PHerc0500P2: exact labels vs the previous labels (image registration: IoU fit of the mesh-valid mask to the ds8
    label support, a per-mesh 4-parameter affine).  Fraction of pixels whose label changes, and the AUC / letter-r of
    clean models (correct face, fc_maps) under both label sets (mesh bootstrap).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ink9face.metrics import mesh_paired, score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scratch", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
S = a.scratch
out = {"registration": {}, "old_vs_exact_0500P2": {}}
for sc, d in (("PHerc0500P2", "heldout_v2"), ("PHerc0814", "heldout0814_v2"), ("PHerc0343P", "heldout0343P_v2"),
              ("PHerc0009B", "heldout0009B_v2"), ("PHerc0139", "set3/heldout0139_v2")):
    rows = []
    for f in sorted(glob.glob(f"{S}/{d}/*_w*.npz")):
        m = json.loads(str(np.load(f)["meta"]))
        v = m["validation"]; mp = m["mapping"]
        rows.append(dict(w=os.path.basename(f)[:-4], node_med=v["node3d_vx"]["median"], node_p95=v["node3d_vx"]["p95"],
                         pix_med=v["pix3d_vx"]["median"], pix_p95=v["pix3d_vx"]["p95"], passed=v["pass"],
                         T_source=mp["T_source"][:9], est_vs_pub_p95=(mp.get("T_est_check", {}).get("est_vs_published_vx") or {}).get("p95")))
    arr = lambda k: np.array([r[k] for r in rows], float)
    evp = [r["est_vs_pub_p95"] for r in rows if r["est_vs_pub_p95"] is not None]
    out["registration"][sc] = dict(windows=len(rows), passed=int(sum(r["passed"] for r in rows)),
                                   node3d_p95_max=float(arr("node_p95").max()), node3d_median_median=float(np.median(arr("node_med"))),
                                   pix3d_median_median=float(np.median(arr("pix_med"))), pix3d_p95_median=float(np.median(arr("pix_p95"))),
                                   T_source=sorted({r["T_source"] for r in rows}), lsq_vs_published_p95_max=float(max(evp)) if evp else None, rows=rows)
    print(sc, {k: v for k, v in out["registration"][sc].items() if k != "rows"})

# (2) old vs exact labels on PHerc0500P2
res = {}
for f in sorted(glob.glob(f"{S}/fc_maps/*_w0.npz")):
    w = os.path.basename(f)[:-4]
    old = np.load(f"{S}/heldout/{w}.npz"); new = np.load(f"{S}/heldout_v2/{w}.npz"); M = np.load(f)
    mo = old["valid"] & old["acc"]; mn = new["valid"] & new["acc"]; both = mo & mn
    r = dict(seg=w.rsplit("_w", 1)[0], changed=float((old["ink"] != new["ink"])[both].mean()))
    for mod, ks in (("ink9avg", ("ink9_42", "ink9_43")), ("klavis", ("ink9_klavis_all7",))):
        p = np.mean([M[k].astype(np.float32) for k in ks], 0)
        r[mod] = dict(old=score(p, mo, old["ink"]), exact=score(p, mn, new["ink"]))
    res[w] = r
    print(w[:40], "changed %.3f" % r["changed"], {m: round(r[m]["exact"]["auc"] - r[m]["old"]["auc"], 4) for m in ("ink9avg", "klavis")}, flush=True)
ids = sorted(res); meshes = [res[w]["seg"] for w in ids]
summ = dict(n=len(ids), changed_median=float(np.median([res[w]["changed"] for w in ids])), changed_max=float(max(res[w]["changed"] for w in ids)))
for mod in ("ink9avg", "klavis"):
    for met in ("auc", "auc_37um", "letter"):
        o = np.array([res[w][mod]["old"][met] for w in ids]); e = np.array([res[w][mod]["exact"][met] for w in ids])
        summ[f"{mod}_{met}"] = dict(old=float(np.nanmean(o)), exact=float(np.nanmean(e)), exact_minus_old=mesh_paired(e - o, meshes))
out["old_vs_exact_0500P2"] = dict(summary=summ, rows=res)
print(json.dumps(summ, indent=1))
json.dump(out, open(a.out, "w"), indent=1)
