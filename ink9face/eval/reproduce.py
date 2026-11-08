"""Recompute the face table from public data only.

    python eval/reproduce.py labels --cache CACHE            # download + transfer the 99 window labels (public bucket)
    python eval/reproduce.py table  --cache CACHE --maps MAPS --out results/face_eval.json [--check results/face_eval.json]
    python eval/reproduce.py all    --cache CACHE --maps MAPS --out ... --check ...

`labels` downloads, per segment, the label-volume mesh, the organizers' 2.2-2.4 um ink-detection tif, the coarse
8.64 / 9.362 um mesh and the volume transform from s3://vesuvius-challenge-open-data (anonymous HTTPS), transfers the
label onto each window crop listed in eval/windows.json with ink9face.labels, writes CACHE/labels/<window>.npz
(ink, acc, valid, validation) and then deletes the large label tif and label mesh (pass --keep to keep them).
Peak disk use is one segment (<= ~0.6 GB); about 3 GB is downloaded in total.

`table` scores every model map in MAPS (layout in face_table.py: <maps>/<model>/<window>.npz with native and
anti_normal) against those labels, excludes each model on the segments it was trained on (eval/exclusions.json,
from public training lists) and writes the same JSON as results/face_eval.json.  --check compares the summary with a
reference file and exits 1 on any difference larger than --tol.

MAPS comes from either the released maps bundle or eval/infer_two_faces.py (runs a public checkpoint on both layer
orders of rendered windows; needs a villa clone and the window stacks).  With --models the newest checkpoints
(reader_v2 reader_v3 d9v4C dense_native) are scored in a separate table (results/face_eval_newck.json).
"""
import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import face_table as FT  # noqa: E402
from ink9face import labels as L  # noqa: E402
from ink9face import face, tifxyz  # noqa: E402


def load_windows(path=None):
    return json.load(open(path or os.path.join(HERE, "windows.json")))["windows"]


def cmd_labels(a):
    W = load_windows(a.windows)
    segs = OrderedDict()
    for x in W:
        segs.setdefault((x["scroll"], x["seg"]), []).append(x)
    od = os.path.join(a.cache, "labels"); os.makedirs(od, exist_ok=True)
    rp = os.path.join(a.cache, "labels_report.json")
    report = json.load(open(rp)) if os.path.exists(rp) else {}
    for (sc, seg), ws in segs.items():
        todo = [x for x in ws if not os.path.exists(os.path.join(od, x["w"] + ".npz"))]
        if not todo:
            continue
        p = L.fetch_segment(sc, seg, os.path.join(a.cache, "dl"), label_name=ws[0].get("label_tif"))
        P9 = tifxyz.load_xyz(p["coarse_mesh"])
        for x in todo:
            raw, acc, valid, meta = L.transfer(p["coarse_mesh"], p["label_mesh"], p["label_tif"], p["transform"], p["label_um"],
                                               crop=tuple(x["crop"]))
            r0, c0, h, w = x["crop"]
            Pc = P9[r0:r0 + h + 1, c0:c0 + w + 1]
            axis_native = face.toward_axis_is_native(Pc, tifxyz.vc_normals(Pc), x["axis_xy"])
            v = meta["validation"]
            np.savez_compressed(os.path.join(od, x["w"] + ".npz"), ink=raw > 127, ink_raw=raw, acc=acc, valid=valid,
                                meta=json.dumps(dict(scroll=sc, seg=seg, crop=x["crop"], toward_axis_is_reversed=not axis_native,
                                                     T_source=meta["T_source"], validation=v)))
            report[x["w"]] = dict(passed=v["pass"], node3d_p95=v["node3d_vx"]["p95"], toward_axis_matches_list=(not axis_native) == x["toward_axis_is_reversed"])
            print(sc, x["w"], "PASS" if v["pass"] else "FAIL", f"node3d p95 {v['node3d_vx']['p95']:.3f} vx", flush=True)
        if not a.keep:   # keep only the coarse mesh (small); drop the label tif and the label mesh
            for f in ("label.tif", "x.tif", "y.tif", "z.tif"):
                fp = os.path.join(p["label_mesh"], f)
                if os.path.exists(fp):
                    os.remove(fp)
    json.dump(report, open(rp, "w"), indent=1)
    bad = [w for w, r in report.items() if not r["passed"] or not r["toward_axis_matches_list"]]
    print(len(report), "windows transferred;", len(bad), "failed the gate or disagree on the axis side:", bad[:5])
    return 1 if bad else 0


def build_rows(cache, maps, models, clean, W, excl):
    rows = {}
    for x in W:
        f = os.path.join(cache, "labels", x["w"] + ".npz")
        if not os.path.exists(f):
            continue
        Z = np.load(f)
        meta = json.loads(str(Z["meta"]))
        M = FT.load_maps(maps, x["w"], models)
        if not M:
            continue
        cont = [m for m in models if m != "trace" and FT.trained(excl, x["scroll"], m, x["seg"])]
        rows[x["w"]] = FT.score_window(x["scroll"], x["seg"], Z["ink"], Z["valid"] & Z["acc"], meta["toward_axis_is_reversed"],
                                       M, cont, models=models, clean=clean)
        print(x["scroll"], x["w"][:44], {k: round(v["reversed"]["auc"] - v["native"]["auc"], 3) for k, v in rows[x["w"]]["scores"].items()}, flush=True)
    return rows


def cmd_table(a):
    W = load_windows(a.windows)
    excl = FT.load_exclusions(a.exclusions)
    models = tuple(a.models) if a.models else FT.MODELS
    clean = tuple(m for m in models if m in FT.CLEAN) if not a.models else tuple(m for m in models if m != "trace")
    rows = build_rows(a.cache, a.maps, models, clean, W, excl)
    S = FT.summarise(rows, models)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(summary=S, rows=rows), open(a.out, "w"), indent=1)
    json.dump(rows, open(a.out.replace(".json", "_rows.json"), "w"))
    FT.print_summary(S)
    if a.check:
        ref = json.load(open(a.check))["summary"]
        d = FT.compare(S, ref, a.tol)
        print("CHECK vs", a.check, ":", "IDENTICAL within tol %g" % a.tol if not d else f"{len(d)} differences")
        for x in d[:40]:
            print("  ", x)
        return 1 if d else 0
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("labels", "table", "all"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--maps")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "face_eval_repro.json"))
    ap.add_argument("--check")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--windows")
    ap.add_argument("--exclusions")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    # only the labels step downloads (one segment on disk at a time); the table step reads maps already on disk
    if a.step in ("labels", "all") and \
            shutil.disk_usage(os.path.dirname(os.path.abspath(a.cache)) or ".").free < 1.5 * 2 ** 30:
        sys.exit("DISK_LOW: need > 1.5 GB free")
    rc = 0
    if a.step in ("labels", "all"):
        rc |= cmd_labels(a)
    if a.step in ("table", "all"):
        if not a.maps:
            sys.exit("--maps is required for the table")
        rc |= cmd_table(a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
