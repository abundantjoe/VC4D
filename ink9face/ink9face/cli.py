"""ink9face command line.

  ink9face labels  ...   exact geometry-only label transfer (2.2 um labels -> 8.6 / 9.4 um mesh), from S3 or local files
  ink9face face    ...   which layer order to feed a 9 um ink model for a tifxyz mesh (+ optional two-face check)
  ink9face depth   ...   label-free depth-convention offset between two sets of rendered stacks
  ink9face score   ...   AUC / tolerant AUC / 37 um AUC / letter-scale r of an ink map against a label

Run `ink9face <command> -h` for options; examples in README.md.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def _read(path):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        z = np.load(path)
        return z[z.files[0]]
    import tifffile
    a = tifffile.imread(path)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., 0]
    return a


def _prob(a):
    a = a.astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def cmd_labels(a):
    from . import labels as L
    import tifffile
    if a.scroll and a.segment:
        p = L.fetch_segment(a.scroll, a.segment, a.cache)
        coarse, lmesh, ltif, tj, um = p["coarse_mesh"], p["label_mesh"], p["label_tif"], p["transform"], p["label_um"]
    else:
        coarse, lmesh, ltif, tj, um = a.coarse_mesh, a.label_mesh, a.label_tif, a.transform, a.label_um
        if not (coarse and lmesh and ltif):
            sys.exit("give --scroll/--segment or --coarse-mesh/--label-mesh/--label-tif")
    raw, acc, valid, meta = L.transfer(coarse, lmesh, ltif, tj, um or "", crop=tuple(a.crop) if a.crop else None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tifffile.imwrite(a.out + "_ink.tif", raw, compression="zlib")
    tifffile.imwrite(a.out + "_mask.tif", ((acc & valid) * 255).astype(np.uint8), compression="zlib")
    json.dump(meta, open(a.out + ".json", "w"), indent=1)
    v = meta["validation"]
    if not v["node3d_vx"] or not (acc & valid).any():
        print("FAIL: the crop has no valid mesh / label support"); return 3
    print(f"{a.out}_ink.tif {raw.shape}  ink share {float((raw > 127)[acc & valid].mean()):.3f}  "
          f"node3d median {v['node3d_vx']['median']:.3f} p95 {v['node3d_vx']['p95']:.3f} vx  -> {'PASS' if v['pass'] else 'FAIL'}  T: {meta['T_source']}")
    return 0 if v["pass"] else 3


def cmd_face(a):
    from . import tifxyz, face
    P = tifxyz.load_xyz(a.mesh)
    if a.crop:
        r0, c0, h, w = a.crop
        P = P[r0:r0 + h + 1, c0:c0 + w + 1]
    n = tifxyz.vc_normals(P)
    ok = np.all(np.isfinite(n), -1)
    nm = n[ok].mean(0); coh = float(np.linalg.norm(nm))
    out = dict(mesh=a.mesh, valid_nodes=int(ok.sum()), mean_normal=(nm / max(coh, 1e-9)).round(4).tolist(), normal_coherence=round(coh, 4),
               convention="+n = normalize(dP/dcol x dP/drow) (VC3D); native stack = layer index increasing along +n "
                          "(vc_render_tifxyz)",
               recommendation="anti_normal: feed the native stack REVERSED (reverse_layers=True) to ink_9um-family / KLAVIS / "
                              "Reader / TRACE / v8-in models on organizer meshes")
    if a.axis:
        out["toward_axis_is_native"] = face.toward_axis_is_native(P, n, a.axis)
        out["note_toward_axis"] = "toward-axis rule would feed " + ("native" if out["toward_axis_is_native"] else "reversed") + \
                                  " -> " + ("DISAGREES with" if out["toward_axis_is_native"] else "agrees with") + " anti_normal"
    if a.map_native and a.map_reversed:
        pn, pr = _prob(_read(a.map_native)), _prob(_read(a.map_reversed))
        m = _read(a.mask).astype(bool) if a.mask else np.ones(pn.shape, bool)
        pick, en, er = face.choose_band(pn, pr, m)
        out["two_face_check"] = dict(band_energy_native=en, band_energy_reversed=er, band_says=pick,
                                     agrees_with_anti_normal=pick == "reversed",
                                     note="weak per window (about 2/3 correct on PHerc0500P2); trust anti_normal unless a whole scroll votes against it")
    print(json.dumps(out, indent=1))


def cmd_depth(a):
    from . import depth
    def profs(d):
        P = []
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            st = z[a.stack_key]; v = z[a.valid_key] if a.valid_key in z.files else np.all(st > 0, 0)
            if a.reverse:
                st = st[::-1]
            P.append(depth.profile(st, v))
        return np.array(P)
    r = depth.compare(profs(a.ref), profs(a.tgt), a.max_shift)
    print(json.dumps(r, indent=1))


def cmd_score(a):
    from .metrics import score
    p = _prob(_read(a.map)); y = _read(a.label) > (127 if _read(a.label).max() > 1 else 0.5)
    m = (_read(a.mask) > 0) if a.mask else np.ones(p.shape, bool)
    print(json.dumps(score(p, m, y), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ink9face", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("labels", help="exact label transfer onto the coarse mesh")
    p.add_argument("--scroll", help="public scroll id, e.g. PHerc0500P2 (fetches from the open-data bucket)")
    p.add_argument("--segment", help="segment id, e.g. 20250611132918-z_dbg_gen_00302_inp_hr")
    p.add_argument("--cache", default="ink9face_cache")
    p.add_argument("--coarse-mesh"); p.add_argument("--label-mesh"); p.add_argument("--label-tif")
    p.add_argument("--transform", help="transform.json (coarse -> label volume); omit to fit an LSQ affine")
    p.add_argument("--label-um", default="", help="label voxel size tag, e.g. 2.215um (to accept the published transform)")
    p.add_argument("--crop", type=int, nargs=4, metavar=("R0", "C0", "ROWS", "COLS"), help="coarse grid cells; default whole mesh")
    p.add_argument("--out", required=True, help="output prefix")
    p.set_defaults(f=cmd_labels)
    p = sp.add_parser("face", help="layer order for a tifxyz mesh")
    p.add_argument("--mesh", required=True); p.add_argument("--crop", type=int, nargs=4, metavar=("R0", "C0", "ROWS", "COLS"))
    p.add_argument("--axis", type=float, nargs=2, metavar=("X", "Y"), help="scroll axis position (voxels) to compare with the toward-axis rule")
    p.add_argument("--map-native", help="model map on the native stack (.npy/.npz/.tif)")
    p.add_argument("--map-reversed", help="model map on the reversed stack")
    p.add_argument("--mask")
    p.set_defaults(f=cmd_face)
    p = sp.add_parser("depth", help="depth-convention offset between stack sets")
    p.add_argument("--ref", required=True, help="dir of reference .npz stacks (read on the correct face)")
    p.add_argument("--tgt", required=True, help="dir of target .npz stacks")
    p.add_argument("--stack-key", default="st"); p.add_argument("--valid-key", default="valid")
    p.add_argument("--reverse", action="store_true", help="reverse target stacks first"); p.add_argument("--max-shift", type=int, default=6)
    p.set_defaults(f=cmd_depth)
    p = sp.add_parser("score", help="metrics of an ink map against a label")
    p.add_argument("--map", required=True); p.add_argument("--label", required=True); p.add_argument("--mask")
    p.set_defaults(f=cmd_score)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
