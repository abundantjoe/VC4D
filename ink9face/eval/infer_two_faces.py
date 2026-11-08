"""Run a public 9 um ink checkpoint on both layer orders of rendered windows (CPU is fine; needs villa).

    python eval/infer_two_faces.py --villa <villa clone> --ref-ckpt <ink_9um step-075000.pth> \
        --model reader_v2=<reader-v2-step040000.pth>:model --stacks <dir of window .npz with 'st'> \
        --windows w1,w2 --out <maps dir>

Model construction uses villa's own loader (`koine_machines.inference.infer.build_repo_training_model_bundle`) on a
reference ink_9um checkpoint, then loads the third-party tensors (weights_only, no pickled code).  Every public
`vesuvius_unet_3d_stem_2d` 9 um checkpoint we tested has the same architecture, so the reference only supplies the
module graph.  Key handling: `path:key` takes ck[key] (`model`, `ema_model`); a `.safetensors` path is a bare state dict.

Window stack: `st` (25, H, W) uint8 rendered at offsets -12..+12 voxels along the mesh normal, with `meta.flipped`
saying whether `st` is the reversed native (+n) order.  Native = layer index increasing along +n (vc_render_tifxyz).
The model input is the centred in_chans slab (shift `--shift` voxels), patches of roi size at stride roi/2 with a
Hann blend, robust normalisation per patch, bf16 autocast on CPU.  Output per window: <out>/<model>/<w>.npz with
uint8 maps `native` and `anti_normal` (= the same slab fed in reversed order).
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def load_model(villa, ref_ckpt, spec):
    import torch
    sys.path.insert(0, os.path.join(villa, "ink-detection"))
    sys.path.insert(0, os.path.join(villa, "vesuvius", "src"))
    from koine_machines.inference import infer as I
    B = I.build_repo_training_model_bundle(torch.load(ref_ckpt, map_location="cpu", weights_only=False), ref_ckpt)
    path, _, key = spec.partition(":") if not spec.endswith(".safetensors") else (spec, "", "")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        d = torch.load(path, map_location="cpu", weights_only=True)
        sd = d[key] if key and key in d else d
    sd = {("model." + k if not k.startswith("model.") else k): v.float() for k, v in sd.items() if hasattr(v, "shape")}
    res = B.model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if "num_batches" not in k]
    if res.unexpected_keys or missing:
        raise RuntimeError(f"state dict mismatch: unexpected {res.unexpected_keys[:3]} missing {missing[:3]}")
    return B.model.eval(), int(B.roi_size), int(B.in_chans)


def predict(net, roi, vol3, bf16=True):
    import torch
    from vesuvius.image_proc.intensity.normalization import normalize_robust
    hw = np.outer(np.hanning(roi + 2)[1:-1], np.hanning(roi + 2)[1:-1]).astype(np.float32)
    _, H, W = vol3.shape
    acc = np.zeros((H, W), np.float32); ws = np.zeros((H, W), np.float32); s = roi // 2
    ys = list(range(0, max(1, H - roi) + 1, s)); xs = list(range(0, max(1, W - roi) + 1, s))
    if ys[-1] != max(0, H - roi): ys.append(max(0, H - roi))
    if xs[-1] != max(0, W - roi): xs.append(max(0, W - roi))
    batch, pos = [], []

    def flush():
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            p = torch.sigmoid(net(torch.from_numpy(np.stack(batch))[:, None]).float()).numpy().reshape(len(batch), -1, roi, roi)[:, 0]
        for (y, x), pp in zip(pos, p):
            acc[y:y + roi, x:x + roi] += pp * hw; ws[y:y + roi, x:x + roi] += hw
        batch.clear(); pos.clear()
    for y in ys:
        for x in xs:
            pt = vol3[:, y:y + roi, x:x + roi]
            if pt.shape[1:] != (roi, roi) or not pt.any():
                continue
            batch.append(normalize_robust(pt.astype(np.float32)).astype(np.float32)); pos.append((y, x))
            if len(batch) == 8:
                flush()
    if batch:
        flush()
    return np.where(ws > 0, acc / np.maximum(ws, 1e-6), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--villa", required=True)
    ap.add_argument("--ref-ckpt", required=True)
    ap.add_argument("--model", required=True, help="name=path[:key]")
    ap.add_argument("--stacks", required=True)
    ap.add_argument("--windows", required=True, help="comma-separated window ids (file stems in --stacks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    name, spec = a.model.split("=", 1)
    net, roi, dn = load_model(a.villa, a.ref_ckpt, spec)
    lo = 12 - dn // 2 + a.shift
    od = os.path.join(a.out, name); os.makedirs(od, exist_ok=True)
    for w in a.windows.split(","):
        out = os.path.join(od, w + ".npz")
        res = dict(np.load(out)) if os.path.exists(out) else {}
        if "native" in res and "anti_normal" in res:
            continue
        Z = np.load(os.path.join(a.stacks, w + ".npz"))
        meta = json.loads(str(Z["meta"])); meta = meta.get("old_meta", meta)
        st = Z["st"]; native = st[::-1] if meta.get("flipped") else st
        slab = np.ascontiguousarray(native[lo:lo + dn])
        for k, x in (("native", slab), ("anti_normal", np.ascontiguousarray(slab[::-1]))):
            if k in res:
                continue
            t0 = time.time()
            res[k] = np.clip(np.rint(predict(net, roi, x) * 255), 0, 255).astype(np.uint8)
            np.savez_compressed(out + ".tmp.npz", **res); os.replace(out + ".tmp.npz", out)
            print(name, w, k, f"{time.time() - t0:.0f}s", flush=True)
    print("DONE", name, flush=True)


if __name__ == "__main__":
    main()
