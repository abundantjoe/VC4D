"""unroller command line.

  unroller info     SEGMENT                       grid health report (json)
  unroller flatten  SEGMENT OUT [--iterations]    flatten to a new tifxyz + distortion report
  unroller render   SEGMENT VOLUME OUT [-n 65]    layered tif stack along normals
  unroller unroll   SEGMENT VOLUME OUT            flatten + render + all reports, one command
  unroller obj2tifxyz MESH.obj OUT                import an OBJ with UVs
  unroller phantom  OUT                           synthetic scroll for validation (volume + segment + truth)
  unroller validate                               run the phantom benchmark and print pass/fail
  unroller bench    -n 300                        score trace+regularize+flatten+render on random phantoms
  unroller trace    VOLUME OUT --seed X Y Z       grow a new segment (never-segmented scrolls)
  unroller refine   SEGMENT VOLUME OUT            sheet-centring calibration of a segment
  unroller regularize SEGMENT VOLUME OUT          cut folds, remove along-normal jitter, fill pinholes (traced)
  unroller budget   WORKDIR --max-gb 50           delete least-recently-used caches/outputs over a budget
  unroller crackle  LAYERS OUT [--labels PNG]     ink-side fine-valley texture score map + overlay of a layer stack
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _dump(obj, path=None):
    txt = json.dumps(obj, indent=1)
    if path:
        with open(path, "w") as f:
            f.write(txt + "\n")
    print(txt)


def _crop(a):
    return tuple(a.crop) if a.crop else None


def cmd_info(a):
    from .metrics import surface_report
    from .surface import Surface

    _dump(surface_report(Surface.load(a.segment), a.voxel_size))


def _flatten(a, seg, out):
    from .flatten import flatten
    from .metrics import flatten_report
    from .resample import flat_surface

    t = time.time()
    uv = flatten(seg, iterations=a.iterations, downsample=a.downsample, init=a.init, log=_log if a.verbose else None)
    _log(f"flattened in {time.time() - t:.1f}s")
    rep = flatten_report(seg, uv)
    flat = flat_surface(seg, uv, out_scale=a.out_scale)
    flat.save(out, flatten={"method": f"{a.init}+arap", "iterations": a.iterations, "downsample": a.downsample})
    _dump(rep, os.path.join(out, "flatten_report.json"))
    return flat


def cmd_flatten(a):
    from .surface import Surface

    _flatten(a, Surface.load(a.segment), a.out)


def _render(a, seg, vol, out):
    from .render import render_stream, zarr_layers_to_tifs

    t = time.time()
    rep = render_stream(seg, vol, os.path.join(out, "layers.zarr"), n_layers=a.layers, step=a.step, scale=a.scale,
                        level=a.level, crop=_crop(a), workers=a.workers, normal_sigma=a.normal_sigma,
                        progress=(lambda d, n: _log(f"render tile {d}/{n}")) if a.verbose else None)
    _log(f"rendered in {time.time() - t:.1f}s -> {rep['bytes_written'] / 2**20:.1f} MiB compressed "
         f"({rep['uncompressed_bytes'] / 2**20:.1f} MiB raw)")
    if a.tif:
        zarr_layers_to_tifs(os.path.join(out, "layers.zarr"), os.path.join(out, "layers"))
    if hasattr(vol, "stats"):
        rep["volume_io"] = vol.stats()
    _dump(rep, os.path.join(out, "stack_report.json"))
    return rep


def _volume(a, allow_16bit=False):
    from .volume import Volume

    if getattr(a, "volume_16bit", None):
        if not allow_16bit:  # trace / refine / sheet-continuity thresholds are calibrated on the 8-bit volume
            raise SystemExit("--volume-16bit is only supported by `render` (other commands assume the 8-bit "
                             "standardised volume's intensity scale)")
        from .tiffstack import CellGridVolume, TiffVolume

        if "/volume_grids/" in a.volume_16bit.rstrip("/") + "/":  # 500^3 TIFF cells: one request per chunk
            return CellGridVolume(a.volume_16bit, cache_gb=a.cache_gb, disk_cache=a.disk_cache,
                                  disk_cache_gb=a.disk_cache_gb)
        return TiffVolume(a.volume_16bit, cache_gb=a.cache_gb, disk_cache=a.disk_cache, disk_cache_gb=a.disk_cache_gb,
                          brick=a.brick)
    return Volume(a.volume, cache_gb=a.cache_gb, disk_cache=a.disk_cache, disk_cache_gb=a.disk_cache_gb,
                  readahead=a.readahead)


def cmd_render(a):
    from .surface import Surface

    os.makedirs(a.out, exist_ok=True)
    _render(a, Surface.load(a.segment), _volume(a, allow_16bit=True), a.out)


def cmd_unroll(a):
    from .metrics import surface_report
    from .surface import Surface

    from .metrics import sheet_continuity

    seg = Surface.load(a.segment)
    vol = _volume(a)
    os.makedirs(a.out, exist_ok=True)
    rep = surface_report(seg, vol.voxel_size)
    cont = sheet_continuity(seg, vol)
    rep["sheet_gap_fraction"] = cont["gap_fraction"]  # rips/holes/off-sheet: text there is not trustworthy
    import tifffile

    tifffile.imwrite(os.path.join(a.out, "sheet_gaps.tif"), cont["gap_map"].astype(np.uint8) * 255, compression="zlib")
    _dump(rep, os.path.join(a.out, "surface_report.json"))
    flat = _flatten(a, seg, os.path.join(a.out, "flat"))
    _render(a, flat, vol, a.out)


def cmd_obj2tifxyz(a):
    from .resample import obj_to_surface

    s = obj_to_surface(a.mesh, step=a.step, uv_size=tuple(a.uv_size) if a.uv_size else None)
    s.save(a.out)
    _log(f"wrote {a.out}: grid {s.shape}, valid {s.valid.mean():.3f}")


def cmd_phantom(a):
    from .phantom import SpiralScroll
    from .volume import write_ome_zarr

    P = SpiralScroll(size=tuple(a.size), seed=a.seed)
    write_ome_zarr(os.path.join(a.out, "volume.zarr"), P.volume(), levels=2, voxel_size=7.91)
    P.surface(step=a.grid_step).save(os.path.join(a.out, "segment"))
    import tifffile

    tifffile.imwrite(os.path.join(a.out, "ink_truth.tif"), (P.ink * 255).astype(np.uint8))
    _log(f"phantom written to {a.out}")


def cmd_validate(a):
    from .validate import run

    res = run(log=_log)
    _dump(res)
    sys.exit(0 if res["passed"] else 1)


def cmd_bench(a):
    from .bench import run

    summ = run(n=a.n, start=a.start, workers=a.workers, out=a.out, log=_log, two_layer=a.two_layer,
               pipeline=a.pipeline)
    _dump(summ)


def cmd_trace(a):
    from .metrics import surface_report
    from .trace import trace

    vol = _volume(a)
    s = trace(vol, a.seed, step=a.grid_step, generations=a.generations, snap_radius=a.snap_radius,
              umbilicus=a.umbilicus, log=_log, seed_check=a.seed_check)
    if a.refine:
        from .refine import refine

        s, rep = refine(s, vol, log=_log)
        s.meta["refine"] = rep
    if a.regularize:
        s = _regularize(s, vol, a.fill_holes)
    s.save(a.out)
    _dump(surface_report(s, vol.voxel_size), os.path.join(a.out, "surface_report.json"))


def _regularize(s, vol, fill):
    from .regularize import fill_holes, regularize, remove_folds

    s, rep = remove_folds(s)
    if rep["removed_vertices"]:
        _log(f"remove_folds: cut {rep['removed_vertices']} vertices in {rep['flap_components']} folded flaps")
    s.meta["remove_folds"] = rep
    s, rep = regularize(s, vol)
    _log(f"regularize: median |shift| {rep['shift_p50']:.2f} vx, p95 {rep['shift_p95']:.2f} vx")
    s.meta["regularize"] = {k: v for k, v in rep.items() if k != "params"}
    if fill:
        s, rep = fill_holes(s, vol, max_size=fill)
        _log(f"fill_holes: {rep['filled_vertices']} vertices in {rep['filled_holes']} of {rep['holes']} interior holes")
        s.meta["fill_holes"] = rep
    return s


def cmd_regularize(a):
    from .surface import Surface

    s = _regularize(Surface.load(a.segment), _volume(a), a.fill_holes)
    s.save(a.out)
    _dump({k: s.meta.get(k) for k in ("remove_folds", "regularize", "fill_holes")})


def cmd_refine(a):
    from .refine import refine
    from .surface import Surface

    s, rep = refine(Surface.load(a.segment), _volume(a), iterations=a.iterations,
                    max_shift=a.max_shift, log=_log)
    s.save(a.out, refine=rep)
    _dump(rep)


def cmd_budget(a):
    from .store import WorkBudget

    B = WorkBudget(a.workdir, int(a.max_gb * 2**30))
    for k in a.keep or []:
        B.protect(k)
    before = B.usage()
    if a.dry_run:
        total, would = before, []
        for _, name, size in B.entries():
            if total <= B.max_bytes:
                break
            if name not in B._protected:
                would.append(name)
                total -= size
        _dump({"usage_gb": before / 2**30, "max_gb": a.max_gb, "would_remove": would})
        return
    removed = B.enforce()
    _dump({"usage_gb_before": before / 2**30, "usage_gb_after": B.usage() / 2**30, "max_gb": a.max_gb,
           "removed": removed})


def cmd_crackle(a):
    from dataclasses import replace

    import tifffile
    from PIL import Image

    from .crackle import CrackleModel, CrackleParams, binary_scores, crackle_score, default_model, overlay, \
        read_layers

    Image.MAX_IMAGE_PIXELS = None
    t = time.time()
    stack, nums = read_layers(a.layers)
    H, W = stack.shape[1:]
    model = CrackleModel.from_json(open(a.model).read()) if a.model else default_model()
    params = CrackleParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in model.params.items()})
    surf = a.surface if a.surface is not None else (32 if 32 in nums else nums[len(nums) // 2])
    params = replace(params, surface=nums.index(surf), step=a.step,
                     ink_side=a.ink_side if a.ink_side else params.ink_side,
                     edge_margin=params.edge_margin if a.edge_margin is None else a.edge_margin)

    def png(path, thr):  # ink labels may be zero-padded beyond the layer grid: crop to (H, W)
        return np.array(Image.open(path))[:H, :W] > thr
    mask = png(a.mask, 0) if a.mask else None
    if mask is None and stack.dtype == np.uint8:
        mask = stack[nums.index(surf)] > 0  # unroller renders store invalid pixels as 0
    sc, feats = crackle_score(stack, mask, model, params)
    os.makedirs(a.out, exist_ok=True)
    tifffile.imwrite(os.path.join(a.out, "crackle_score.tif"), sc.astype(np.float32), compression="zlib")
    labels = png(a.labels, 127) if a.labels else None
    k = nums.index(surf)
    grey = np.mean([np.asarray(stack[i], np.float32) for i in range(max(k - 1, 0), min(k + 2, len(stack)))], 0)
    overlay(grey, sc, labels, feats["valid"], os.path.join(a.out, "crackle_overlay.png"), max_side=a.max_side)
    ok = feats["scored"]
    thr = model.percentile_threshold(sc[ok])
    rep = {"layers": [nums[0], nums[-1]], "surface_layer": surf, "shape": [len(stack), H, W],
           "valid_fraction": float(feats["valid"].mean()), "edge_margin_px": params.edge_margin,
           "scored_fraction": float(ok.mean()), "seconds": round(time.time() - t, 1),
           "model": {"names": list(model.names), "weights": list(model.weights), "bias": model.bias,
                     "trained_on": model.info.get("train_segment")},
           "operating_point": {"kind": "per-segment percentile", "top_fraction": model.top_fraction,
                               "threshold": thr, "fraction_flagged": float(np.mean(sc[ok] >= thr))},
           "absolute_threshold": {"threshold": model.threshold,
                                  "fraction_above_threshold": float(np.mean(sc[ok] >= model.threshold)),
                                  "note": "uncalibrated across segments: use operating_point"}}
    if labels is not None:
        rep["labels"] = binary_scores(sc[ok], labels[ok], thr)
    _dump(rep, os.path.join(a.out, "crackle_report.json"))


def main(argv=None):
    from .surface import NORMAL_SIGMA_VX

    p = argparse.ArgumentParser(prog="unroller", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def flat_args(q):
        q.add_argument("--iterations", type=int, default=200, help="max ARAP iterations (0 = initial map only)")
        q.add_argument("--init", choices=["grid", "lscm"], default="grid", help="initial map before ARAP")
        q.add_argument("--downsample", type=int, default=1, help="flatten every k-th vertex, upsample UVs")
        q.add_argument("--out-scale", type=float, default=None, help="output grid cells per voxel (default: input)")

    def render_args(q):
        q.add_argument("-n", "--layers", type=int, default=65)
        q.add_argument("--step", type=float, default=1.0, help="layer spacing along normal (voxels)")
        q.add_argument("--scale", type=float, default=1.0, help="output pixels per voxel")
        q.add_argument("--level", type=int, default=0, help="pyramid level to sample")
        q.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
        q.add_argument("--workers", type=int, default=4, help="parallel render tiles")
        q.add_argument("--normal-sigma", type=float, default=NORMAL_SIGMA_VX,
                       help="layer normals from the grid smoothed over this many voxels (0 = raw normals)")
        q.add_argument("--no-tif", dest="tif", action="store_false", help="only write layers.zarr")
        vol_args(q)

    def vol_args(q):
        q.add_argument("--cache-gb", type=float, default=4.0, help="RAM for compressed chunks")
        q.add_argument("--disk-cache", default=None, help="directory for a persistent compressed chunk cache")
        q.add_argument("--disk-cache-gb", type=float, default=20.0, help="cap for --disk-cache (LRU eviction)")
        q.add_argument("--readahead", action="store_true", help="background-fetch neighbouring chunks (remote)")
        q.add_argument("--volume-16bit", default=None, metavar="URL",
                       help="render only: uncompressed uint16 TIFF slice stack, or a .../volume_grids/<uuid> 500^3 cell "
                            "grid (dir or URL), streamed by range requests, used instead of the positional volume "
                            "(pass '-' there); writes uint16 layers")
        q.add_argument("--brick", type=int, default=64, help="brick edge for --volume-16bit (voxels)")

    q = sub.add_parser("info"); q.add_argument("segment"); q.add_argument("--voxel-size", type=float)
    q.set_defaults(fn=cmd_info)
    q = sub.add_parser("flatten"); q.add_argument("segment"); q.add_argument("out"); flat_args(q)
    q.set_defaults(fn=cmd_flatten)
    q = sub.add_parser("render"); q.add_argument("segment"); q.add_argument("volume"); q.add_argument("out")
    render_args(q); q.set_defaults(fn=cmd_render)
    q = sub.add_parser("unroll"); q.add_argument("segment"); q.add_argument("volume"); q.add_argument("out")
    flat_args(q); render_args(q); q.set_defaults(fn=cmd_unroll)
    q = sub.add_parser("obj2tifxyz"); q.add_argument("mesh"); q.add_argument("out")
    q.add_argument("--step", type=float, default=1.0, help="output grid spacing in voxels")
    q.add_argument("--uv-size", type=float, nargs=2, metavar=("W", "H"), help="pixels spanned by UV [0,1]")
    q.set_defaults(fn=cmd_obj2tifxyz)
    q = sub.add_parser("phantom"); q.add_argument("out")
    q.add_argument("--size", type=int, nargs=3, default=[96, 224, 224], metavar=("Z", "Y", "X"))
    q.add_argument("--grid-step", type=float, default=2.0); q.add_argument("--seed", type=int, default=0)
    q.set_defaults(fn=cmd_phantom)
    q = sub.add_parser("validate"); q.set_defaults(fn=cmd_validate)
    q = sub.add_parser("bench", help="score the full pipeline on many random phantom scrolls")
    q.add_argument("-n", type=int, default=100); q.add_argument("--start", type=int, default=0)
    q.add_argument("--workers", type=int, default=4); q.add_argument("--out", default="bench.json")
    q.add_argument("--single-layer", dest="two_layer", action="store_false", help="single-layer phantoms")
    q.add_argument("--pipeline", choices=("regularize", "refine"), default="regularize",
                   help="after the trace: cut folds + regularize + fill pinholes (default), or the previous refine")
    q.set_defaults(fn=cmd_bench)
    q = sub.add_parser("trace", help="grow a new segment from a seed point (unsegmented scrolls)")
    q.add_argument("volume"); q.add_argument("out")
    q.add_argument("--seed", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    q.add_argument("--umbilicus", type=float, nargs=3, metavar=("X", "Y", "Z"), help="a point on the scroll axis")
    q.add_argument("--grid-step", type=float, default=4.0); q.add_argument("--generations", type=int, default=60)
    q.add_argument("--snap-radius", type=float, default=None, help="default: 0.35 x measured sheet spacing")
    q.add_argument("--seed-check", action="store_true", help="refuse a seed unlike its surroundings (default: warn)")
    q.add_argument("--refine", action="store_true")
    q.add_argument("--regularize", action="store_true", help="cut folds, remove along-normal jitter, fill pinholes")
    q.add_argument("--fill-holes", type=int, default=4, help="largest interior hole filled (vertices; 0 = none)")
    vol_args(q)
    q.set_defaults(fn=cmd_trace)
    q = sub.add_parser("regularize", help="cut folds, remove along-normal jitter of a traced segment, fill small holes")
    q.add_argument("segment"); q.add_argument("volume"); q.add_argument("out")
    q.add_argument("--fill-holes", type=int, default=4, help="largest interior hole filled (vertices; 0 = none)")
    vol_args(q)
    q.set_defaults(fn=cmd_regularize)
    q = sub.add_parser("refine", help="move a segment onto the local sheet centre")
    q.add_argument("segment"); q.add_argument("volume"); q.add_argument("out")
    q.add_argument("--iterations", type=int, default=3); q.add_argument("--max-shift", type=float, default=3.0)
    vol_args(q)
    q.set_defaults(fn=cmd_refine)

    q = sub.add_parser("budget", help="keep a work directory under a size budget (LRU deletion of entries)")
    q.add_argument("workdir"); q.add_argument("--max-gb", type=float, required=True)
    q.add_argument("--keep", nargs="*", help="entries never to delete"); q.add_argument("--dry-run", action="store_true")
    q.set_defaults(fn=cmd_budget)
    q = sub.add_parser("crackle", help="ink-side fine-valley texture score map and overlay of a layer stack "
                       "(a weak ink hint, see README)")
    q.add_argument("layers", help="directory of NN.tif layers (VC or unroller render) or a layers.zarr")
    q.add_argument("out")
    q.add_argument("--surface", type=int, default=None, help="layer number of the surface (default 32 / middle)")
    q.add_argument("--step", type=float, default=1.0, help="voxels between layers")
    q.add_argument("--ink-side", type=int, choices=[-1, 1], default=None, help="override the model's ink side")
    q.add_argument("--mask", help="segment mask PNG (default: all pixels, or non-zero surface layer for uint8)")
    q.add_argument("--labels", help="ink labels PNG to draw as contours and score against (>127 = ink)")
    q.add_argument("--model", help="CrackleModel JSON (default: the bundled model)")
    q.add_argument("--edge-margin", type=float, default=None,
                   help="suppress scores within this many px of the mask edge (default: the model's, 64)")
    q.add_argument("--max-side", type=int, default=4096, help="overlay PNG size limit (integer downsampling)")
    q.set_defaults(fn=cmd_crackle)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
