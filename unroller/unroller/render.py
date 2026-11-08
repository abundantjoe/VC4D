"""Render a surface into a layered image stack by sampling along its normals.

Layer offsets match vc_render_tifxyz: off_i = (i - (n-1)/2) * step, so the middle layer is
the surface itself and the stack is symmetric about it.

Layers are sampled along normals of the grid smoothed over ``normal_sigma`` voxels (Surface.normals):
raw finite-difference normals of a traced grid follow its vertex jitter (on traced phantoms: median 8.5 deg,
p90 18 deg from the true sheet normal), so layers at an offset drift sideways and, where a triangle is flipped,
sample the other face.  normal_sigma=0 = raw.
"""
from __future__ import annotations

import os

import numpy as np
import tifffile

from .surface import NORMAL_SIGMA_VX, Surface
from .volume import Volume


def layer_offsets(n: int, step: float = 1.0) -> np.ndarray:
    return (np.arange(n) - 0.5 * (n - 1)) * step


def render(surface: Surface, volume: Volume, n_layers: int = 1, step: float = 1.0, scale: float = 1.0,
           level: int = 0, crop=None, tile_rows: int = 256, progress=None, normals=None,
           normal_sigma: float = NORMAL_SIGMA_VX) -> np.ndarray:
    """Return a float32 stack (n_layers, H, W); NaN where the surface is invalid.

    ``scale`` = output pixels per level-0 voxel along the surface.  ``step`` is in level-0 voxels.
    ``normals`` = surface.normals(normal_sigma) if already computed (render_stream computes them once, not per tile).
    """
    if normals is None and n_layers > 1:
        normals = surface.normals(normal_sigma)
    pts, nrm = surface.resample(scale, crop, normals=normals)
    H, W = pts.shape[:2]
    if not np.isfinite(pts[..., 0]).any():
        return np.full((n_layers, H, W), np.nan, np.float32)
    offs = layer_offsets(n_layers, step)
    out = np.full((n_layers, H, W), np.nan, np.float32)
    for r0 in range(0, H, tile_rows):
        p = pts[r0:r0 + tile_rows]
        q = nrm[r0:r0 + tile_rows]
        if hasattr(volume, "sample_layers"):  # fused: identical to the loop below, no temporaries
            out[:, r0:r0 + tile_rows] = volume.sample_layers(p, q, offs, level=level)
        else:
            for i, o in enumerate(offs):
                out[i, r0:r0 + tile_rows] = volume.sample(p + o * q, level=level)
        if progress:
            progress(min(r0 + tile_rows, H), H)
    return out


def render_positions(surface: Surface, n_layers: int = 1, step: float = 1.0, scale: float = 1.0, crop=None,
                     normals=None, normal_sigma: float = NORMAL_SIGMA_VX):
    """The sample positions ``render`` uses (same arguments and normal defaults): (pts (H, W, 3), nrm (H, W, 3),
    offsets (n_layers,)); layer l of pixel (r, c) samples pts[r, c] + offsets[l] * nrm[r, c] (xyz, level-0 voxels;
    NaN where invalid)."""
    if normals is None and n_layers > 1:
        normals = surface.normals(normal_sigma)
    pts, nrm = surface.resample(scale, crop, normals=normals)
    return pts, nrm, layer_offsets(n_layers, step)


def save_layers(stack: np.ndarray, outdir: str, dtype=np.uint8, compression: str | None = "zlib"):
    """Write layers/00.tif, 01.tif, ... (the layout ink-detection models consume). NaN -> 0.
    Lossless deflate by default (standard TIFF, read by tifffile / PIL / OpenCV); None = uncompressed."""
    os.makedirs(outdir, exist_ok=True)
    width = max(2, len(str(len(stack) - 1)))
    for i, layer in enumerate(stack):
        img = np.nan_to_num(layer, nan=0.0)
        if np.issubdtype(dtype, np.integer):
            img = np.clip(np.rint(img), 0, np.iinfo(dtype).max)
        tifffile.imwrite(os.path.join(outdir, f"{i:0{width}d}.tif"), img.astype(dtype), compression=compression)


def _tile_chunks(surface, nrm, crop, scale, offs, lvl, f):
    """Chunk keys a render tile will probably read (for background prefetch only, so approximate):
    the grid vertices under the tile, displaced to 5 offsets spanning the layer range."""
    sx, sy = surface.scale
    x0, y0, w, h = crop
    H, W = surface.shape
    c0, c1 = max(int(np.floor(x0 / scale * sx)) - 1, 0), min(int(np.ceil((x0 + w) / scale * sx)) + 2, W)
    r0, r1 = max(int(np.floor(y0 / scale * sy)) - 1, 0), min(int(np.ceil((y0 + h) / scale * sy)) + 2, H)
    sub = (slice(r0, r1, 4), slice(c0, c1, 4))  # every 4th vertex: chunks are >= 64 voxels wide
    p = surface.points[sub].reshape(-1, 3).astype(np.float64)
    n = nrm[sub].reshape(-1, 3).astype(np.float64)
    ok = np.isfinite(p).all(1) & np.isfinite(n).all(1)
    p, n = p[ok], n[ok]
    if not len(p):
        return []
    lo, hi = float(offs.min()), float(offs.max())
    q = np.concatenate([p + o * n for o in np.linspace(lo, hi, 5)]) * f
    c = np.array(lvl.chunks[::-1], float)  # xyz chunk sizes
    k = np.floor(np.concatenate([q - 2.0, q + 2.0]) / c).astype(np.int64)
    grid = np.array([-(-s // ch) for s, ch in zip(lvl.shape[::-1], lvl.chunks[::-1])])
    k = k[np.all((k >= 0) & (k < grid), 1)]
    flat = np.unique((k[:, 2] * grid[1] + k[:, 1]) * grid[0] + k[:, 0])  # 1-D key: a fast sort
    z, rem = np.divmod(flat, grid[1] * grid[0])
    y, x = np.divmod(rem, grid[0])
    return list(zip(z.tolist(), y.tolist(), x.tolist()))


def render_stream(surface: Surface, volume: Volume, out: str, n_layers: int = 65, step: float = 1.0,
                  scale: float = 1.0, level: int = 0, crop=None, tile: int = 256, workers: int = 4,
                  clevel: int = 3, progress=None, normal_sigma: float = NORMAL_SIGMA_VX) -> dict:
    """Render straight to a compressed zarr array ``out`` (layers, rows, cols), tile by tile: uint8, or
    uint16 when the sampled level is 16-bit (e.g. a TiffVolume), so no precision is lost.

    Memory is bounded by one tile (n_layers x tile x tile floats per worker) regardless of segment
    size; tiles with no valid surface are never written (absent chunk = 0), and chunks are
    blosc-zstd compressed.  Returns the stack_report computed from running sums.
    """
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import numcodecs

    sx, sy = surface.scale
    H0, W0 = surface.shape
    full_w = int(np.floor((W0 - 1) / sx * scale)) + 1
    full_h = int(np.floor((H0 - 1) / sy * scale)) + 1
    x0, y0, W, H = crop if crop else (0, 0, full_w, full_h)
    # the shuffle is chosen on the first non-empty tile in tile order (store.choose_codec): unshuffled wins
    # on dense stacks, bit-shuffle on sparse ones; any blosc chunk decodes the same way either way
    from .store import choose_codec

    codec = numcodecs.Blosc(cname="zstd", clevel=clevel, shuffle=numcodecs.Blosc.NOSHUFFLE)
    lv = volume.levels[level] if hasattr(volume, "levels") else None
    odt = np.dtype(np.uint16) if lv is not None and np.dtype(lv.dtype).itemsize == 2 else np.dtype(np.uint8)
    os.makedirs(out, exist_ok=True)

    def write_meta(c):
        with open(os.path.join(out, ".zarray"), "w") as f:
            json.dump({"chunks": [n_layers, tile, tile], "compressor": c.get_config(), "dimension_separator": "/",
                       "dtype": odt.str, "fill_value": 0, "filters": None, "order": "C",
                       "shape": [n_layers, H, W], "zarr_format": 2}, f)
    write_meta(codec)
    with open(os.path.join(out, ".zattrs"), "w") as f:
        json.dump({"unroller": {"layer_offsets_vx": layer_offsets(n_layers, step).tolist(), "scale": scale,
                                "level": level, "crop": [x0, y0, W, H], "segment": surface.meta.get("uuid"),
                                "normal_sigma_vx": normal_sigma}}, f)
    sums = np.zeros(n_layers)
    cnts = np.zeros(n_layers)
    lock = threading.Lock()
    tiles = [(ty, tx) for ty in range(-(-H // tile)) for tx in range(-(-W // tile))]
    done = [0]
    nrm = surface.normals(normal_sigma)  # once for the whole segment, not per tile
    lvl = volume.levels[level] if hasattr(volume, "levels") else None
    lookahead = 16 if lvl is not None and hasattr(lvl, "prefetch_async") and lvl.store.remote else 0
    offs = layer_offsets(n_layers, step)

    def prefetch(i):  # fetch the chunks of tile i in the background (timing only, never values)
        if i < len(tiles):
            ty, tx = tiles[i]
            crop_i = (x0 + tx * tile, y0 + ty * tile, min(tile, W - tx * tile), min(tile, H - ty * tile))
            lvl.prefetch_async(_tile_chunks(surface, nrm, crop_i, scale, offs, lvl, 2.0 ** -level))
    for i in range(lookahead):
        prefetch(i)
    index = {t: i for i, t in enumerate(tiles)}
    status = [None] * len(tiles)  # per tile: None = not rendered yet, False = empty, True = has data
    chosen = [None]
    cond = threading.Condition()

    def codec_for(i, nonempty, blk):
        """The array's codec, decided deterministically by the first non-empty tile in tile order."""
        with cond:
            status[i] = nonempty
            cond.notify_all()
            if not nonempty:
                return None
            while chosen[0] is None:
                if all(s is False for s in status[:i]):  # this is the first non-empty tile
                    chosen[0] = choose_codec(blk, clevel)
                    write_meta(chosen[0])
                    cond.notify_all()
                    break
                cond.wait()  # an earlier tile is still rendering or is deciding
            return chosen[0]

    def work(t):
        ty, tx = t
        if lookahead:
            prefetch(index[t] + lookahead)
        h = min(tile, H - ty * tile)
        w = min(tile, W - tx * tile)
        st = render(surface, volume, n_layers, step, scale, level, crop=(x0 + tx * tile, y0 + ty * tile, w, h),
                    normals=nrm)
        ok = np.isfinite(st)
        i = index[t]
        tile_sums[i] = np.where(ok, st, 0).sum((1, 2))
        tile_cnts[i] = ok.sum((1, 2))
        with lock:
            done[0] += 1
            if progress:
                progress(done[0], len(tiles))
        blk = None
        if ok.any():
            blk = np.zeros((n_layers, tile, tile), odt)
            blk[:, :h, :w] = np.clip(np.rint(np.nan_to_num(st, nan=0.0)), 0, np.iinfo(odt).max)
        codec = codec_for(index[t], blk is not None, blk)
        if blk is None:
            return 0
        d = os.path.join(out, "0", str(ty))
        os.makedirs(d, exist_ok=True)
        data = codec.encode(blk)
        with open(os.path.join(d, str(tx)), "wb") as f:
            f.write(data)
        return len(data)

    tile_sums = np.zeros((len(tiles), n_layers))
    tile_cnts = np.zeros((len(tiles), n_layers))
    with ThreadPoolExecutor(workers) as ex:
        written = sum(ex.map(work, tiles))
    for i in range(len(tiles)):  # accumulate in tile order: the report no longer depends on thread timing
        sums += tile_sums[i]
        cnts += tile_cnts[i]
    with np.errstate(invalid="ignore"):
        prof = sums / cnts
    stack = prof[:, None, None]  # reuse stack_report's peak logic on the mean profile
    from .metrics import stack_report

    rep = stack_report(stack, step)
    rep["valid_fraction"] = float(cnts[n_layers // 2] / (H * W))
    rep["bytes_written"] = int(written)
    rep["uncompressed_bytes"] = int(n_layers * H * W * odt.itemsize)
    return rep


def zarr_layers_to_tifs(zpath: str, outdir: str, compression: str = "zlib", mem_bytes: int = 256 * 2**20,
                        workers: int = 4):
    """Export a render_stream zarr to layers/NN.tif.  Layers are exported in groups that fit ``mem_bytes``:
    each chunk is decoded once per group (not once per layer) and the group's tifs are compressed in
    parallel threads (zlib releases the GIL).  The files are byte-identical to a layer-by-layer export."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    import numcodecs

    with open(os.path.join(zpath, ".zarray")) as f:
        m = json.load(f)
    n, H, W = m["shape"]
    _, th, tw = m["chunks"]
    codec = numcodecs.get_codec(m["compressor"])
    dt = np.dtype(m["dtype"])
    os.makedirs(outdir, exist_ok=True)
    width = max(2, len(str(n - 1)))
    group = int(max(1, min(n, mem_bytes // max(H * W * dt.itemsize, 1))))
    keys = [(ty, tx) for ty in range(-(-H // th)) for tx in range(-(-W // tw))
            if os.path.exists(os.path.join(zpath, "0", str(ty), str(tx)))]
    with ThreadPoolExecutor(workers) as ex:
        for g0 in range(0, n, group):
            g1 = min(n, g0 + group)
            buf = np.zeros((g1 - g0, H, W), dt)

            def fill(k):
                ty, tx = k
                with open(os.path.join(zpath, "0", str(ty), str(tx)), "rb") as f:
                    blk = np.frombuffer(codec.decode(f.read()), dt).reshape(n, th, tw)
                h, w = min(th, H - ty * th), min(tw, W - tx * tw)
                buf[:, ty * th:ty * th + h, tx * tw:tx * tw + w] = blk[g0:g1, :h, :w]

            list(ex.map(fill, keys))
            list(ex.map(lambda i: tifffile.imwrite(os.path.join(outdir, f"{i:0{width}d}.tif"), buf[i - g0],
                                                   compression=compression), range(g0, g1)))
