"""Lean storage: lossless compressed arrays and a capped work directory.

Everything that can be large (CT cubes, rendered layer stacks, label volumes, intermediate fields) is
kept compressed -- never lossy: the bytes that come back are exactly the bytes that went in.

  save(path, a)   ``*.zarr`` -> chunked zarr v2 array, blosc-zstd (readable by Volume, VC3D, zarr-python);
                  ``*.npz``  -> numpy savez_compressed (zlib), single file.
  load(path)      the array back (a region of a .zarr can be read without decoding the rest).
  WorkBudget      keeps a work directory under a byte budget by deleting least-recently-used entries.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time

import numpy as np
import numcodecs

DEFAULT_CLEVEL = 3


def _codec(dtype: np.dtype, clevel: int = DEFAULT_CLEVEL):
    """blosc-zstd.  Shuffle measured on real data: Scroll 1 uint8 CT (256^3 cube) compresses 2.79x with
    no shuffle vs 2.17x bit-shuffled; float32 grids (tifxyz points / normals) 1.32x / 2.51x byte-shuffled
    vs 1.13x / 2.03x unshuffled and 1.47x / 1.28x bit-shuffled -> byte shuffle for multi-byte types."""
    shuffle = numcodecs.Blosc.NOSHUFFLE if np.dtype(dtype).itemsize == 1 else numcodecs.Blosc.SHUFFLE
    return numcodecs.Blosc(cname="zstd", clevel=clevel, shuffle=shuffle)


def choose_codec(sample: np.ndarray, clevel: int = DEFAULT_CLEVEL):
    """blosc-zstd with the shuffle that compresses ``sample`` (a representative chunk) best.  For uint8 it
    is data-dependent: dense Scroll 1 CT/stacks compress 1-4% better unshuffled, sparse (partly empty)
    stacks up to ~20% better bit-shuffled.  Multi-byte types: byte shuffle (see _codec).  Ties -> no shuffle."""
    if np.dtype(sample.dtype).itemsize != 1:
        return _codec(sample.dtype, clevel)
    buf = np.ascontiguousarray(sample)
    best = None
    for sh in (numcodecs.Blosc.NOSHUFFLE, numcodecs.Blosc.BITSHUFFLE):
        c = numcodecs.Blosc(cname="zstd", clevel=clevel, shuffle=sh)
        n = len(c.encode(buf))
        if best is None or n < best[0]:
            best = (n, c)
    return best[1]


def _du(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    tot = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            try:
                tot += os.path.getsize(os.path.join(dp, fn))
            except FileNotFoundError:
                pass
    return tot


def _report(path, a, fmt, sec):
    stored = _du(path)
    return {"path": path, "format": fmt, "shape": list(a.shape), "dtype": a.dtype.str, "raw_bytes": int(a.nbytes),
            "stored_bytes": int(stored), "ratio": a.nbytes / max(stored, 1), "seconds": sec}


def save(path: str, a: np.ndarray, chunks=None, clevel: int = DEFAULT_CLEVEL, workers: int = 4) -> dict:
    """Write ``a`` losslessly compressed; the format follows the extension (.zarr or .npz).
    Returns a size report {raw_bytes, stored_bytes, ratio, ...}."""
    a = np.asarray(a)
    t = time.perf_counter()
    if path.endswith(".npz"):
        tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, a=a)
        os.replace(tmp, path)
        return _report(path, a, "npz", time.perf_counter() - t)
    if not path.endswith(".zarr"):
        raise ValueError("store.save: path must end in .zarr or .npz")
    if chunks is None:
        chunks = tuple(min(s, c) for s, c in zip(a.shape, (128,) * 3 if a.ndim >= 3 else (1024,) * a.ndim))
        chunks = chunks + a.shape[len(chunks):]
    chunks = tuple(int(max(1, min(c, s))) for c, s in zip(chunks, a.shape))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    grid = [range(-(-s // c)) for s, c in zip(a.shape, chunks)]
    keys = list(np.ndindex(*[len(g) for g in grid])) if a.size else []

    def block(key):
        sl = tuple(slice(k * c, (k + 1) * c) for k, c in zip(key, chunks))
        src = a[sl]
        blk = np.zeros(chunks, a.dtype)
        blk[tuple(slice(0, s) for s in src.shape)] = src
        return blk
    # the codec is chosen on the first non-empty chunk in key order (deterministic)
    first = next((k for k in keys if np.any(a[tuple(slice(i * c, (i + 1) * c) for i, c in zip(k, chunks))] != 0)),
                 None)
    codec = choose_codec(block(first), clevel) if first is not None else _codec(a.dtype, clevel)
    with open(os.path.join(path, ".zarray"), "w") as f:
        json.dump({"chunks": list(chunks), "compressor": codec.get_config(), "dimension_separator": "/",
                   "dtype": a.dtype.str, "fill_value": 0, "filters": None, "order": "C",
                   "shape": list(a.shape), "zarr_format": 2}, f)
    made, lock = set(), threading.Lock()

    def put(key):
        sl = tuple(slice(k * c, (k + 1) * c) for k, c in zip(key, chunks))
        src = a[sl]
        if not np.any(src != 0):  # absent chunk == fill value 0 (NaN != 0, so NaN chunks are stored)
            return
        blk = np.zeros(chunks, a.dtype)
        blk[tuple(slice(0, s) for s in src.shape)] = src
        d = os.path.join(path, *map(str, key[:-1]))
        with lock:
            if d not in made:
                os.makedirs(d, exist_ok=True)
                made.add(d)
        with open(os.path.join(d, str(key[-1])), "wb") as f:
            f.write(codec.encode(blk))

    if workers > 1 and len(keys) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(put, keys))
    else:
        for k in keys:
            put(k)
    return _report(path, a, "zarr", time.perf_counter() - t)


def load(path: str, region=None) -> np.ndarray:
    """Read an array written by save() (or any C-order zarr v2 array). ``region``: tuple of slices
    (step 1) for .zarr -- only the chunks it overlaps are decoded."""
    if path.endswith(".npz"):
        with np.load(path) as z:
            a = z["a"] if "a" in z.files else z[z.files[0]]
        return a if region is None else a[region]
    with open(os.path.join(path, ".zarray")) as f:
        m = json.load(f)
    shape, chunks, dt = tuple(m["shape"]), tuple(m["chunks"]), np.dtype(m["dtype"])
    sep = m.get("dimension_separator", ".")
    codec = numcodecs.get_codec(m["compressor"]) if m.get("compressor") else None
    filters = [numcodecs.get_codec(x) for x in (m.get("filters") or [])]
    region = tuple(region or ()) + (slice(None),) * (len(shape) - len(tuple(region or ())))
    lo = [r.indices(s)[0] for r, s in zip(region, shape)]
    hi = [max(r.indices(s)[1], l) for r, s, l in zip(region, shape, lo)]
    out = np.full([h - l for l, h in zip(lo, hi)], m.get("fill_value") or 0, dt)
    if out.size == 0:
        return out
    k0 = [l // c for l, c in zip(lo, chunks)]
    k1 = [(h - 1) // c + 1 for h, c in zip(hi, chunks)]
    for key in np.ndindex(*[b - a for a, b in zip(k0, k1)]):
        key = tuple(int(k + a) for k, a in zip(key, k0))
        p = os.path.join(path, sep.join(map(str, key))) if sep == "/" else os.path.join(path, sep.join(map(str, key)))
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            buf = f.read()
        buf = codec.decode(buf) if codec else buf
        for flt in reversed(filters):
            buf = flt.decode(buf)
        blk = np.frombuffer(buf, dt).reshape(chunks)
        a = [max(l, k * c) for l, k, c in zip(lo, key, chunks)]
        b = [min(h, (k + 1) * c) for h, k, c in zip(hi, key, chunks)]
        out[tuple(slice(x - l, y - l) for x, y, l in zip(a, b, lo))] = \
            blk[tuple(slice(x - k * c, y - k * c) for x, y, k, c in zip(a, b, key, chunks))]
    return out


def size_report(path: str, raw_bytes: int | None = None) -> dict:
    """Bytes on disk of a file or directory (and the compression ratio if the raw size is known)."""
    stored = _du(path)
    rep = {"path": path, "stored_bytes": stored}
    if raw_bytes is not None:
        rep["raw_bytes"] = int(raw_bytes)
        rep["ratio"] = raw_bytes / max(stored, 1)
    return rep


class WorkBudget:
    """Keep the top-level entries (files or directories) of ``root`` under ``max_bytes`` in total by
    deleting the least recently used ones.  Recency = the later of the newest file mtime inside an entry
    and the last ``touch``; ``protect``ed entries (e.g. the run in progress) are never deleted.

    Typical use: one budget for a work directory holding chunk caches, cubes and rendered stacks::

        B = WorkBudget(work, max_bytes=50 * 2**30); B.protect(current_out); ...; B.enforce()
    """

    INDEX = ".budget.json"

    def __init__(self, root: str, max_bytes: int):
        self.root = os.path.abspath(root)
        self.max_bytes = int(max_bytes)
        self._protected: set[str] = set()
        os.makedirs(self.root, exist_ok=True)

    def _index(self) -> dict:
        try:
            with open(os.path.join(self.root, self.INDEX)) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _entry(self, path: str) -> str:
        rel = os.path.relpath(os.path.abspath(path), self.root)
        if rel.startswith(".."):
            raise ValueError(f"{path} is not inside {self.root}")
        return rel.split(os.sep)[0]

    def touch(self, path: str):
        """Mark the entry containing ``path`` as just used."""
        idx = self._index()
        idx[self._entry(path)] = time.time()
        tmp = os.path.join(self.root, f"{self.INDEX}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(idx, f)
        os.replace(tmp, os.path.join(self.root, self.INDEX))

    def protect(self, path: str):
        self._protected.add(self._entry(path))

    def entries(self) -> list[tuple[float, str, int]]:
        """(last use, name, bytes) for every top-level entry, oldest first."""
        idx = self._index()
        out = []
        for name in os.listdir(self.root):
            if name.startswith(self.INDEX):
                continue
            p = os.path.join(self.root, name)
            newest, size = 0.0, 0
            if os.path.isfile(p):
                st = os.stat(p)
                newest, size = st.st_mtime, st.st_size
            else:
                for dp, _, fns in os.walk(p):
                    for fn in fns:
                        try:
                            st = os.stat(os.path.join(dp, fn))
                        except FileNotFoundError:
                            continue
                        newest, size = max(newest, st.st_mtime), size + st.st_size
            out.append((max(newest, idx.get(name, 0.0)), name, size))
        return sorted(out)

    def usage(self) -> int:
        return sum(e[2] for e in self.entries())

    def enforce(self) -> list[str]:
        """Delete least-recently-used unprotected entries until usage <= max_bytes; returns them."""
        ents = self.entries()
        total = sum(e[2] for e in ents)
        removed = []
        for _, name, size in ents:
            if total <= self.max_bytes:
                break
            if name in self._protected:
                continue
            p = os.path.join(self.root, name)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
            total -= size
            removed.append(p)
        if removed:
            idx = self._index()
            for p in removed:
                idx.pop(os.path.basename(p), None)
            with open(os.path.join(self.root, self.INDEX), "w") as f:
                json.dump(idx, f)
        return removed
