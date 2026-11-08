"""Streamed access to an uncompressed 16-bit TIFF slice stack (e.g. Scroll 1 ``volumes/20230205180739``).

The standardised OME-Zarr keeps 5 bits per voxel (values are multiples of 8, air clipped to 0); the
original reconstruction is a stack of uncompressed uint16 TIFFs, one per z slice.  This backend reads
only the slices and rows it needs with HTTP range requests and serves them through the same chunk-pool
machinery as ``Volume``:

  * every slice's TIFF header is parsed once (strip offsets cached in memory and in the disk cache); an
    unseen slice's header is requested together with its first rows, assuming slice 0's layout, and the
    rows are re-read only if the parsed header differs;
  * per slice, all the pieces a batch of bricks needs -- the needed columns of each needed row, or full
    rows for a brick row that needs >= 30% of the width -- go into as few multipart range requests as
    possible (<= 256 ranges each).  dl.ash2txt.org (nginx) serves ~40-130 requests/s whatever their size
    (a 1 MB range: 0.22 s; 4 MB: 0.34 s), so the request count sets the wall time; bricks either side of
    the needed ones are built too, so neighbouring render tiles rarely re-request a row;
  * voxels are assembled into ``brick``^3 bricks (the "chunks" of a virtual zarr level), compressed
    losslessly (blosc-zstd, byte shuffle) into the RAM / disk caches, and decoded into the numba
    sampling pool on demand -- so ``sample``, ``sample_layers``, ``gather`` and ``read`` behave exactly
    as for ``Volume``, with uint16 values.  Scroll 1's 16-bit CT is nearly incompressible losslessly
    (15.4 bits/voxel zeroth-order entropy; blosc ratio 1.01-1.07), so the caches hold ~2 bytes/voxel;
  * coarser levels are 2x2x2 mean-pooled (rounded) from the level below, brick by brick.

Coordinates and conventions are those of ``unroller.volume`` (public xyz, arrays ZYX).
"""
from __future__ import annotations

import json
import os
import struct
import threading
from collections import OrderedDict

import numpy as np
import numcodecs

from .volume import IO_WORKERS, Level, Volume, _executor, _is_url, _Store

_TYPES = {1: ("B", 1), 2: ("B", 1), 3: ("H", 2), 4: ("I", 4), 16: ("Q", 8)}  # TIFF field type -> struct code, size


def read_ifd(get, bo: str, off: int):
    """Tags {tag: [values]} of the IFD at ``off`` and the offset of the next IFD; ``get(offset, n)`` -> bytes."""
    n = struct.unpack(bo + "H", get(off, 2))[0]
    ent = get(off + 2, 12 * n + 4)
    tags = {}
    for i in range(n):
        tag, typ, cnt = struct.unpack(bo + "HHI", ent[12 * i:12 * i + 8])
        code, sz = _TYPES.get(typ, ("B", 1))
        raw = ent[12 * i + 8:12 * i + 12]
        if cnt * sz > 4:
            raw = get(struct.unpack(bo + "I", raw)[0], cnt * sz)
        tags[tag] = list(struct.unpack(bo + code * cnt, raw[:cnt * sz]))
    return tags, struct.unpack(bo + "I", ent[12 * n:12 * n + 4])[0]


def _layout(tags, bo):
    if tags.get(259, [1])[0] != 1:
        raise ValueError("compressed TIFF pages are not supported (need random row access)")
    if tags.get(277, [1])[0] != 1 or tags.get(284, [1])[0] != 1:
        raise ValueError("only single-channel, contiguous TIFF pages are supported")
    if 322 in tags:
        raise ValueError("tiled TIFF pages are not supported")
    bits = tags[258][0]
    fmt = tags.get(339, [1])[0]
    if fmt != 1 or bits not in (8, 16):
        raise ValueError(f"unsupported sample format {fmt} / {bits} bits")
    W, H = tags[256][0], tags[257][0]
    return {"width": W, "height": H, "bits": bits, "dtype": (bo + ("u1" if bits == 8 else "u2")),
            "rows_per_strip": tags.get(278, [H])[0], "strip_offsets": tags[273]}


def parse_tiff_header(read, size_hint: int = 16384) -> dict:
    """Layout of the first image of a classic (non-BigTIFF) uncompressed TIFF.  ``read(offset, n)`` -> bytes.

    Returns width, height, bits, dtype, rows_per_strip and the strip offsets (one per strip)."""
    head = read(0, size_hint)
    bo = {b"II": "<", b"MM": ">"}.get(head[:2])
    if bo is None or struct.unpack(bo + "H", head[2:4])[0] != 42:
        raise ValueError("not a classic TIFF (BigTIFF is not supported)")

    def get(o, n):
        return head[o:o + n] if o + n <= len(head) else read(o, n)

    tags, _ = read_ifd(get, bo, struct.unpack(bo + "I", head[4:8])[0])
    return _layout(tags, bo)


def _parse_multipart(body: bytes, ctype: str) -> list[tuple[int, bytes]]:
    """(start offset, bytes) parts of a multipart/byteranges response."""
    b = ctype.split("boundary=")[1].strip().strip('"').encode()
    out = []
    for part in body.split(b"--" + b)[1:]:
        if part.startswith(b"--"):
            break
        hdr, _, data = part.partition(b"\r\n\r\n")
        cr = [l for l in hdr.split(b"\r\n") if l.lower().startswith(b"content-range:")][0]
        a, e = cr.split(b" ")[-1].split(b"/")[0].split(b"-")
        out.append((int(a), data[:int(e) - int(a) + 1]))
    return out


class TiffStackStore(_Store):
    """Key -> blosc-compressed brick bytes, built from a TIFF slice stack on first use.

    Keys are virtual zarr keys ``g/kz/ky/kx`` (plus ``g/.zarray``, ``.zattrs``); bricks are cached
    compressed in RAM by the Level and on disk here (LRU, ``disk_cache_gb``) exactly like zarr chunks."""

    MAX_RANGES = 256  # byte ranges per request (dl.ash2txt.org accepts 400, refuses 800: request header size)

    def __init__(self, root: str, brick: int = 64, disk_cache: str | None = None, disk_cache_gb: float = 20.0,
                 n_levels: int = 4, pattern: str | None = None, clevel: int = 1):
        super().__init__(root, disk_cache, disk_cache_gb)
        self.local = not self.remote
        if self.local and disk_cache:  # the base class only caches remote stores
            self.disk = disk_cache
            os.makedirs(disk_cache, exist_ok=True)
        self.brick = int(brick)
        self.codec = numcodecs.Blosc(cname="zstd", clevel=clevel, shuffle=numcodecs.Blosc.SHUFFLE)
        self._hdr: dict[int, dict] = {}
        self._hlock = threading.Lock()
        self._staged: OrderedDict[str, bytes] = OrderedDict()  # built, not yet picked up (LRU, bounded)
        self._staged_bytes = 0
        self.staged_max = 256 * 2**20  # TiffVolume sets this to its RAM cache budget
        self._slock = threading.Lock()
        self.names = self._list_slices(pattern)
        h0 = self.header(0)
        self.dtype = np.dtype(h0["dtype"]).newbyteorder("=")
        self.shape = (len(self.names), h0["height"], h0["width"])
        self.n_levels = n_levels
        self.levels: list = []  # set by TiffVolume (coarser levels pool from finer ones)
        self.meta = None
        m = self._raw_file("meta.json")
        if m is not None:
            try:
                self.meta = json.loads(m)
            except ValueError:
                pass
        self.bytes_assembled = 0  # uncompressed bytes of bricks built from slices
        self.mispredicted = 0  # slices whose header differed from slice 0's (rows re-read)

    # ---- raw file access -------------------------------------------------------------------------
    def _raw_file(self, name):
        if self.local:
            p = os.path.join(self.root, name)
            if not os.path.exists(p):
                return None
            with open(p, "rb") as f:
                return f.read()
        r = self._session.get(f"{self.root}/{name}", timeout=60)
        with self._disk_lock:
            self.requests += 1
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with self._disk_lock:
            self.bytes_fetched += len(r.content)
        return r.content

    def _list_slices(self, pattern):
        if pattern is not None:
            n = json.loads(self._raw_file("meta.json"))["slices"]
            return [pattern.format(z) for z in range(n)]
        if self.local:
            names = sorted(f for f in os.listdir(self.root) if f.lower().endswith((".tif", ".tiff")))
        else:
            import re

            html = self._raw_file("").decode("utf-8", "replace")
            names = sorted(set(re.findall(r'href="([^"/?]+\.tiff?)"', html, re.I)))
        if not names:
            raise FileNotFoundError(f"no .tif slices in {self.root}")
        return names

    def _ranges(self, z: int, spans: list[tuple[int, int]], name: str | None = None) -> list[bytes]:
        """Bytes [a, b) of slice z (or of file ``name``) for each span (in order), in as few requests as possible."""
        out = [None] * len(spans)
        name = name or self.names[z]
        if self.local:
            with open(os.path.join(self.root, name), "rb") as f:
                for i, (a, b) in enumerate(spans):
                    f.seek(a)
                    out[i] = f.read(b - a)
            return out
        order = sorted(range(len(spans)), key=lambda i: spans[i][0])
        for g in range(0, len(order), self.MAX_RANGES):
            idx = order[g:g + self.MAX_RANGES]
            hdr = "bytes=" + ",".join(f"{spans[i][0]}-{spans[i][1] - 1}" for i in idx)
            r = self._session.get(f"{self.root}/{name}", headers={"Range": hdr}, timeout=120)
            r.raise_for_status()
            with self._disk_lock:
                self.requests += 1
                self.bytes_fetched += len(r.content)
            ct = r.headers.get("Content-Type", "")
            if r.status_code == 200:  # server ignored the range: whole file
                for i in idx:
                    out[i] = r.content[spans[i][0]:spans[i][1]]
            elif ct.startswith("multipart/byteranges"):
                parts = dict(_parse_multipart(r.content, ct))
                for i in idx:  # a server may coalesce adjacent ranges into one part
                    a, b = spans[i]
                    if a in parts and len(parts[a]) >= b - a:
                        out[i] = parts[a][:b - a]
                    else:
                        s0 = max(s for s in parts if s <= a)
                        out[i] = parts[s0][a - s0:b - s0]
            else:  # single range
                assert len(idx) == 1
                out[idx[0]] = r.content
        return out

    def header(self, z: int) -> dict:
        """Parsed TIFF layout of slice z (fetched once, cached in RAM and in the disk cache)."""
        with self._hlock:
            h = self._hdr.get(z)
        if h is not None:
            return h
        p = self._disk_path(f"hdr_{self.names[z]}") if self.disk else None
        if p and os.path.exists(p):
            with open(p) as f:
                h = json.load(f)
        else:
            h = parse_tiff_header(lambda o, n: self._ranges(z, [(o, o + n)])[0], size_hint=self.HEAD)
            if p:
                self._disk_put(p, json.dumps(h).encode())
        with self._hlock:
            self._hdr[z] = h
        return h

    # ---- bricks ----------------------------------------------------------------------------------
    def zarray(self, g: int) -> bytes:
        s = tuple(-(-n // 2 ** g) for n in self.shape)
        return json.dumps({"zarr_format": 2, "shape": list(s), "chunks": [self.brick] * 3, "dtype": self.dtype.str,
                           "compressor": self.codec.get_config(), "fill_value": 0, "filters": None,
                           "order": "C", "dimension_separator": "/"}).encode()

    HEAD = 1024  # bytes of a slice requested for its header (the IFD of these stacks sits at offset 8)
    FULL_FRAC = 0.3  # a brick row needing >= this fraction of the width is read as full rows (one range)
    PAD = 1  # also build this many bricks either side of the needed ones (later tiles need them; ~free)

    @staticmethod
    def _spans(h, y0, y1, xspans):
        """Byte spans of columns ``xspans`` of rows [y0, y1); full-width spans of adjacent rows (contiguous
        strips) are merged into one."""
        isz = np.dtype(h["dtype"]).itemsize
        rb = h["width"] * isz
        rps, so = h["rows_per_strip"], h["strip_offsets"]
        full = len(xspans) == 1 and xspans[0] == (0, h["width"])
        spans = []
        for y in range(y0, y1):
            base = so[y // rps] + (y % rps) * rb
            if full and spans and spans[-1][1] == base:
                spans[-1] = (spans[-1][0], base + rb)
                continue
            for a, b in xspans:
                spans.append((base + a * isz, base + b * isz))
        return spans

    def build_bricks(self, kz: int, rows: dict):
        """Assemble the level-0 bricks (kz, ky, kx) for ky, kxs in ``rows``, plus PAD neighbours in x; a brick
        row that needs >= FULL_FRAC of the width is read at full width and built entirely.  Per slice, all
        pieces (and an unseen slice's header) go into as few range requests as possible: this server serves
        ~40-130 requests/s whatever their size, so the request count, not the bytes, sets the wall time."""
        B = self.brick
        Z, Y, X = self.shape
        nx = -(-X // B)
        z0, z1 = kz * B, min((kz + 1) * B, Z)
        plan = {}  # ky -> (kxs, xspans)
        for ky, kxs in rows.items():
            ks = sorted({k + d for k in kxs for d in range(-self.PAD, self.PAD + 1) if 0 <= k + d < nx})
            if len(ks) * B >= self.FULL_FRAC * X:
                ks = list(range(nx))
            xspans = []
            for kx in ks:
                a, b = kx * B, min((kx + 1) * B, X)
                if xspans and xspans[-1][1] == a:
                    xspans[-1] = (xspans[-1][0], b)
                else:
                    xspans.append((a, b))
            plan[ky] = (ks, xspans)
        bricks = {(ky, kx): np.zeros((B, B, B), self.dtype) for ky, (ks, _) in plan.items() for kx in ks}

        def layout(h):
            spans, where = [], []
            for ky, (_, xspans) in plan.items():
                ya, yb = ky * B, min((ky + 1) * B, Y)
                sp = self._spans(h, ya, yb, xspans)
                where.append((ky, len(spans), len(sp)))
                spans += sp
            return spans, where

        def slice_job(z):
            with self._hlock:
                h = self._hdr.get(z)
            if h is None and self.disk and os.path.exists(self._disk_path(f"hdr_{self.names[z]}")):
                h = self.header(z)
            if h is not None:
                spans, where = layout(h)
                parts = self._ranges(z, spans)
            else:  # unseen slice: predict its layout from slice 0, fetch header + pieces together, verify
                pred = self.header(0)
                spans, where = layout(pred)
                parts = self._ranges(z, [(0, self.HEAD)] + spans)
                head, parts = parts[0], parts[1:]
                h = parse_tiff_header(lambda o, n: head[o:o + n] if o + n <= len(head) else
                                      self._ranges(z, [(o, o + n)])[0], size_hint=self.HEAD)
                with self._hlock:
                    self._hdr[z] = h
                if self.disk:
                    self._disk_put(self._disk_path(f"hdr_{self.names[z]}"), json.dumps(h).encode())
                if h != pred:  # different layout: the speculative pieces are wrong, read them again
                    self.mispredicted += 1
                    spans, where = layout(h)
                    parts = self._ranges(z, spans)
            sdt = np.dtype(h["dtype"])  # per slice: byte order / depth may differ between files
            for ky, i0, n in where:
                ya, yb = ky * B, min((ky + 1) * B, Y)
                xspans = plan[ky][1]
                buf = np.frombuffer(b"".join(parts[i0:i0 + n]), sdt).reshape(yb - ya, -1)
                c = 0
                for a, b in xspans:
                    seg = buf[:, c:c + b - a]
                    c += b - a
                    for kx in range(a // B, -(-b // B)):
                        c0, c1 = max(a, kx * B), min(b, (kx + 1) * B)
                        bricks[(ky, kx)][z - z0, :yb - ya, c0 - kx * B:c1 - kx * B] = seg[:, c0 - a:c1 - a]

        zs = list(range(z0, z1))
        if self.local or len(zs) == 1:
            for z in zs:
                slice_job(z)
        else:
            list(_executor("tiffio", IO_WORKERS).map(slice_job, zs))
        out = {}
        for (ky, kx), a in bricks.items():
            out[f"0/{kz}/{ky}/{kx}"] = self.codec.encode(np.ascontiguousarray(a))
            self.bytes_assembled += a.nbytes
        return out

    def build_row(self, kz: int, ky: int, kxs: list[int]):
        return self.build_bricks(kz, {ky: list(kxs)})

    def build_coarse(self, g: int, kz: int, ky: int, kx: int) -> bytes:
        """Level g brick: 2x2x2 mean (rounded) of the level g-1 bricks below it."""
        B = self.brick
        fine = self.levels[g - 1]
        grid = [-(-s // B) for s in fine.shape]
        fine.prefetch([(z, y, x) for z in (2 * kz, 2 * kz + 1) for y in (2 * ky, 2 * ky + 1)
                       for x in (2 * kx, 2 * kx + 1) if z < grid[0] and y < grid[1] and x < grid[2]])
        a = fine.read(2 * kz * B, 2 * (kz + 1) * B, 2 * ky * B, 2 * (ky + 1) * B, 2 * kx * B, 2 * (kx + 1) * B)
        m = a.reshape(B, 2, B, 2, B, 2).astype(np.float64).mean(axis=(1, 3, 5))
        return self.codec.encode(np.ascontiguousarray(np.rint(m).astype(self.dtype)))

    def stage(self, items: dict):
        """Keep freshly built bricks until the Level picks them up (and write them to the disk cache)."""
        with self._slock:  # bricks built speculatively (neighbours, full rows) may never be asked for:
            for k, v in items.items():  # keep the most recent within budget (evicted ones are on disk or rebuilt)
                old = self._staged.pop(k, None)
                self._staged_bytes += len(v) - (len(old) if old is not None else 0)
                self._staged[k] = v
            while self._staged_bytes > self.staged_max and len(self._staged) > 1:
                self._staged_bytes -= len(self._staged.popitem(last=False)[1])
        if self.disk:
            for k, v in items.items():
                self._disk_put(self._disk_path(k), v)

    def get(self, key: str) -> bytes | None:
        if key == ".zattrs":
            return json.dumps({"multiscales": [{"version": "0.4", "datasets": [{"path": str(g)} for g in
                               range(self.n_levels)], "axes": [{"name": a, "type": "space"} for a in "zyx"]}]}).encode()
        if key == "meta.json":
            vs = (self.meta or {}).get("voxelsize")
            return json.dumps({"voxelsize": vs}).encode() if vs else None
        g, _, rest = key.partition("/")
        if rest == ".zarray":
            return self.zarray(int(g)) if int(g) < self.n_levels else None
        if rest == ".zgroup" or not rest:
            return None
        with self._slock:
            v = self._staged.pop(key, None)
            if v is not None:
                self._staged_bytes -= len(v)
        if v is not None:
            return v
        if self.disk:
            p = self._disk_path(key)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = f.read()
                with self._disk_lock:
                    if p in self._disk_lru:
                        self._disk_lru.move_to_end(p)
                return data
        kz, ky, kx = map(int, rest.split("/"))
        if int(g) == 0:
            built = self.build_row(kz, ky, [kx])
            if self.disk:
                self._disk_put(self._disk_path(key), built[key])
            return built[key]
        v = self.build_coarse(int(g), kz, ky, kx)
        if self.disk:
            self._disk_put(self._disk_path(key), v)
        return v


class TiffLevel(Level):
    """A Level whose chunks are bricks of a TIFF slice stack: missing bricks of one brick row are
    fetched together (one range request per slice), then decoded into the sampling pool as usual."""

    def __init__(self, store: TiffStackStore, g: int, cache_bytes: int, decoded_bytes: int):
        super().__init__(store, str(g), cache_bytes, decoded_bytes)
        self.g = g

    def prefetch(self, keys, workers: int | None = None):
        with self._lock:
            keys = [k for k in keys if k not in self._cache and k not in self._raw]
        st = self.store
        if self.g == 0 and keys:
            todo = {}
            for k in keys:
                name = f"0/{k[0]}/{k[1]}/{k[2]}"
                if st.disk and os.path.exists(st._disk_path(name)):
                    continue
                todo.setdefault((k[0], k[1]), []).append(k[2])
            byz = {}
            for (kz, ky), kxs in todo.items():
                byz.setdefault(kz, {})[ky] = kxs
            groups = list(byz.items())
            if len(groups) == 1 or st.local:
                for kz, rows in groups:
                    st.stage(st.build_bricks(kz, rows))
            elif groups:  # brick layers in parallel; each parallelises over its slices on the shared io pool
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(min(4, len(groups))) as ex:
                    for built in ex.map(lambda g: st.build_bricks(g[0], g[1]), groups):
                        st.stage(built)
        for k in keys:
            self._raw_get(k)

    def read(self, z0, z1, y0, y1, x0, x1) -> np.ndarray:
        cz, cy, cx = self.chunks
        g = [-(-s // c) for s, c in zip(self.shape, self.chunks)]
        self.prefetch([(kz, ky, kx) for kz in range(max(z0, 0) // cz, min(-(-z1 // cz), g[0]))
                       for ky in range(max(y0, 0) // cy, min(-(-y1 // cy), g[1]))
                       for kx in range(max(x0, 0) // cx, min(-(-x1 // cx), g[2]))])
        return super().read(z0, z1, y0, y1, x0, x1)

    def prefetch_async(self, keys):
        return []  # brick assembly is batched in prefetch(); background readahead is not implemented


class TiffVolume(Volume):
    """``Volume`` over an uncompressed TIFF slice stack (directory or URL), streamed by range requests.

    ``brick``: edge of the cubic bricks the stack is assembled into (the sampling pool's chunks);
    ``cache_gb`` / ``decoded_cache_gb`` / ``disk_cache`` / ``disk_cache_gb``: as for ``Volume``
    (compressed bricks in RAM, decoded bricks in the pool, persistent compressed bricks + headers on disk);
    ``levels``: number of pyramid levels (level g is 2x2x2 mean pooled from g-1)."""

    def __init__(self, path: str, cache_gb: float = 4.0, disk_cache: str | None = None, disk_cache_gb: float = 20.0,
                 decoded_cache_gb: float = 0.5, readahead: bool | None = None, brick: int = 64, levels: int = 4,
                 pattern: str | None = None):
        self.store = st = TiffStackStore(path, brick, disk_cache, disk_cache_gb, n_levels=levels, pattern=pattern)
        self.path = path
        self.voxel_size = (st.meta or {}).get("voxelsize")
        cache = int(cache_gb * 2**30)
        self.levels = [TiffLevel(st, g, cache, int(decoded_cache_gb * 2**30)) for g in range(levels)]
        st.levels = self.levels
        st.staged_max = cache

    @property
    def dtype(self):
        return self.store.dtype

    def stats(self) -> dict:
        s = super().stats()
        s["brick_bytes_assembled"] = self.store.bytes_assembled
        return s


def open_volume(path: str, **kw) -> Volume:
    """A Volume for an OME-Zarr / zarr array, or a TiffVolume for a directory / URL of .tif slices."""
    if not _is_url(path) and os.path.isdir(path) and any(f.lower().endswith((".tif", ".tiff")) for f in os.listdir(path)):
        return TiffVolume(path, **kw)
    return Volume(path, **kw)


# ---------------------------------------------------------------------------------------------------
# Cell grid: the same volume as 500^3 multi-page TIFF cells (``volume_grids/<uuid>/cell_yxz_YYY_XXX_ZZZ.tif``)
# ---------------------------------------------------------------------------------------------------


class CellGridStore(_Store):
    """Key -> blosc-compressed chunk bytes, cut from a grid of uncompressed multi-page TIFF cells.

    Cell ``cell_yxz_{iy:03d}_{ix:03d}_{iz:03d}.tif`` (1-based) holds voxels [(i-1)*C, i*C) per axis as C
    pages of C x C uint16.  A chunk is (cz, cy, C): one row band of ``cy`` full cell-width rows from each
    of ``cz`` consecutive pages -- each band is contiguous in the file, so a chunk is ONE request of cz
    byte ranges (vs one request per z slice for the slice stack)."""

    MAX_RANGES = 256

    def __init__(self, root: str, cell: int | None = None, chunk=(50, 50),
                 disk_cache: str | None = None, disk_cache_gb: float = 20.0, n_levels: int = 4, clevel: int = 1,
                 shape=None):
        super().__init__(root, disk_cache, disk_cache_gb)
        self.local = not self.remote
        if self.local and disk_cache:
            self.disk = disk_cache
            os.makedirs(disk_cache, exist_ok=True)
        self._cells: dict[tuple, dict | None] = {}
        self._clock = threading.Lock()
        if cell is None:  # cell edge = width of the first cell (all cells are padded to the same size)
            first = self.cell((1, 1, 1))
            if first is None:
                raise FileNotFoundError(f"no {self.cell_name(1, 1, 1)} in {self.root}")
            cell = first["width"]
        self.C = int(cell)
        cz, cy = chunk
        if self.C % cz or self.C % cy:
            raise ValueError("chunk sizes must divide the cell size")
        self.chunks = (cz, cy, self.C)
        self.codec = numcodecs.Blosc(cname="zstd", clevel=clevel, shuffle=numcodecs.Blosc.SHUFFLE)
        self.meta = None
        if shape is None:  # the volume's meta.json (volpkg/volumes/<uuid>/): edge cells are zero-padded to C
            base, _, uuid = self.root.rstrip("/").rpartition("/")
            m = self._raw(f"../../volumes/{uuid}/meta.json") if base.endswith("volume_grids") else None
            if m is not None:
                try:
                    self.meta = json.loads(m)
                    shape = (self.meta["slices"], self.meta["height"], self.meta["width"])
                except (ValueError, KeyError):
                    shape = None
        if shape is None:  # fall back to the cell grid (edge cells may be padded)
            import re

            html = self._raw("").decode("utf-8", "replace")
            idx = np.array([list(map(int, m)) for m in re.findall(r'cell_yxz_(\d+)_(\d+)_(\d+)\.tif', html)])
            n = idx.max(0)  # cells per axis (y, x, z)
            last = {a: self.cell((n[0] if a == 0 else 1, n[1] if a == 1 else 1, n[2] if a == 2 else 1))
                    for a in range(3)}
            shape = ((n[2] - 1) * self.C + last[2]["pages"], (n[0] - 1) * self.C + last[0]["height"],
                     (n[1] - 1) * self.C + last[1]["width"])
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype("<u2")
        self.n_levels = n_levels
        self.levels: list = []
        self.bytes_assembled = 0

    def _raw(self, name, rng=None):
        if not self.remote:
            p = os.path.join(self.root, name)
            if not os.path.exists(p):
                return None
            with open(p, "rb") as f:
                if rng is None:
                    return f.read()
                f.seek(rng[0])
                return f.read(rng[1] - rng[0])
        r = self._session.get(f"{self.root}/{name}", timeout=120,
                              headers={"Range": f"bytes={rng[0]}-{rng[1] - 1}"} if rng else None)
        with self._disk_lock:
            self.requests += 1
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with self._disk_lock:
            self.bytes_fetched += len(r.content)
        return r.content

    @staticmethod
    def cell_name(iy, ix, iz):
        return f"cell_yxz_{iy:03d}_{ix:03d}_{iz:03d}.tif"

    def cell(self, i):
        """Layout of cell i = (iy, ix, iz) (1-based): width, height, pages, dtype, page data offsets (parsed
        from the first IFD; the pages are verified to be equally spaced from the second).  None if absent."""
        with self._clock:
            if i in self._cells:
                return self._cells[i]
        name = self.cell_name(*i)
        p = self._disk_path(f"cellhdr_{name}") if self.disk else None
        if p and os.path.exists(p):
            with open(p) as f:
                lay = json.load(f)
        else:
            head = self._raw(name, (0, 8))
            lay = None
            if head is not None:
                bo = "<" if head[:2] == b"II" else ">"
                blk = {}

                def get(o, n):  # each IFD (and its entries) lies in one 1 KB window: one request per IFD
                    for (a0, b0), v in blk.items():
                        if a0 <= o and o + n <= b0:
                            return v[o - a0:o - a0 + n]
                    v = self._raw(name, (o, o + max(n, 1024)))
                    blk[(o, o + len(v))] = v
                    return v[:n]

                t0, nxt = read_ifd(get, bo, struct.unpack(bo + "I", head[4:8])[0])
                t1, _ = read_ifd(get, bo, nxt)
                h0, h1 = _layout(t0, bo), _layout(t1, bo)
                W, H = h0["width"], h0["height"]
                if len(h0["strip_offsets"]) != 1 or h0["rows_per_strip"] < H:
                    raise ValueError("cell pages must be single-strip")
                step = h1["strip_offsets"][0] - h0["strip_offsets"][0]
                size = int(self._head_size(name))
                # pages equally spaced by ``step`` (IFD before or after each page's data): the last page's data
                # must end inside the file
                pages = (size - h0["strip_offsets"][0] - W * H * np.dtype(h0["dtype"]).itemsize) // step + 1
                if pages < 1:
                    raise ValueError(f"{name}: unexpected page layout")
                lay = {"width": W, "height": H, "pages": int(pages), "dtype": h0["dtype"],
                       "offset0": h0["strip_offsets"][0], "step": int(step), "bo": bo}
            if p and lay is not None:
                self._disk_put(p, json.dumps(lay).encode())
        if lay is not None:  # an absent cell is not remembered: a transient 404 must not stick
            with self._clock:
                self._cells[i] = lay
        return lay

    def _head_size(self, name):
        if not self.remote:
            return os.path.getsize(os.path.join(self.root, name))
        r = self._session.head(f"{self.root}/{name}", timeout=60)
        with self._disk_lock:
            self.requests += 1
        r.raise_for_status()  # an error page's length would give a wrong page count (cached on disk)
        return int(r.headers["Content-Length"])

    def zarray(self, g):
        s = tuple(-(-n // 2 ** g) for n in self.shape)
        ch = self.chunks if g == 0 else (64, 64, 64)
        return json.dumps({"zarr_format": 2, "shape": list(s), "chunks": list(ch), "dtype": self.dtype.str,
                           "compressor": self.codec.get_config(), "fill_value": 0, "filters": None,
                           "order": "C", "dimension_separator": "/"}).encode()

    def build(self, kz, ky, kx):
        """Level-0 chunk (kz, ky, kx): ``cz`` pages x ``cy`` rows of cell (iy, ix, iz) in one request."""
        cz, cy, C = self.chunks
        z0, y0 = kz * cz, ky * cy
        i = (y0 // C + 1, kx + 1, z0 // C + 1)
        out = np.zeros(self.chunks, self.dtype)
        lay = self.cell(i)
        if lay is None:  # every cell inside the volume exists: absent must not read as fill value 0
            raise FileNotFoundError(f"{self.cell_name(*i)} is missing from {self.root} (chunk {kz}/{ky}/{kx})")
        pz0, py0 = z0 - (i[2] - 1) * C, y0 - (i[0] - 1) * C
        npg, nrow = min(cz, lay["pages"] - pz0), min(cy, lay["height"] - py0)
        if npg <= 0 or nrow <= 0:
            return self.codec.encode(out)
        rb = lay["width"] * 2
        spans = [(lay["offset0"] + (pz0 + k) * lay["step"] + py0 * rb,
                  lay["offset0"] + (pz0 + k) * lay["step"] + (py0 + nrow) * rb) for k in range(npg)]
        name = self.cell_name(*i)
        if self.remote:
            parts = TiffStackStore._ranges(self, None, spans, name=name)
        else:
            parts = [self._raw(name, s) for s in spans]
        dt = np.dtype(lay["bo"] + "u2")
        for k, b in enumerate(parts):
            out[k, :nrow, :lay["width"]] = np.frombuffer(b, dt).reshape(nrow, lay["width"])
        self.bytes_assembled += out.nbytes
        return self.codec.encode(out)

    def get(self, key):
        if key == ".zattrs":
            return json.dumps({"multiscales": [{"version": "0.4", "datasets": [{"path": str(g)} for g in
                               range(self.n_levels)], "axes": [{"name": a, "type": "space"} for a in "zyx"]}]}).encode()
        if key == "meta.json":
            return None
        g, _, rest = key.partition("/")
        if rest == ".zarray":
            return self.zarray(int(g)) if int(g) < self.n_levels else None
        if rest == ".zgroup" or not rest:
            return None
        if self.disk:
            p = self._disk_path(key)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = f.read()
                with self._disk_lock:
                    if p in self._disk_lru:
                        self._disk_lru.move_to_end(p)
                return data
        kz, ky, kx = map(int, rest.split("/"))
        v = self.build(kz, ky, kx) if int(g) == 0 else self.build_coarse(int(g), kz, ky, kx)
        if self.disk and v is not None:
            self._disk_put(self._disk_path(key), v)
        return v

    def build_coarse(self, g, kz, ky, kx):
        B = 64
        fine = self.levels[g - 1]
        a = fine.read(2 * kz * B, 2 * (kz + 1) * B, 2 * ky * B, 2 * (ky + 1) * B, 2 * kx * B, 2 * (kx + 1) * B)
        m = a.reshape(B, 2, B, 2, B, 2).astype(np.float64).mean(axis=(1, 3, 5))
        return self.codec.encode(np.ascontiguousarray(np.rint(m).astype(self.dtype)))


class _GridLevel(Level):
    def __init__(self, store, g, cache_bytes, decoded_bytes):
        super().__init__(store, str(g), cache_bytes, decoded_bytes)

    def read(self, z0, z1, y0, y1, x0, x1):
        cz, cy, cx = self.chunks
        g = [-(-s // c) for s, c in zip(self.shape, self.chunks)]
        self.prefetch([(kz, ky, kx) for kz in range(max(z0, 0) // cz, min(-(-z1 // cz), g[0]))
                       for ky in range(max(y0, 0) // cy, min(-(-y1 // cy), g[1]))
                       for kx in range(max(x0, 0) // cx, min(-(-x1 // cx), g[2]))])
        return super().read(z0, z1, y0, y1, x0, x1)


class CellGridVolume(Volume):
    """``Volume`` over the 500^3 TIFF cell grid of a scan (e.g. Scroll 1 ``volume_grids/20230205180739``):
    uint16, chunks of (cz, cy, 500) fetched with one range request each.  Knobs as for ``Volume``."""

    def __init__(self, path: str, cache_gb: float = 4.0, disk_cache: str | None = None, disk_cache_gb: float = 20.0,
                 decoded_cache_gb: float = 1.0, readahead: bool | None = None, chunk=(50, 50), levels: int = 4,
                 shape=None, voxel_size=None, cell: int | None = None):
        self.store = st = CellGridStore(path, cell=cell, chunk=chunk, disk_cache=disk_cache, disk_cache_gb=disk_cache_gb,
                                        n_levels=levels, shape=shape)
        self.path = path
        self.voxel_size = voxel_size or (st.meta or {}).get("voxelsize")
        cache = int(cache_gb * 2**30)
        self.levels = [_GridLevel(st, g, cache, int(decoded_cache_gb * 2**30)) for g in range(levels)]
        st.levels = self.levels

    @property
    def dtype(self):
        return self.store.dtype

    def stats(self) -> dict:
        s = super().stats()
        s["brick_bytes_assembled"] = self.store.bytes_assembled
        return s
