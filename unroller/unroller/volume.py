"""OME-Zarr (zarr v2) volume access, local or over HTTP, with an LRU chunk cache.

Conventions (identical to VC3D):
  * arrays are stored Z, Y, X;
  * every public coordinate in this package is (x, y, z) in *level-0 voxels*;
  * level g is downsampled by 2**g in every axis.
"""
from __future__ import annotations

import functools
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numcodecs


def _is_url(p: str) -> bool:
    return p.startswith("http://") or p.startswith("https://")


_POOLS: dict = {}
_POOLS_LOCK = threading.Lock()


def _executor(kind: str, workers: int) -> ThreadPoolExecutor:
    """Process-wide thread pools (created once): 'io' for HTTP/disk fetches, 'cpu' for chunk decoding
    (blosc releases the GIL when called from worker threads)."""
    with _POOLS_LOCK:
        ex = _POOLS.get(kind)
        if ex is None:
            ex = _POOLS[kind] = ThreadPoolExecutor(workers, thread_name_prefix=f"unroller-{kind}")
        return ex


# Scroll 1 over HTTPS: 51 MB/s with 8, 59 with 16, 57 with 32 workers (scripts/perf.py --remote); the 2025 open-data
# S3 scans (uncompressed 2 MB chunks): 54-65 MB/s for 8-128 workers (bandwidth-bound).  UNROLLER_IO_WORKERS overrides.
IO_WORKERS = int(os.environ.get("UNROLLER_IO_WORKERS", 16))
CPU_WORKERS = os.cpu_count() or 4


HTTP_RETRIES = int(os.environ.get("UNROLLER_HTTP_RETRIES", 5))


@functools.lru_cache(maxsize=None)
def _retry_session_cls():
    import requests

    class _RetrySession(requests.Session):
        """``requests.Session`` that also retries failures urllib3's ``max_retries`` does not cover: a body cut off
        mid-transfer (``ChunkedEncodingError``, e.g. a relay dropping the connection), connection resets, read
        timeouts and transient 429/5xx answers.  Exponential backoff with jitter; the last error is re-raised."""

        retries = HTTP_RETRIES
        backoff = 1.0

        def request(self, method, url, *args, **kw):
            import random
            import time

            transient = (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                         requests.exceptions.ReadTimeout, requests.exceptions.ContentDecodingError)
            for k in range(self.retries + 1):
                try:
                    r = super().request(method, url, *args, **kw)
                except transient:
                    if k == self.retries:
                        raise
                else:
                    if r.status_code not in (429, 500, 502, 503, 504) or k == self.retries:
                        return r
                time.sleep(self.backoff * 2 ** k * (0.5 + random.random()))

    return _RetrySession


def _RetrySession():
    return _retry_session_cls()()


class _Store:
    """Key -> bytes.  For remote stores an optional on-disk cache keeps chunks exactly as served
    (still compressed), capped at ``disk_cache_gb`` with least-recently-used eviction, so a
    0.9 TB scroll never needs more local disk than the cap."""

    def __init__(self, root: str, disk_cache: str | None = None, disk_cache_gb: float = 20.0):
        self.root = root.rstrip("/")
        self.remote = _is_url(root)
        self.bytes_fetched = 0
        self.requests = 0
        if self.remote:
            import requests
            from requests.adapters import HTTPAdapter

            # keep-alive connections, enough of them for the parallel fetch pool (requests' default is 10,
            # beyond which every extra concurrent request opens -- and discards -- a new TLS connection)
            self._session = _RetrySession()
            ad = HTTPAdapter(pool_connections=4, pool_maxsize=IO_WORKERS, max_retries=3)
            self._session.mount("https://", ad)
            self._session.mount("http://", ad)
        self.disk = disk_cache if self.remote else None
        self._disk_max = int(disk_cache_gb * 2**30)
        self._disk_lru: OrderedDict[str, int] = OrderedDict()
        self._disk_bytes = 0
        self._disk_lock = threading.Lock()
        if self.disk:
            os.makedirs(self.disk, exist_ok=True)
            files = []
            for dp, _, fns in os.walk(self.disk):
                for fn in fns:
                    p = os.path.join(dp, fn)
                    st = os.stat(p)
                    files.append((st.st_mtime, p, st.st_size))
            for _, p, sz in sorted(files):
                self._disk_lru[p] = sz
                self._disk_bytes += sz

    def _disk_path(self, key):
        return os.path.join(self.disk, key.replace("/", "_") or "_root")

    def get(self, key: str) -> bytes | None:
        if not self.remote:
            path = os.path.join(self.root, key)
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                return f.read()
        if self.disk:
            p = self._disk_path(key)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = f.read()
                with self._disk_lock:
                    if p in self._disk_lru:
                        self._disk_lru.move_to_end(p)
                return data
        r = self._session.get(f"{self.root}/{key}", timeout=60)
        if r.status_code == 404:
            with self._disk_lock:
                self.requests += 1
            return None
        r.raise_for_status()
        data = r.content
        with self._disk_lock:  # counters are updated from many fetch threads
            self.requests += 1
            self.bytes_fetched += len(data)
        if self.disk:
            self._disk_put(self._disk_path(key), data)
        return data

    def _disk_put(self, p, data):
        tmp = f"{p}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
        with self._disk_lock:
            self._disk_bytes += len(data) - self._disk_lru.pop(p, 0)
            self._disk_lru[p] = len(data)
            while self._disk_bytes > self._disk_max and len(self._disk_lru) > 1:
                old, sz = self._disk_lru.popitem(last=False)
                self._disk_bytes -= sz
                try:
                    os.remove(old)
                except FileNotFoundError:
                    pass


_NO_NRM = np.empty((0, 3))
_NO_OFFS = np.zeros(1)


def _split_order(q, f, cz):
    """Order splitting points into two spatially coherent halves (by chunk row in z, then y)."""
    key = np.floor(np.nan_to_num(q[:, 2] * f) / cz) * 1e6 + np.floor(np.nan_to_num(q[:, 1] * f))
    return np.argsort(key, kind="stable")


class Level:
    """One resolution level of the pyramid (a single zarr v2 array)."""

    def __init__(self, store: _Store, path: str, cache_bytes: int, decoded_bytes: int = 512 * 2**20):
        self.store, self.path = store, path
        self.prefix = f"{path}/" if path else ""
        meta = store.get(f"{self.prefix}.zarray")
        if meta is None:
            raise FileNotFoundError(f"no .zarray at {store.root}/{path}")
        m = json.loads(meta)
        if m.get("zarr_format") != 2:
            raise ValueError("only zarr v2 arrays are supported")
        if m.get("order", "C") != "C":
            raise ValueError("only C-order arrays are supported")
        self.shape = tuple(m["shape"])
        self.chunks = tuple(m["chunks"])
        self.dtype = np.dtype(m["dtype"])
        self.fill = m.get("fill_value") or 0
        self.sep = m.get("dimension_separator", ".")
        self.codec = numcodecs.get_codec(m["compressor"]) if m.get("compressor") else None
        self.filters = [numcodecs.get_codec(f) for f in (m.get("filters") or [])]
        # two tiers: compressed bytes (large budget, ~4-10x denser) and decoded arrays (small budget)
        self._raw: OrderedDict[tuple, bytes | None] = OrderedDict()
        self._raw_bytes, self._raw_max = 0, cache_bytes
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cache_bytes, self._cache_max = 0, max(decoded_bytes // 4, 1)
        self._lock = threading.Lock()
        # sampling pool: decoded chunks in fixed slots + chunk-grid -> slot table, read by compiled kernels
        self._pool_bytes = decoded_bytes
        self._pool = None
        self._plock = threading.RLock()
        self._inflight: dict = {}  # key -> Future of a background fetch
        self.readahead = False  # prefetch the face neighbours of newly needed chunks (see _ensure)
        self.decoded = 0  # chunks decoded into the sampling pool (cache misses)

    def _decode(self, raw: bytes) -> np.ndarray:
        buf = self.codec.decode(raw) if self.codec else raw
        for f in reversed(self.filters):
            buf = f.decode(buf)
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.chunks)

    def _raw_get(self, key):
        with self._lock:
            if key in self._raw:
                self._raw.move_to_end(key)
                return self._raw[key]
        raw = self.store.get(self.prefix + self.sep.join(map(str, key)))
        with self._lock:
            self._raw[key] = raw
            self._raw_bytes += len(raw) if raw else 0
            while self._raw_bytes > self._raw_max and len(self._raw) > 1:
                _, old = self._raw.popitem(last=False)
                self._raw_bytes -= len(old) if old else 0
        return raw

    def chunk(self, key: tuple[int, int, int]) -> np.ndarray:
        with self._lock:
            c = self._cache.get(key)
            if c is not None:
                self._cache.move_to_end(key)
                return c
        raw = self._raw_get(key)
        c = np.full(self.chunks, self.fill, self.dtype) if raw is None else self._decode(raw)
        with self._lock:
            self._cache[key] = c
            self._cache_bytes += c.nbytes
            while self._cache_bytes > self._cache_max and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= old.nbytes
        return c

    def _pool_init(self):
        cb = int(np.prod(self.chunks)) * self.dtype.itemsize
        n = int(max(16, self._pool_bytes // cb))
        self._pool = np.empty((n,) + self.chunks, self.dtype)
        grid = tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))
        self._table = np.full(grid, -1, np.int32)  # -1 not loaded, -2 absent in store (fill), >=0 slot
        self._mark = np.zeros(grid, np.uint8)  # scratch for the kernels, always all-zero between calls
        # LRU by recency stamps: the sampling kernels write stamp[slot] = epoch for every slot they read;
        # free slots have stamp -1 and cell -1 (flat chunk-grid index of the chunk held by a slot)
        self._stamp = np.full(n, -1, np.int64)
        self._slot_cell = np.full(n, -1, np.int64)
        self._epoch = 0

    def _decode_into(self, raw: bytes, dst: np.ndarray):
        if self.codec is not None and not self.filters:
            self.codec.decode(raw, out=dst)  # straight into the pool slot, no temporary
        else:
            dst[...] = self._decode(raw)

    def _load_cells(self, cells):
        """Fetch (parallel HTTP / disk) and decode (parallel threads, straight into the pool) the chunks
        at flat grid indices ``cells``, evicting the least recently used slots of earlier calls."""
        grid = self._table.shape
        keys = list(zip(*(c.tolist() for c in np.unravel_index(cells, grid))))
        self.prefetch(keys)  # parallel fetch of the compressed bytes
        raws = [self._raw_get(k) for k in keys]
        present = [i for i, r in enumerate(raws) if r is not None]
        tflat = self._table.reshape(-1)
        for i, r in enumerate(raws):
            if r is None:
                tflat[cells[i]] = -2  # absent from the store: fill value
        if not present:
            return
        k = len(present)
        # slots never stamped in this call (stamp < epoch) are the eviction candidates; the caller
        # guarantees enough of them exist (needed cells <= pool size)
        slots = np.argpartition(self._stamp, k - 1)[:k] if k < len(self._stamp) else np.arange(k)
        assert np.all(self._stamp[slots] < self._epoch)
        old = self._slot_cell[slots]
        tflat[old[old >= 0]] = -1
        new = cells[present]
        tflat[new] = slots
        self._slot_cell[slots] = new
        self._stamp[slots] = self._epoch
        jobs = [(raws[i], int(s)) for i, s in zip(present, slots)]
        if len(jobs) > 1:  # decode in parallel threads straight into the pool
            list(_executor("cpu", CPU_WORKERS).map(lambda j: self._decode_into(j[0], self._pool[j[1]]), jobs))
        else:
            self._decode_into(jobs[0][0], self._pool[jobs[0][1]])
        self.decoded += len(jobs)

    def _ensure(self, q, nrm, offs, f, order):
        """Make every chunk the points need resident in the pool.  False if they do not all fit."""
        from . import _kernels as K

        bb = K.call(K.mark_cells, len(q) * max(len(nrm) and len(offs), 1), self._mark, *self.chunks, self.shape,
                    q, nrm, offs, f, order)
        cells = K.collect_cells(self._mark, bb)
        if len(cells) > len(self._pool):
            return False
        st = self._table.reshape(-1)[cells]
        self._stamp[st[st >= 0]] = self._epoch  # protect resident chunks this call needs
        need = cells[st == -1]
        if len(need):
            self._load_cells(need)
            if self.readahead:
                self._readahead(need)
        return True

    def _run(self, kernel, args, ensure_args, optimistic=True, work=0):
        """Sample optimistically from the resident chunks; only if some point touched a missing chunk,
        find and load the missing chunks and sample again.  False if the chunks needed do not fit.
        optimistic=False skips the first pass (pieces of a split batch: their chunks are mostly absent)."""
        if self._pool is None:
            self._pool_init()
        self._epoch += 1
        from . import _kernels as K

        common = (self._pool, self._table, float(self.fill), *self.chunks, self.shape)
        if optimistic and K.call(kernel, work, *common, *args, self._stamp, self._epoch) == 0:
            return True
        if not self._ensure(*ensure_args):
            return False
        miss = K.call(kernel, work, *common, *args, self._stamp, self._epoch)
        assert miss == 0
        return True

    def sample_points(self, p: np.ndarray, f: float = 1.0, order: int = 1) -> np.ndarray:
        """Sample at (N,3) xyz level-0 positions; f = 2**-level (compiled kernel)."""
        q = np.ascontiguousarray(p, dtype=np.float64)
        out = np.empty(len(q), np.float32)
        if len(q) == 0:
            return out
        with self._plock:
            self._sample_into(q, None, None, f, order, out, True)
        return out

    def sample_layers(self, p: np.ndarray, nrm: np.ndarray, offs: np.ndarray, f: float = 1.0) -> np.ndarray:
        """Trilinear samples (L, N) at p + offs[l] * nrm (xyz level-0, f = 2**-level), fused: the offset
        points are never materialised.  Bit-identical to sample_points(p + o * nrm) per layer."""
        q = np.ascontiguousarray(p, dtype=np.float64)
        n = np.ascontiguousarray(nrm, dtype=np.float64)
        o = np.ascontiguousarray(offs, dtype=np.float64)
        out = np.empty((len(o), len(q)), np.float32)
        if len(q) == 0:
            return out
        with self._plock:
            self._sample_into(q, n, o, f, 1, out, True)
        return out

    def _sample_into(self, q, n, o, f, order, out, top):
        """Sample q (and layers o along n, if given) into out.  If the chunks needed do not fit the pool,
        the points are ordered spatially ONCE (by chunk row in z, then y) and the batch is halved by index
        recursively, so each piece is coherent; any split gives the same values, only reuse differs."""
        from . import _kernels as K

        if n is None:
            fn = K.nearest_chunked if order == 0 else K.trilinear_chunked
            ok = self._run(fn, (q, f, out), (q, _NO_NRM, _NO_OFFS, f, order), optimistic=top, work=len(q))
        else:
            ok = self._run(K.trilinear_chunked_layers, (q, n, o, f, out), (q, n, o, f, 1), optimistic=top,
                           work=len(q) * len(o))
        if ok:
            return
        if len(q) == 1:  # one point whose layers span more chunks than the pool: layer by layer
            for i, oo in enumerate(o):
                out[i] = self.sample_points(q + oo * n, f, 1)
            return
        if top:  # sort once; the halves below stay sorted, so no piece is sorted again
            srt = _split_order(q, f, self.chunks[0])
            q, n = q[srt], (None if n is None else n[srt])
            res = np.empty_like(out)
        else:
            res = out
        h = len(q) // 2
        for sl in (slice(0, h), slice(h, None)):
            sub = res[sl] if n is None else res[:, sl]
            dst = np.empty_like(sub)
            self._sample_into(np.ascontiguousarray(q[sl]), None if n is None else np.ascontiguousarray(n[sl]),
                              o, f, order, dst, False)
            sub[...] = dst
        if top:
            if n is None:
                out[srt] = res
            else:
                out[:, srt] = res

    def prefetch(self, keys, workers: int | None = None):
        """Fetch (compressed) chunks in parallel; decoding happens lazily on first use."""
        with self._lock:
            keys = [k for k in keys if k not in self._cache and k not in self._raw]
            waiting = [self._inflight[k] for k in keys if k in self._inflight]
            keys = [k for k in keys if k not in self._inflight]
        for fut in waiting:  # already being fetched in the background: do not request it twice
            fut.result()
        if not keys:
            return
        if not self.store.remote or len(keys) == 1:
            for k in keys:
                self._raw_get(k)
            return
        list(_executor("io", IO_WORKERS).map(self._raw_get, keys))

    def prefetch_async(self, keys):
        """Start fetching chunks in the background (remote stores) and return immediately; used to
        overlap network latency with compute (readahead one ring of chunks beyond a growing front)."""
        if not self.store.remote:
            return []
        ex = _executor("io", IO_WORKERS)
        futs = []
        with self._lock:
            for k in keys:
                if k in self._raw or k in self._inflight:
                    continue

                def job(k=k):
                    try:
                        return self._raw_get(k)
                    finally:
                        with self._lock:
                            self._inflight.pop(k, None)
                self._inflight[k] = f = ex.submit(job)
                futs.append(f)
        return futs

    def _readahead(self, cells, limit: int = 64):
        """Background-fetch the face neighbours of chunks that just had to be loaded: a trace or render
        front that entered these chunks is likely to enter their neighbours next."""
        grid = np.array(self._table.shape)
        c = np.stack(np.unravel_index(cells, self._table.shape), 1)
        nb_ = (c[:, None, :] + np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])).reshape(-1, 3)
        nb_ = nb_[np.all((nb_ >= 0) & (nb_ < grid), 1)]
        flat = np.unique(np.ravel_multi_index(nb_.T, self._table.shape))
        flat = flat[self._table.reshape(-1)[flat] == -1][:limit]
        if len(flat):
            self.prefetch_async(list(zip(*(x.tolist() for x in np.unravel_index(flat, self._table.shape)))))

    def gather(self, zyx: np.ndarray) -> np.ndarray:
        """Exact voxel values at integer (z, y, x) indices (N,3). Out of bounds -> 0."""
        zyx = np.asarray(zyx, dtype=np.int64)
        out = np.zeros(len(zyx), dtype=np.float32)
        inb = np.all((zyx >= 0) & (zyx < np.array(self.shape)), axis=1)
        idx = np.nonzero(inb)[0]
        if idx.size == 0:
            return out
        p = zyx[idx]
        ck = p // np.array(self.chunks)
        # sort by chunk so each chunk is touched once
        flat = (ck[:, 0] * 1_000_003 + ck[:, 1]) * 1_000_003 + ck[:, 2]
        order = np.argsort(flat, kind="stable")
        flat_s = flat[order]
        starts = np.r_[0, np.nonzero(np.diff(flat_s))[0] + 1]
        keys = [tuple(int(v) for v in ck[order[s]]) for s in starts]
        self.prefetch(keys)
        ends = np.r_[starts[1:], len(order)]
        for key, s, e in zip(keys, starts, ends):
            sel = order[s:e]
            c = self.chunk(key)
            local = p[sel] - np.array(key) * np.array(self.chunks)
            out[idx[sel]] = c[local[:, 0], local[:, 1], local[:, 2]]
        return out

    def read(self, z0, z1, y0, y1, x0, x1) -> np.ndarray:
        """Dense read of a ZYX box (half-open), zero padded outside the array."""
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), self.dtype)
        cz, cy, cx = self.chunks
        rz, ry, rx = (range(max(a, 0) // c, (min(b, n) - 1) // c + 1)
                      for a, b, c, n in ((z0, z1, cz, self.shape[0]), (y0, y1, cy, self.shape[1]), (x0, x1, cx, self.shape[2])))
        self.prefetch([(kz, ky, kx) for kz in rz for ky in ry for kx in rx])  # parallel fetch, then copy
        for kz in range(max(z0, 0) // cz, (min(z1, self.shape[0]) - 1) // cz + 1):
            for ky in range(max(y0, 0) // cy, (min(y1, self.shape[1]) - 1) // cy + 1):
                for kx in range(max(x0, 0) // cx, (min(x1, self.shape[2]) - 1) // cx + 1):
                    c = self.chunk((kz, ky, kx))
                    a = [max(z0, kz * cz), max(y0, ky * cy), max(x0, kx * cx)]
                    b = [min(z1, (kz + 1) * cz, self.shape[0]), min(y1, (ky + 1) * cy, self.shape[1]),
                         min(x1, (kx + 1) * cx, self.shape[2])]
                    if any(bb <= aa for aa, bb in zip(a, b)):
                        continue
                    out[a[0] - z0:b[0] - z0, a[1] - y0:b[1] - y0, a[2] - x0:b[2] - x0] = c[
                        a[0] - kz * cz:b[0] - kz * cz, a[1] - ky * cy:b[1] - ky * cy, a[2] - kx * cx:b[2] - kx * cx]
        return out


class Volume:
    """A scroll volume.  ``Volume(path_or_url)`` accepts an OME-Zarr group or a bare zarr array."""

    def __init__(self, path: str, cache_gb: float = 4.0, disk_cache: str | None = None,
                 disk_cache_gb: float = 20.0, decoded_cache_gb: float = 0.5, readahead: bool | None = None):
        """``cache_gb``: RAM for compressed chunks; ``decoded_cache_gb``: RAM for decoded chunks;
        ``disk_cache``: directory for a persistent compressed chunk cache (remote volumes only);
        ``readahead``: background-fetch the neighbours of newly entered chunks (remote only; off by default:
        on Scroll 1 it cut a cold 40-generation trace from 2.2-2.6 s to 1.5 s at one seed, was neutral at
        another, and fetched ~3x the bytes; it changes only when bytes arrive, never sampled values)."""
        self.store = _Store(path, disk_cache, disk_cache_gb)
        self.path = path
        self.voxel_size = None
        attrs = self.store.get(".zattrs")
        paths = []
        if attrs is not None:
            a = json.loads(attrs)
            ms = a.get("multiscales")
            if ms:
                paths = [d["path"] for d in ms[0]["datasets"]]
        if not paths:
            if self.store.get(".zarray") is not None:
                paths = [""]
            else:
                paths = [str(i) for i in range(16) if self.store.get(f"{i}/.zarray") is not None]
        if not paths:
            raise FileNotFoundError(f"{path} is not a zarr array or OME-Zarr group")
        meta = self.store.get("meta.json")
        if meta is not None:
            self.voxel_size = json.loads(meta).get("voxelsize")
        cache = int(cache_gb * 2**30)
        self.levels = [Level(self.store, p.strip("/"), cache, int(decoded_cache_gb * 2**30)) for p in paths]
        for l in self.levels:
            l.readahead = bool(readahead) and self.store.remote

    def stats(self) -> dict:
        """How lean the run was: bytes actually transferred vs. the size of the full volume."""
        full = sum(int(np.prod(l.shape)) * l.dtype.itemsize for l in self.levels[:1])
        touched = sum(len(l._raw) for l in self.levels)
        return {"bytes_fetched": self.store.bytes_fetched, "http_requests": self.store.requests,
                "chunks_touched": touched, "level0_uncompressed_bytes": full,
                "fetched_fraction_of_level0": self.store.bytes_fetched / full if full else None}

    @property
    def shape(self):
        return self.levels[0].shape

    def sample(self, xyz: np.ndarray, level: int = 0, order: int = 1) -> np.ndarray:
        """Sample at (..., 3) xyz positions given in level-0 voxels.

        order=1 is trilinear (VC3D default), order=0 nearest. NaN positions -> NaN.  Same mapping as
        vc_render_tifxyz (p * 2**-g); for mean-pooled pyramids the exact centre mapping would be
        (p - (2**g-1)/2) / 2**g, i.e. a <=0.5 level-0 voxel shift.
        """
        xyz = np.asarray(xyz)
        shp = xyz.shape[:-1]
        out = self.levels[level].sample_points(xyz.reshape(-1, 3), 2.0 ** -level, order=0 if order == 0 else 1)
        return out.reshape(shp)

    def sample_layers(self, xyz: np.ndarray, normals: np.ndarray, offsets, level: int = 0) -> np.ndarray:
        """Trilinear samples at xyz + o * normals for every offset o -> (len(offsets), ...).

        Bit-identical to ``np.stack([self.sample(xyz + o * normals) for o in offsets])`` but fused in one
        compiled pass: the offset positions are never materialised (render's inner loop).
        """
        xyz = np.asarray(xyz)
        shp = xyz.shape[:-1]
        offs = np.asarray(offsets, np.float64).reshape(-1)
        out = self.levels[level].sample_layers(xyz.reshape(-1, 3), np.asarray(normals).reshape(-1, 3), offs,
                                               2.0 ** -level)
        return out.reshape((len(offs),) + shp)


class _ArrayLevel:
    """In-memory stand-in for Level (same gather/read interface)."""

    def __init__(self, a: np.ndarray):
        self.a = a
        self.shape = a.shape
        self.chunks = a.shape
        self.dtype = a.dtype

    def sample_points(self, p, f: float = 1.0, order: int = 1):
        from . import _kernels as K

        q = np.ascontiguousarray(p, dtype=np.float64)
        out = np.empty(len(q), np.float32)
        K.call(K.nearest_dense if order == 0 else K.trilinear_dense, len(q), self.a, q, f, out)
        return out

    def sample_layers(self, p, nrm, offs, f: float = 1.0):
        from . import _kernels as K

        q = np.ascontiguousarray(p, dtype=np.float64)
        out = np.empty((len(offs), len(q)), np.float32)
        K.call(K.trilinear_dense_layers, len(q) * len(offs), self.a, q, np.ascontiguousarray(nrm, dtype=np.float64),
               np.ascontiguousarray(offs, dtype=np.float64), f, out)
        return out

    def gather(self, zyx):
        zyx = np.asarray(zyx, dtype=np.int64)
        out = np.zeros(len(zyx), np.float32)
        inb = np.all((zyx >= 0) & (zyx < np.array(self.shape)), axis=1)
        p = zyx[inb]
        out[inb] = self.a[p[:, 0], p[:, 1], p[:, 2]]
        return out

    def read(self, z0, z1, y0, y1, x0, x1):
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), self.dtype)
        a, sh = self.a, self.shape
        cz0, cy0, cx0 = max(z0, 0), max(y0, 0), max(x0, 0)
        cz1, cy1, cx1 = min(z1, sh[0]), min(y1, sh[1]), min(x1, sh[2])
        if cz1 > cz0 and cy1 > cy0 and cx1 > cx0:
            out[cz0 - z0:cz1 - z0, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = a[cz0:cz1, cy0:cy1, cx0:cx1]
        return out


def mean_pool2(a: np.ndarray) -> np.ndarray:
    """One 2x2x2 mean-pooling step with ``write_ome_zarr``'s rule: crop each axis to even length, average,
    and for integer dtypes round (half to even, numpy's ``round``) and cast back.  Float arrays keep the mean
    (``write_ome_zarr`` rounds them too, which would destroy fields such as a winding coordinate)."""
    a = np.asarray(a)
    z, y, x = (s // 2 * 2 for s in a.shape)
    m = a[:z, :y, :x].reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))
    if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
        m = m.round()
    return m.astype(a.dtype)


class ArrayVolume(Volume):
    """A Volume backed by numpy ZYX arrays (tests, phantoms, benchmarks).

    ``levels``: None = one level (``a``); an int n = n levels, level g+1 = ``mean_pool2(level g)`` (for
    integer data bit-identical to the pyramid ``write_ome_zarr`` writes); a list of arrays = an explicit
    pyramid (``a`` must then be None or ``levels[0]``; each level must have the halved shape s // 2).
    Level g is sampled like a zarr pyramid level: ``sample(xyz, level=g)`` maps L0 xyz by 2**-g."""

    def __init__(self, a: np.ndarray | None = None, voxel_size=None, levels=None):
        self.path = None
        self.voxel_size = voxel_size
        if levels is None or isinstance(levels, (int, np.integer)):
            if a is None:
                raise ValueError("ArrayVolume needs an array (or an explicit list of levels)")
            n = 1 if levels is None else int(levels)
            if n < 1:
                raise ValueError("levels must be >= 1")
            arrs = [np.asarray(a)]
            for _ in range(n - 1):
                if min(arrs[-1].shape) < 2:
                    raise ValueError(f"cannot pool shape {arrs[-1].shape} to {n} levels")
                arrs.append(mean_pool2(arrs[-1]))
        else:
            arrs = [np.asarray(l) for l in levels]
            if not arrs:
                raise ValueError("levels is empty")
            if a is not None and not (a is levels[0] or np.array_equal(np.asarray(a), arrs[0])):
                raise ValueError("a and levels[0] differ")
            for g in range(1, len(arrs)):
                want = tuple(s // 2 for s in arrs[g - 1].shape)
                if arrs[g].shape != want:
                    raise ValueError(f"level {g} has shape {arrs[g].shape}, expected {want} (2x pooling)")
        self.levels = [_ArrayLevel(l) for l in arrs]


def write_ome_zarr(path: str, data: np.ndarray, chunks=(64, 64, 64), levels: int = 1, voxel_size=None):
    """Write a ZYX uint8 array as an OME-Zarr v2 group (blosc/zstd, '/' separator like the scroll data).

    Downsampled levels use 2x2x2 mean pooling.
    """
    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=0)
    os.makedirs(path, exist_ok=True)
    datasets = []
    cur = np.asarray(data)
    for g in range(levels):
        lp = os.path.join(path, str(g))
        os.makedirs(lp, exist_ok=True)
        with open(os.path.join(lp, ".zarray"), "w") as f:
            json.dump({"chunks": list(chunks), "compressor": codec.get_config(), "dimension_separator": "/",
                       "dtype": cur.dtype.str, "fill_value": 0, "filters": None, "order": "C",
                       "shape": list(cur.shape), "zarr_format": 2}, f)
        for kz in range(-(-cur.shape[0] // chunks[0])):
            for ky in range(-(-cur.shape[1] // chunks[1])):
                for kx in range(-(-cur.shape[2] // chunks[2])):
                    blk = np.zeros(chunks, cur.dtype)
                    src = cur[kz * chunks[0]:(kz + 1) * chunks[0], ky * chunks[1]:(ky + 1) * chunks[1],
                              kx * chunks[2]:(kx + 1) * chunks[2]]
                    if not src.any():
                        continue  # absent chunk == fill value
                    blk[:src.shape[0], :src.shape[1], :src.shape[2]] = src
                    d = os.path.join(lp, str(kz), str(ky))
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, str(kx)), "wb") as f:
                        f.write(codec.encode(np.ascontiguousarray(blk)))
        datasets.append({"path": str(g), "coordinateTransformations": [{"type": "scale", "scale": [2.0**g] * 3}]})
        z, y, x = (s // 2 * 2 for s in cur.shape)
        cur = cur[:z, :y, :x].reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5)).round().astype(data.dtype)
    with open(os.path.join(path, ".zgroup"), "w") as f:
        json.dump({"zarr_format": 2}, f)
    with open(os.path.join(path, ".zattrs"), "w") as f:
        json.dump({"multiscales": [{"version": "0.4", "name": "volume", "datasets": datasets,
                                    "axes": [{"name": a, "type": "space"} for a in "zyx"]}]}, f)
    sz = data.shape
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"type": "vol", "format": "zarr", "width": sz[2], "height": sz[1], "slices": sz[0],
                   "voxelsize": voxel_size}, f)
