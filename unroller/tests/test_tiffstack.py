"""TiffVolume: a tiny synthetic uint16 TIFF stack, read from disk and over a local HTTP server."""
import http.server
import os
import re
import threading

import numpy as np
import pytest
import tifffile

from unroller import Surface
from unroller.render import render, render_stream
from unroller.tiffstack import TiffVolume, parse_tiff_header
from unroller.volume import ArrayVolume

SHAPE = (21, 37, 45)  # odd sizes: partial bricks on every face


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    d = tmp_path_factory.mktemp("stack")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 65536, SHAPE, dtype=np.uint16)
    for z in range(SHAPE[0]):  # several strips per slice; one big-endian slice
        tifffile.imwrite(d / f"{z:05d}.tif", a[z].astype(">u2" if z == 7 else "<u2"), rowsperstrip=5,
                         byteorder=">" if z == 7 else "<")
    return d, a


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with single and multipart byte-range support (like nginx)."""
    multipart = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or os.path.isdir(path) or not os.path.exists(path):
            return super().do_GET()
        data = open(path, "rb").read()
        spans = [(int(a), int(b)) for a, b in re.findall(r"(\d+)-(\d+)", rng)]
        if len(spans) == 1 or not self.multipart:
            if len(spans) > 1:  # a server without multipart support: whole file
                return super().do_GET()
            a, b = spans[0]
            body, ctype = data[a:b + 1], "application/octet-stream"
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            bd = "XYZBOUNDARY"
            body = b"".join(f"\r\n--{bd}\r\nContent-Type: image/tiff\r\nContent-Range: bytes {a}-{b}/{len(data)}"
                            f"\r\n\r\n".encode() + data[a:b + 1] for a, b in spans) + f"\r\n--{bd}--\r\n".encode()
            ctype = f"multipart/byteranges; boundary={bd}"
            self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module", params=[True, False], ids=["multipart", "no-range-support"])
def server(stack, request):
    d, _ = stack
    h = type("H", (_RangeHandler,), {"multipart": request.param})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: h(*a, directory=str(d), **k))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_parse_header_multistrip(stack):
    d, a = stack
    raw = open(d / "00003.tif", "rb").read()
    h = parse_tiff_header(lambda o, n: raw[o:o + n], size_hint=64)
    assert (h["height"], h["width"], h["bits"], h["rows_per_strip"]) == (37, 45, 16, 5)
    assert len(h["strip_offsets"]) == 8


def _check(V, a):
    assert V.shape == a.shape and V.dtype == np.uint16
    assert np.array_equal(V.levels[0].read(0, a.shape[0], 0, a.shape[1], 0, a.shape[2]), a)
    rng = np.random.default_rng(1)
    zyx = np.stack([rng.integers(0, s, 500) for s in a.shape], 1)
    assert np.array_equal(V.levels[0].gather(zyx), a[zyx[:, 0], zyx[:, 1], zyx[:, 2]].astype(np.float32))
    p = rng.uniform([0, 0, 0], [a.shape[2] - 1, a.shape[1] - 1, a.shape[0] - 1], (2000, 3))
    ref = ArrayVolume(a).sample(p)
    assert np.allclose(V.sample(p), ref, rtol=0, atol=1e-2)
    # level 1: 2x2x2 mean, rounded (from bricks of level 0)
    sub = a[:20, :36, :44].astype(np.float64).reshape(10, 2, 18, 2, 22, 2).mean((1, 3, 5))
    assert np.array_equal(V.levels[1].read(0, 10, 0, 18, 0, 22), np.rint(sub).astype(np.uint16))


def test_local_stack_is_exact(stack, tmp_path):
    d, a = stack
    _check(TiffVolume(str(d), brick=16, levels=2, disk_cache=str(tmp_path / "c")), a)


def test_http_stack_is_exact_and_cached(stack, server, tmp_path):
    d, a = stack
    V = TiffVolume(server, brick=16, levels=2, disk_cache=str(tmp_path / "c"))
    _check(V, a)
    assert V.stats()["http_requests"] > 0
    W = TiffVolume(server, brick=16, levels=2, disk_cache=str(tmp_path / "c"))  # warm disk cache: no slice reads
    n0 = W.store.requests
    assert np.array_equal(W.levels[0].read(0, a.shape[0], 0, a.shape[1], 0, a.shape[2]), a)
    assert W.store.requests == n0


def test_render_stream_writes_uint16(stack, tmp_path):
    d, a = stack
    V = TiffVolume(str(d), brick=16, levels=1)
    yy, xx = np.mgrid[2:30, 3:40].astype(float)
    pts = np.stack([xx, yy, np.full_like(xx, 10.25)], -1)
    s = Surface(pts, (1.0, 1.0))
    render_stream(s, V, str(tmp_path / "r.zarr"), n_layers=3, tile=16, workers=2)
    import json

    import numcodecs

    m = json.load(open(tmp_path / "r.zarr" / ".zarray"))
    assert m["dtype"] == "<u2"
    blk = np.frombuffer(numcodecs.get_codec(m["compressor"]).decode(open(tmp_path / "r.zarr" / "0" / "0" / "0", "rb").read()),
                        np.uint16).reshape(3, 16, 16)
    ref = render(s, ArrayVolume(a), 3)
    assert np.array_equal(blk[:, :16, :16], np.clip(np.rint(ref[:, :16, :16]), 0, 65535).astype(np.uint16))


def test_http_column_pieces_are_exact(stack, server, tmp_path):
    """Narrow reads (needed columns of each row as multipart pieces, no x padding) give the same voxels."""
    d, a = stack
    V = TiffVolume(server, brick=16, levels=2)
    V.store.FULL_FRAC, V.store.PAD = 2.0, 0
    _check(V, a)


@pytest.fixture(scope="module")
def grid(tmp_path_factory, stack):
    """The same volume as a cell grid (cell 10, edge cells zero-padded like volume_grids) + meta.json."""
    import json

    root = tmp_path_factory.mktemp("volpkg")
    d = root / "volume_grids" / "u1"
    d.mkdir(parents=True)
    (root / "volumes" / "u1").mkdir(parents=True)
    _, a = stack
    C = 10
    (root / "volumes" / "u1" / "meta.json").write_text(json.dumps({"slices": SHAPE[0], "height": SHAPE[1],
                                                                   "width": SHAPE[2], "voxelsize": 7.91}))
    for iz in range(-(-SHAPE[0] // C)):
        for iy in range(-(-SHAPE[1] // C)):
            for ix in range(-(-SHAPE[2] // C)):
                cell = np.zeros((C, C, C), np.uint16)
                src = a[iz * C:(iz + 1) * C, iy * C:(iy + 1) * C, ix * C:(ix + 1) * C]
                cell[:src.shape[0], :src.shape[1], :src.shape[2]] = src
                with tifffile.TiffWriter(d / f"cell_yxz_{iy + 1:03d}_{ix + 1:03d}_{iz + 1:03d}.tif") as w:
                    for page in cell:
                        w.write(page, contiguous=False)
    return d, a


def test_cell_grid_is_exact(grid, tmp_path):
    from unroller.tiffstack import CellGridStore, CellGridVolume

    d, a = grid
    V = CellGridVolume(str(d), chunk=(5, 5), levels=1, disk_cache=str(tmp_path / "c"))
    assert V.shape == a.shape and V.voxel_size == 7.91
    assert np.array_equal(V.levels[0].read(0, a.shape[0], 0, a.shape[1], 0, a.shape[2]), a)
    rng = np.random.default_rng(3)
    p = rng.uniform([0, 0, 0], [a.shape[2] - 1, a.shape[1] - 1, a.shape[0] - 1], (1000, 3))
    assert np.allclose(V.sample(p), ArrayVolume(a).sample(p), rtol=0, atol=1e-2)
    assert CellGridStore.cell_name(1, 2, 3) == "cell_yxz_001_002_003.tif"


def test_cell_grid_over_http(grid):
    from unroller.tiffstack import CellGridVolume

    d, a = grid
    root = d.parent.parent
    h = type("H", (_RangeHandler,), {"multipart": True})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *x, **k: h(*x, directory=str(root), **k))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        V = CellGridVolume(f"http://127.0.0.1:{srv.server_address[1]}/volume_grids/u1", chunk=(5, 5), levels=1)
        assert V.shape == a.shape
        assert np.array_equal(V.levels[0].read(0, a.shape[0], 0, a.shape[1], 0, a.shape[2]), a)
        n_chunks = len(V.levels[0]._raw)
        n_cells = 3 * 4 * 5
        assert V.store.requests <= 6 * n_cells + n_chunks + 2  # per cell: header, IFD reads, size; per chunk: 1
    finally:
        srv.shutdown()


def test_cell_grid_missing_cell_raises(grid, tmp_path):
    # a cell missing from the grid (or a 404 for it) must raise, never read as fill value 0 (the community
    # uint16 zarr reads its absent chunks as 0)
    import shutil

    from unroller.tiffstack import CellGridVolume

    d, a = grid
    root = tmp_path / "volume_grids" / "u1"
    shutil.copytree(d, root)
    if (d.parent.parent / "volumes").exists():
        shutil.copytree(d.parent.parent / "volumes", tmp_path / "volumes")
    names = sorted(p.name for p in root.glob("cell_yxz_*.tif"))
    (root / names[len(names) // 2]).unlink()
    h = type("H", (_RangeHandler,), {"multipart": True})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *x, **k: h(*x, directory=str(tmp_path), **k))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for path in (str(root), f"http://127.0.0.1:{srv.server_address[1]}/volume_grids/u1"):
            V = CellGridVolume(path, chunk=(5, 5), levels=1, shape=a.shape)
            with pytest.raises(FileNotFoundError):
                V.levels[0].read(0, a.shape[0], 0, a.shape[1], 0, a.shape[2])
    finally:
        srv.shutdown()


def test_staged_bricks_stay_within_ram_budget(stack):
    # bricks built speculatively (neighbours, full rows) that are never asked for must not pile up in RAM
    from unroller.tiffstack import TiffVolume

    d, a = stack
    V = TiffVolume(str(d), cache_gb=4 * 8**3 * 2 / 2**30, decoded_cache_gb=1e-5, brick=8, levels=1)
    rng = np.random.default_rng(0)
    Z, Y, X = a.shape
    for _ in range(60):
        c = rng.uniform([1, 1, 1], [X - 1, Y - 1, Z - 1])
        V.sample(c[None] + rng.normal(scale=0.5, size=(20, 3)))
    assert V.store._staged_bytes <= V.store.staged_max + 2 * 8**3 * 2
    assert V.store._staged_bytes == sum(len(v) for v in V.store._staged.values())


class _FlakyHandler(_RangeHandler):
    """Like nginx behind a draining relay: every 3rd GET (and the first one) is cut off mid-body or reset."""
    lock = threading.Lock()
    n = 0
    dropped = 0
    truncated = 0

    def do_GET(self):
        cls = type(self)
        with cls.lock:
            cls.n += 1
            drop = cls.n % 3 == 1
            mode = cls.n % 2
            if drop:
                cls.dropped += 1
                cls.truncated += mode
        if not drop:
            return super().do_GET()
        if mode:  # headers announce more bytes than are sent: ChunkedEncodingError / IncompleteRead
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", "100000")
            self.end_headers()
            self.wfile.write(b"\0" * 100)
        self.wfile.flush()
        self.close_connection = True
        try:  # connection reset (RST, not FIN)
            import socket
            import struct

            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        except OSError:
            pass
        self.connection.close()


def test_http_stack_retries_dropped_transfers(stack, monkeypatch):
    """TiffStackStore must survive bodies cut off mid-transfer and connection resets (retry with exponential
    backoff, ``UNROLLER_HTTP_RETRIES`` tries) and still return exact data."""
    from unroller import volume

    d, a = stack
    h = type("Flaky", (_FlakyHandler,), {"multipart": True, "n": 0, "dropped": 0, "truncated": 0, "lock": threading.Lock()})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *x, **k: h(*x, directory=str(d), **k))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cls = volume._retry_session_cls()
    monkeypatch.setattr(cls, "backoff", 0.001)
    sleeps = []
    import time as _time

    real_sleep = _time.sleep
    monkeypatch.setattr(_time, "sleep", lambda s: (sleeps.append(s), real_sleep(min(s, 0.01))))
    try:
        V = TiffVolume(f"http://127.0.0.1:{srv.server_address[1]}", brick=16, levels=1)
        assert np.array_equal(V.levels[0].read(0, a.shape[0], 0, a.shape[1], 0, a.shape[2]), a)
    finally:
        srv.shutdown()
    # resets before the response are retried by urllib3 too; a body cut off mid-transfer only by the session
    assert h.truncated > 0 and len(sleeps) >= h.truncated
    # exponential: the k-th retry of one request waits backoff * 2**k * [0.5, 1.5)
    assert max(sleeps) < 0.001 * 2 ** volume.HTTP_RETRIES * 1.5
