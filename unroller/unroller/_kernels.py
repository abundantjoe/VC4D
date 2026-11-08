"""Compiled (numba) kernels for the hot loops.  Semantics match the numpy reference exactly:
trilinear interpolation, corners outside the array read 0, NaN positions give NaN.

Points are passed as (x, y, z) in level-0 voxels together with f = 2**-level; the kernels scale and
reorder inline, so callers never build temporary coordinate arrays.
"""
from __future__ import annotations

import os

import numba as nb
import numpy as np

# Parallel kernels pay a fixed cost per launch (waking the thread pool); when other processes already
# load every core that cost explodes (measured: 100-point trilinear call 8 ms with 4 threads under
# contention vs 0.003 ms on 1 thread).  Calls with fewer than SMALL_WORK elements therefore run on the
# calling thread only.  Results do not depend on the thread count (per-element work, integer or max
# reductions only).
SMALL_WORK = 65536


def call(kernel, work, *args):
    """kernel(*args), on one thread if ``work`` (elements processed) is small."""
    if work >= SMALL_WORK:
        return kernel(*args)
    prev = nb.get_num_threads()
    if prev == 1:
        return kernel(*args)
    nb.set_num_threads(1)  # per calling thread in numba, so concurrent render threads are unaffected
    try:
        return kernel(*args)
    finally:
        nb.set_num_threads(prev)


def limit_threads(workers: int):
    """Cap numba threads of this process to its share of the cores (cpu_count // workers); use as the
    initializer of a ProcessPoolExecutor running ``workers`` processes, so they do not oversubscribe."""
    nb.set_num_threads(max(1, min(nb.config.NUMBA_NUM_THREADS, (os.cpu_count() or 1) // max(workers, 1))))


@nb.njit(parallel=True, fastmath=False, cache=True)
def trilinear_dense(a, p, f, out):
    Z, Y, X = a.shape
    for i in nb.prange(p.shape[0]):
        z, y, x = p[i, 2] * f, p[i, 1] * f, p[i, 0] * f
        if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
            out[i] = np.nan
            continue
        z0 = int(np.floor(z))
        y0 = int(np.floor(y))
        x0 = int(np.floor(x))
        tz = np.float32(z - z0)
        ty = np.float32(y - y0)
        tx = np.float32(x - x0)
        acc = np.float32(0.0)
        for dz in range(2):
            zi = z0 + dz
            wz = tz if dz else np.float32(1.0) - tz
            if zi < 0 or zi >= Z:
                continue
            for dy in range(2):
                yi = y0 + dy
                wy = ty if dy else np.float32(1.0) - ty
                if yi < 0 or yi >= Y:
                    continue
                for dx in range(2):
                    xi = x0 + dx
                    wx = tx if dx else np.float32(1.0) - tx
                    if xi < 0 or xi >= X:
                        continue
                    acc += wz * wy * wx * np.float32(a[zi, yi, xi])
        out[i] = acc


@nb.njit(parallel=True, cache=True)
def nearest_dense(a, p, f, out):
    Z, Y, X = a.shape
    for i in nb.prange(p.shape[0]):
        z, y, x = p[i, 2] * f, p[i, 1] * f, p[i, 0] * f
        if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
            out[i] = np.nan
            continue
        zi = int(np.rint(z))
        yi = int(np.rint(y))
        xi = int(np.rint(x))
        if 0 <= zi < Z and 0 <= yi < Y and 0 <= xi < X:
            out[i] = a[zi, yi, xi]
        else:
            out[i] = 0.0


@nb.njit(cache=True, inline="always")
def _cell_range(z, y, x, order, Z, Y, X, cz, cy, cx):
    """Chunk-grid cell range [k*a, k*b] (inclusive) holding the in-bounds interpolation corners of
    one point (order 1: the 8 trilinear corners, order 0: the nearest voxel).  kza > kzb if none."""
    if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
        return 1, 0, 0, 0, 0, 0
    if order == 0:
        z0, y0, x0 = int(np.rint(z)), int(np.rint(y)), int(np.rint(x))
        r = 0
    else:
        z0, y0, x0 = int(np.floor(z)), int(np.floor(y)), int(np.floor(x))
        r = 1
    za, zb = max(z0, 0), min(z0 + r, Z - 1)
    ya, yb = max(y0, 0), min(y0 + r, Y - 1)
    xa, xb = max(x0, 0), min(x0 + r, X - 1)
    if za > zb or ya > yb or xa > xb:
        return 1, 0, 0, 0, 0, 0
    kz, ky, kx = za // cz, ya // cy, xa // cx  # corners span <= 2 voxels, so at most one chunk further
    return (kz, kz + (zb - kz * cz >= cz), ky, ky + (yb - ky * cy >= cy), kx, kx + (xb - kx * cx >= cx))


_BLOCK = 16384


@nb.njit(parallel=True, cache=True)
def mark_cells(mark, cz, cy, cx, shape, p, nrm, offs, f, order):
    """Set mark = 1 on every chunk-grid cell needed to sample the points (all in-bounds trilinear
    corners, exactly the cells the sampling kernels read).  Points are p + offs[l] * nrm (nrm may be
    empty -> offs ignored, points are p).  Parallel over blocks of points; concurrent writers only ever
    store the same value 1, and a cell is read before it is written so hot cells are not contended.
    Returns per-block bounding boxes (nblk, 6) of the touched cells (lo > hi: none), so that the
    collection pass only scans the touched part of the grid."""
    Z, Y, X = shape
    n = p.shape[0]
    L = offs.shape[0] if nrm.shape[0] > 0 else 1
    nblk = (n + _BLOCK - 1) // _BLOCK
    bb = np.empty((nblk, 6), np.int64)
    for b in nb.prange(nblk):
        lz, ly, lx = 1 << 40, 1 << 40, 1 << 40
        hz, hy, hx = -1, -1, -1
        for i in range(b * _BLOCK, min(n, (b + 1) * _BLOCK)):
            for l in range(L):
                if nrm.shape[0] > 0:
                    o = offs[l]
                    x = (p[i, 0] + o * nrm[i, 0]) * f
                    y = (p[i, 1] + o * nrm[i, 1]) * f
                    z = (p[i, 2] + o * nrm[i, 2]) * f
                else:
                    z, y, x = p[i, 2] * f, p[i, 1] * f, p[i, 0] * f
                kza, kzb, kya, kyb, kxa, kxb = _cell_range(z, y, x, order, Z, Y, X, cz, cy, cx)
                if kza > kzb:
                    continue
                for kz in range(kza, kzb + 1):
                    for ky in range(kya, kyb + 1):
                        for kx in range(kxa, kxb + 1):
                            if mark[kz, ky, kx] == 0:
                                mark[kz, ky, kx] = 1
                lz, hz = min(lz, kza), max(hz, kzb)
                ly, hy = min(ly, kya), max(hy, kyb)
                lx, hx = min(lx, kxa), max(hx, kxb)
        bb[b, 0], bb[b, 1], bb[b, 2], bb[b, 3], bb[b, 4], bb[b, 5] = lz, hz, ly, hy, lx, hx
    return bb


@nb.njit(cache=True)
def collect_cells(mark, bb):
    """Flat (C-order) indices of the marked cells inside the union of the boxes ``bb``; resets them
    to 0 so the mark grid is clean for the next call (it is allocated once per level, not per call)."""
    lz, ly, lx = 1 << 40, 1 << 40, 1 << 40
    hz, hy, hx = -1, -1, -1
    for b in range(bb.shape[0]):
        if bb[b, 0] > bb[b, 1]:
            continue
        lz, hz = min(lz, bb[b, 0]), max(hz, bb[b, 1])
        ly, hy = min(ly, bb[b, 2]), max(hy, bb[b, 3])
        lx, hx = min(lx, bb[b, 4]), max(hx, bb[b, 5])
    if hz < 0:
        return np.empty(0, np.int64)
    _, GY, GX = mark.shape
    out = np.empty((hz - lz + 1) * (hy - ly + 1) * (hx - lx + 1), np.int64)
    n = 0
    for kz in range(lz, hz + 1):
        for ky in range(ly, hy + 1):
            for kx in range(lx, hx + 1):
                if mark[kz, ky, kx]:
                    mark[kz, ky, kx] = 0
                    out[n] = (kz * GY + ky) * GX + kx
                    n += 1
    return out[:n]


@nb.njit(cache=True, inline="always")
def _touch(stamp, s, epoch):
    if stamp[s] != epoch:  # read first: hot slots stay shared in every core's cache
        stamp[s] = epoch


@nb.njit(cache=True, inline="always")
def _tri_chunked(pool, table, fill, cz, cy, cx, Z, Y, X, z, y, x, stamp, epoch):
    """Trilinear sample from the chunk pool -> (value, miss).  miss = 1 if a corner lies in a chunk that
    is not resident (table == -1); the value is then incomplete and the caller loads and resamples.
    Every pool slot read gets stamp[slot] = epoch (recency for LRU eviction)."""
    if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
        return np.float32(np.nan), 0
    z0 = int(np.floor(z))
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    tz = np.float32(z - z0)
    ty = np.float32(y - y0)
    tx = np.float32(x - x0)
    if z0 >= 0 and y0 >= 0 and x0 >= 0 and z0 + 1 < Z and y0 + 1 < Y and x0 + 1 < X:
        kz, ky, kx = z0 // cz, y0 // cy, x0 // cx
        az, ay, ax = z0 - kz * cz, y0 - ky * cy, x0 - kx * cx
        if az + 1 < cz and ay + 1 < cy and ax + 1 < cx:
            s = table[kz, ky, kx]
            if s >= 0:
                # fast path, all 8 corners in one resident chunk: same products in the same order as the
                # general loop below, so the result is bit-identical
                _touch(stamp, s, epoch)
                wz0, wy0, wx0 = np.float32(1.0) - tz, np.float32(1.0) - ty, np.float32(1.0) - tx
                acc = np.float32(0.0)
                acc += wz0 * wy0 * wx0 * np.float32(pool[s, az, ay, ax])
                acc += wz0 * wy0 * tx * np.float32(pool[s, az, ay, ax + 1])
                acc += wz0 * ty * wx0 * np.float32(pool[s, az, ay + 1, ax])
                acc += wz0 * ty * tx * np.float32(pool[s, az, ay + 1, ax + 1])
                acc += tz * wy0 * wx0 * np.float32(pool[s, az + 1, ay, ax])
                acc += tz * wy0 * tx * np.float32(pool[s, az + 1, ay, ax + 1])
                acc += tz * ty * wx0 * np.float32(pool[s, az + 1, ay + 1, ax])
                acc += tz * ty * tx * np.float32(pool[s, az + 1, ay + 1, ax + 1])
                return acc, 0
    acc = np.float32(0.0)
    miss = 0
    for dz in range(2):
        zi = z0 + dz
        wz = tz if dz else np.float32(1.0) - tz
        if zi < 0 or zi >= Z:
            continue
        for dy in range(2):
            yi = y0 + dy
            wy = ty if dy else np.float32(1.0) - ty
            if yi < 0 or yi >= Y:
                continue
            for dx in range(2):
                xi = x0 + dx
                wx = tx if dx else np.float32(1.0) - tx
                if xi < 0 or xi >= X:
                    continue
                s = table[zi // cz, yi // cy, xi // cx]
                if s >= 0:
                    _touch(stamp, s, epoch)
                    v = np.float32(pool[s, zi % cz, yi % cy, xi % cx])
                else:
                    miss |= s == -1
                    v = np.float32(fill)
                acc += wz * wy * wx * v
    return acc, miss


@nb.njit(parallel=True, cache=True)
def trilinear_chunked(pool, table, fill, cz, cy, cx, shape, p, f, out, stamp, epoch):
    """Trilinear sampling straight from a pool of decoded chunks; table maps chunk-grid cell ->
    pool slot (>= 0), -2 = chunk absent from the store (fill value), -1 = not resident.
    Returns the number of points that touched a non-resident chunk (their output is invalid)."""
    Z, Y, X = shape
    misses = 0
    for i in nb.prange(p.shape[0]):
        v, m = _tri_chunked(pool, table, fill, cz, cy, cx, Z, Y, X, p[i, 2] * f, p[i, 1] * f, p[i, 0] * f,
                            stamp, epoch)
        out[i] = v
        misses += m
    return misses


@nb.njit(parallel=True, cache=True)
def trilinear_chunked_layers(pool, table, fill, cz, cy, cx, shape, p, nrm, offs, f, out, stamp, epoch):
    """out[l, i] = trilinear sample at (p[i] + offs[l] * nrm[i]) * f -- the same float64 arithmetic as
    sampling the array p + o * nrm, without materialising it (render's inner loop).  Returns misses."""
    Z, Y, X = shape
    misses = 0
    for i in nb.prange(p.shape[0]):
        for l in range(offs.shape[0]):
            o = offs[l]
            v, m = _tri_chunked(pool, table, fill, cz, cy, cx, Z, Y, X, (p[i, 2] + o * nrm[i, 2]) * f,
                                (p[i, 1] + o * nrm[i, 1]) * f, (p[i, 0] + o * nrm[i, 0]) * f, stamp, epoch)
            out[l, i] = v
            misses += m
    return misses


@nb.njit(parallel=True, cache=True)
def trilinear_dense_layers(a, p, nrm, offs, f, out):
    """Dense-array counterpart of trilinear_chunked_layers (bit-identical to trilinear_dense)."""
    Z, Y, X = a.shape
    for i in nb.prange(p.shape[0]):
        for l in range(offs.shape[0]):
            o = offs[l]
            z = (p[i, 2] + o * nrm[i, 2]) * f
            y = (p[i, 1] + o * nrm[i, 1]) * f
            x = (p[i, 0] + o * nrm[i, 0]) * f
            if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
                out[l, i] = np.nan
                continue
            z0 = int(np.floor(z))
            y0 = int(np.floor(y))
            x0 = int(np.floor(x))
            tz = np.float32(z - z0)
            ty = np.float32(y - y0)
            tx = np.float32(x - x0)
            acc = np.float32(0.0)
            for dz in range(2):
                zi = z0 + dz
                wz = tz if dz else np.float32(1.0) - tz
                if zi < 0 or zi >= Z:
                    continue
                for dy in range(2):
                    yi = y0 + dy
                    wy = ty if dy else np.float32(1.0) - ty
                    if yi < 0 or yi >= Y:
                        continue
                    for dx in range(2):
                        xi = x0 + dx
                        wx = tx if dx else np.float32(1.0) - tx
                        if xi < 0 or xi >= X:
                            continue
                        acc += wz * wy * wx * np.float32(a[zi, yi, xi])
            out[l, i] = acc


@nb.njit(parallel=True, cache=True)
def nearest_chunked(pool, table, fill, cz, cy, cx, shape, p, f, out, stamp, epoch):
    Z, Y, X = shape
    misses = 0
    for i in nb.prange(p.shape[0]):
        z, y, x = p[i, 2] * f, p[i, 1] * f, p[i, 0] * f
        if not (np.isfinite(z) and np.isfinite(y) and np.isfinite(x)):
            out[i] = np.nan
            continue
        zi, yi, xi = int(np.rint(z)), int(np.rint(y)), int(np.rint(x))
        if 0 <= zi < Z and 0 <= yi < Y and 0 <= xi < X:
            s = table[zi // cz, yi // cy, xi // cx]
            if s >= 0:
                _touch(stamp, s, epoch)
                out[i] = pool[s, zi % cz, yi % cy, xi % cx]
            else:
                misses += s == -1
                out[i] = fill
        else:
            out[i] = 0.0
    return misses


@nb.njit(parallel=True, cache=True, nogil=True)
def arap_rotations(T, cot, ex, uv, th):
    """ARAP local step: th[t] = angle of the best rotation (closed form, 2D) of the cot-weighted
    edge covariance S = sum_j cot_j (u_k - u_l) ex_j^T.  Returns the ARAP energy of ``uv``.
    ex[t, j] is the rest-frame edge (k_j - l_j), with (k_j, l_j) = ((j+1)%3, (j+2)%3).
    """
    energy = 0.0
    for t in nb.prange(T.shape[0]):
        s00 = 0.0
        s01 = 0.0
        s10 = 0.0
        s11 = 0.0
        for j in range(3):
            k = T[t, (j + 1) % 3]
            l = T[t, (j + 2) % 3]
            c = cot[t, j]
            eu0 = uv[k, 0] - uv[l, 0]
            eu1 = uv[k, 1] - uv[l, 1]
            s00 += c * eu0 * ex[t, j, 0]
            s01 += c * eu0 * ex[t, j, 1]
            s10 += c * eu1 * ex[t, j, 0]
            s11 += c * eu1 * ex[t, j, 1]
        a = np.arctan2(s10 - s01, s00 + s11)
        th[t] = a
        cs = np.cos(a)
        sn = np.sin(a)
        e = 0.0
        for j in range(3):
            k = T[t, (j + 1) % 3]
            l = T[t, (j + 2) % 3]
            d0 = uv[k, 0] - uv[l, 0] - (cs * ex[t, j, 0] - sn * ex[t, j, 1])
            d1 = uv[k, 1] - uv[l, 1] - (sn * ex[t, j, 0] + cs * ex[t, j, 1])
            e += cot[t, j] * (d0 * d0 + d1 * d1)
        energy += e
    return energy


@nb.njit(cache=True, nogil=True)
def arap_rhs(T, cot, ex, th, b):
    """ARAP global-step right-hand side for per-triangle rotations th: b[k] += cot R ex, b[l] -= cot R ex."""
    b[:] = 0.0
    for t in range(T.shape[0]):
        cs = np.cos(th[t])
        sn = np.sin(th[t])
        for j in range(3):
            k = T[t, (j + 1) % 3]
            l = T[t, (j + 2) % 3]
            c = cot[t, j]
            r0 = c * (cs * ex[t, j, 0] - sn * ex[t, j, 1])
            r1 = c * (sn * ex[t, j, 0] + cs * ex[t, j, 1])
            b[k, 0] += r0
            b[k, 1] += r1
            b[l, 0] -= r0
            b[l, 1] -= r1


@nb.njit(parallel=True, cache=True, nogil=True)
def gauss_seidel_rows(indptr, indices, data, diag, r, z, rows):
    """Gauss-Seidel update of the vertices ``rows`` of one colour (no two are coupled): parallel."""
    for s in nb.prange(rows.shape[0]):
        i = rows[s]
        a0 = r[i, 0]
        a1 = r[i, 1]
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            a0 -= data[p] * z[j, 0]
            a1 -= data[p] * z[j, 1]
        z[i, 0] += a0 / diag[i]
        z[i, 1] += a1 / diag[i]


@nb.njit(parallel=True, cache=True, nogil=True)
def csr_matvec2(indptr, indices, data, x, out):
    """out = A x for CSR A and two right-hand sides, parallel over rows."""
    for i in nb.prange(indptr.shape[0] - 1):
        a0 = 0.0
        a1 = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            a0 += data[p] * x[j, 0]
            a1 += data[p] * x[j, 1]
        out[i, 0] = a0
        out[i, 1] = a1
    return out


@nb.njit(parallel=True, cache=True, nogil=True)
def dot2(a, b):
    """Column-wise dot products of two (n, 2) arrays."""
    s0 = 0.0
    s1 = 0.0
    for i in nb.prange(a.shape[0]):
        s0 += a[i, 0] * b[i, 0]
        s1 += a[i, 1] * b[i, 1]
    return np.array([s0, s1])


@nb.njit(parallel=True, cache=True, nogil=True)
def axpy2(y, a, x):
    """y += a * x in place, per column (a has 2 entries)."""
    for i in nb.prange(y.shape[0]):
        y[i, 0] += a[0] * x[i, 0]
        y[i, 1] += a[1] * x[i, 1]


@nb.njit(parallel=True, cache=True, nogil=True)
def xpby2(x, b, y):
    """y = x + b * y in place, per column (b has 2 entries)."""
    for i in nb.prange(y.shape[0]):
        y[i, 0] = x[i, 0] + b[0] * y[i, 0]
        y[i, 1] = x[i, 1] + b[1] * y[i, 1]


@nb.njit(cache=True)
def fill_normals(n, passes):
    """Fill NaN normals (H, W, 3) with the normalised mean of valid 8-neighbours, ring by ring."""
    H, W = n.shape[0], n.shape[1]
    for _ in range(passes):
        src = n.copy()
        changed = False
        for r in range(H):
            for c in range(W):
                if np.isfinite(src[r, c, 0]):
                    continue
                a0 = 0.0
                a1 = 0.0
                a2 = 0.0
                cnt = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < H and 0 <= cc < W and np.isfinite(src[rr, cc, 0]):
                            a0 += src[rr, cc, 0]
                            a1 += src[rr, cc, 1]
                            a2 += src[rr, cc, 2]
                            cnt += 1
                if cnt > 0:
                    nn = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                    if nn > 0:
                        n[r, c, 0] = a0 / nn
                        n[r, c, 1] = a1 / nn
                        n[r, c, 2] = a2 / nn
                        changed = True
        if not changed:
            break


@nb.njit(parallel=True, cache=True)
def bilinear2d(img, r, c, out, nearest_edge):
    """Bilinear lookup of a 2D image at (r, c); outside -> edge value (nearest_edge) or 0."""
    H, W = img.shape
    for i in nb.prange(r.shape[0]):
        rr, cc = r[i], c[i]
        if not (np.isfinite(rr) and np.isfinite(cc)):
            out[i] = np.nan
            continue
        if nearest_edge:
            rr = min(max(rr, 0.0), H - 1.0)
            cc = min(max(cc, 0.0), W - 1.0)
        r0 = int(np.floor(rr))
        c0 = int(np.floor(cc))
        tr = rr - r0
        tc = cc - c0
        acc = 0.0
        for dr in range(2):
            ri = r0 + dr
            wr = tr if dr else 1.0 - tr
            if ri < 0 or ri >= H:
                continue
            for dc in range(2):
                ci = c0 + dc
                wc = tc if dc else 1.0 - tc
                if ci < 0 or ci >= W:
                    continue
                acc += wr * wc * img[ri, ci]
        out[i] = acc


@nb.njit(parallel=True, cache=True)
def trilinear_field(F, p, origin, inv_step, out):
    """Multi-channel trilinear lookup of a vector field F (C, Z, Y, X) sampled on a regular grid
    with grid coordinate = (p - origin) * inv_step (xyz -> zyx inside); clamps to the grid edge."""
    C, Z, Y, X = F.shape
    for i in nb.prange(p.shape[0]):
        gx = min(max((p[i, 0] - origin[0]) * inv_step, 0.0), X - 1.000001)
        gy = min(max((p[i, 1] - origin[1]) * inv_step, 0.0), Y - 1.000001)
        gz = min(max((p[i, 2] - origin[2]) * inv_step, 0.0), Z - 1.000001)
        x0 = int(gx)
        y0 = int(gy)
        z0 = int(gz)
        tx, ty, tz = gx - x0, gy - y0, gz - z0
        for ch in range(C):
            acc = 0.0
            for dz in range(2):
                wz = tz if dz else 1.0 - tz
                for dy in range(2):
                    wy = ty if dy else 1.0 - ty
                    for dx in range(2):
                        wx = tx if dx else 1.0 - tx
                        acc += wz * wy * wx * F[ch, z0 + dz, y0 + dy, x0 + dx]
            out[i, ch] = acc


@nb.njit(cache=True)
def rasterize_tris(g, T, V, out, best):
    """Barycentric rasterisation: g (nT, 3, 2) triangle corners in pixel coords (x=col, y=row);
    each pixel takes the 3D position interpolated in the triangle that contains it most deeply."""
    H, W = best.shape
    for t in range(T.shape[0]):
        x0, y0 = g[t, 0, 0], g[t, 0, 1]
        d1x, d1y = g[t, 1, 0] - x0, g[t, 1, 1] - y0
        d2x, d2y = g[t, 2, 0] - x0, g[t, 2, 1] - y0
        den = d1x * d2y - d1y * d2x
        if den == 0.0 or not np.isfinite(den):
            continue
        lox = int(np.ceil(min(x0, x0 + d1x, x0 + d2x) - 1e-9))
        hix = int(np.floor(max(x0, x0 + d1x, x0 + d2x) + 1e-9))
        loy = int(np.ceil(min(y0, y0 + d1y, y0 + d2y) - 1e-9))
        hiy = int(np.floor(max(y0, y0 + d1y, y0 + d2y) + 1e-9))
        for py in range(max(loy, 0), min(hiy, H - 1) + 1):
            for px in range(max(lox, 0), min(hix, W - 1) + 1):
                rx = px - x0
                ry = py - y0
                b1 = (rx * d2y - ry * d2x) / den
                b2 = (d1x * ry - d1y * rx) / den
                b0 = 1.0 - b1 - b2
                m = min(b0, min(b1, b2))
                if m >= -1e-6 and m > best[py, px]:
                    best[py, px] = m
                    for k in range(3):
                        out[py, px, k] = b0 * V[T[t, 0], k] + b1 * V[T[t, 1], k] + b2 * V[T[t, 2], k]


@nb.njit(cache=True, inline="always")
def _d1(a, b, c, has_a, has_c):
    """One component of surface._deriv at b: central difference where both neighbours give a finite
    one-sided difference, else the finite one-sided difference, else NaN (float32 throughout)."""
    nan = np.float32(np.nan)
    fwd = c - b if has_c else nan
    bwd = b - a if has_a else nan
    cen = (fwd + bwd) / np.float32(2.0)
    if np.isfinite(cen):
        return cen
    if np.isfinite(fwd):
        return fwd
    return bwd


@nb.njit(parallel=True, cache=True)
def grid_normals(p, out):
    """Surface.normals() for a float32 (H, W, 3) grid, bit-identical to the numpy reference
    (_deriv along both axes, np.cross, np.linalg.norm, divide; NaN where the length is not > 0)."""
    H, W = p.shape[0], p.shape[1]
    nan = np.float32(np.nan)
    for r in nb.prange(H):
        du = np.empty(3, np.float32)
        dv = np.empty(3, np.float32)
        for c in range(W):
            for k in range(3):
                b = p[r, c, k]
                if not np.isfinite(b):
                    du[k] = nan
                    dv[k] = nan
                    continue
                du[k] = _d1(p[r, c - 1, k] if c > 0 else b, b, p[r, c + 1, k] if c + 1 < W else b, c > 0, c + 1 < W)
                dv[k] = _d1(p[r - 1, c, k] if r > 0 else b, b, p[r + 1, c, k] if r + 1 < H else b, r > 0, r + 1 < H)
            n0 = du[1] * dv[2] - du[2] * dv[1]
            n1 = du[2] * dv[0] - du[0] * dv[2]
            n2 = du[0] * dv[1] - du[1] * dv[0]
            s = n0 * n0
            s += n1 * n1
            s += n2 * n2
            ln = np.sqrt(s)
            if ln > 0:
                out[r, c, 0] = n0 / ln
                out[r, c, 1] = n1 / ln
                out[r, c, 2] = n2 / ln
            else:
                out[r, c, 0] = nan
                out[r, c, 1] = nan
                out[r, c, 2] = nan


# ---------------------------------------------------------------- mean-shift snap (trace._snap_ms)
# numpy's float64 row sums are pairwise (8 accumulators, blocks of <= 128, recursive halving at multiples
# of 8), starting from the identity 0.  _pairwise reproduces that order exactly, so every sum below is
# bit-identical to ndarray.sum(1) on a C-contiguous (N, K) array.  exp() stays in numpy (its SIMD exp
# differs from libm by 1 ulp on ~5% of inputs), so the snap kernels are split around np.exp.

@nb.njit(cache=True)
def _pairwise_block(a, lo, n):
    """numpy's pairwise sum for n <= 128 (leaf of the recursion)."""
    if n < 8:
        res = -0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    i = 8
    m = n - n % 8
    while i < m:
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@nb.njit(cache=True)
def _pairwise(a, lo, n):
    """numpy's pairwise sum: blocks of <= 128 combined as left + right after halving at multiples of 8.
    The recursion is unrolled onto an explicit stack (numba cannot cache self-recursive functions)."""
    if n <= 128:
        return _pairwise_block(a, lo, n)
    # post-order traversal: frames (lo, n, stage); values pushed when a subtree is done
    flo = np.empty(64, np.int64)
    fn = np.empty(64, np.int64)
    fst = np.empty(64, np.int64)
    vals = np.empty(64)
    sp = 0
    vp = 0
    flo[0], fn[0], fst[0] = lo, n, 0
    sp = 1
    while sp > 0:
        t = sp - 1
        cl, cn = flo[t], fn[t]
        if cn <= 128:
            vals[vp] = _pairwise_block(a, cl, cn)
            vp += 1
            sp -= 1
            continue
        n2 = cn // 2
        n2 -= n2 % 8
        if fst[t] == 0:
            fst[t] = 1
            flo[sp], fn[sp], fst[sp] = cl, n2, 0
            sp += 1
        elif fst[t] == 1:
            fst[t] = 2
            flo[sp], fn[sp], fst[sp] = cl + n2, cn - n2, 0
            sp += 1
        else:
            vals[vp - 2] = vals[vp - 2] + vals[vp - 1]
            vp -= 1
            sp -= 1
    return vals[0]


@nb.njit(cache=True, inline="always")
def _rowsum(a):
    return 0.0 + _pairwise(a, 0, a.shape[0])


@nb.njit(cache=True)
def ms_gauss_arg(offs, d, sigma, out):
    """out[i, k] = -0.5 * ((offs[k] - d[i]) / sigma) ** 2 (the exponent of the mean-shift kernel)."""
    for i in range(out.shape[0]):
        for k in range(out.shape[1]):
            x = (offs[k] - d[i]) / sigma
            out[i, k] = -0.5 * (x * x)


@nb.njit(cache=True)
def ms_prepare(prof, clip, use_clip, lo_inf, hi_inf, offs, sigma, g):
    """In place on the (N, K) profiles, in their own dtype: NaN -> 0 and +-inf -> the dtype's extremes
    (np.nan_to_num), optional np.minimum(prof, clip), then subtract each row's minimum.  Also writes the
    mean-shift kernel exponent for d = 0 into g."""
    N, K = prof.shape
    for i in range(N):
        for k in range(K):
            v = prof[i, k]
            if np.isnan(v):
                v = 0
            elif v == np.inf:
                v = hi_inf
            elif v == -np.inf:
                v = lo_inf
            if use_clip:
                v = min(v, clip)
            prof[i, k] = v
        m = prof[i, 0]
        for k in range(1, K):
            m = min(m, prof[i, k])
        for k in range(K):
            prof[i, k] = prof[i, k] - m
            x = offs[k] / sigma  # (offs - d) / sigma with d = 0
            g[i, k] = -0.5 * (x * x)


@nb.njit(cache=True)
def ms_shift(g, prof, offs, d, sigma, lo, hi, dn, tmp):
    """One mean-shift update given g = exp(-0.5 ((offs - d) / sigma)^2): w = g * prof,
    dn = sum(w offs) / max(sum w, 1e-9) where sum w > 0 (else d), clipped to [lo[i], hi[i]] (+-radius,
    narrowed to the basin when confined).  Then g is overwritten with the exponent for dn (ready for the
    next np.exp).  Returns max |dn - d|."""
    N, K = g.shape
    big = 0.0
    for i in range(N):
        w = g[i]
        t = tmp[i]
        for k in range(K):
            w[k] = w[k] * prof[i, k]
            t[k] = w[k] * offs[k]
        sw = _rowsum(w)
        if sw > 0:
            v = _rowsum(t) / max(sw, 1e-9)
        else:
            v = d[i]
        v = min(max(v, lo[i]), hi[i])
        dn[i] = v
        big = max(big, abs(v - d[i]))
        for k in range(K):
            x = (offs[k] - v) / sigma
            w[k] = -0.5 * (x * x)
    return big


@nb.njit(parallel=True, cache=True)
def ms_shift_par(g, prof, offs, d, sigma, lo, hi, dn, tmp):
    """ms_shift, parallel over rows (for large N; identical results)."""
    N, K = g.shape
    big = 0.0
    for i in nb.prange(N):
        w = g[i]
        t = tmp[i]
        for k in range(K):
            w[k] = w[k] * prof[i, k]
            t[k] = w[k] * offs[k]
        sw = _rowsum(w)
        if sw > 0:
            v = _rowsum(t) / max(sw, 1e-9)
        else:
            v = d[i]
        v = min(max(v, lo[i]), hi[i])
        dn[i] = v
        big = max(big, abs(v - d[i]))
        for k in range(K):
            x = (offs[k] - v) / sigma
            w[k] = -0.5 * (x * x)
    return big


@nb.njit(cache=True)
def ms_finish(g, prof, d, radius, min_contrast, reach, step, half, max_width, use_max_width, want_width,
              strict, ok, width, tmp):
    """Contrast test at the mode (kernel-weighted density vs the profile minimum, g = exp(ms_gauss_arg) at
    the final d), the radius test (strict: the mode must lie inside +-radius) and the bright-run width gate
    of trace._bright_width, in the profile's own dtype."""
    N, K = g.shape
    for i in range(N):
        t = tmp[i]
        for k in range(K):
            t[k] = prof[i, k] * g[i, k]
        at_mode = _rowsum(t) / _rowsum(g[i])
        good = at_mode >= min_contrast
        if strict:
            good = good & (abs(d[i]) < radius - 1e-6)
        if want_width:
            im = int(np.rint((d[i] + reach) / step))
            im = min(max(im, 0), K - 1)
            base = prof[i, 0]
            for k in range(1, K):
                base = min(base, prof[i, k])
            peak = prof[i, im]
            thr = base + half * (peak - base)
            li = im
            while li >= 0 and prof[i, li] > thr:
                li -= 1
            ri = im
            while ri < K and prof[i, ri] > thr:
                ri += 1
            width[i] = (ri - li - 1) * step
            if use_max_width:
                good = good & (width[i] <= max_width)
        ok[i] = good


@nb.njit(parallel=True, cache=True)
def plane_moments(P, fr, fc, wt, M, B, cnt):
    """Weighted least-squares moments of the plane P(dc, dr) = a + b dc + c dr fitted to the valid
    vertices of the (2R+1)^2 grid window around each cell (fr, fc) (the cell itself excluded):
    M = sum w phi phi^T, B = sum w phi P^T with phi = (1, dc, dr), w = wt[dr + R, dc + R]; cnt = number of
    valid vertices.  Accumulated in window order with numpy's operation order (bit-identical to the
    vectorised numpy sums)."""
    N0, N1 = P.shape[0], P.shape[1]
    R = wt.shape[0] // 2
    for i in nb.prange(fr.shape[0]):
        for j in range(3):
            for k in range(3):
                M[i, j, k] = 0.0
                B[i, j, k] = 0.0
        cnt[i] = 0
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                if dr == 0 and dc == 0:
                    continue
                rr = fr[i] + dr
                cc = fc[i] + dc
                if rr < 0 or rr >= N0 or cc < 0 or cc >= N1 or not np.isfinite(P[rr, cc, 0]):
                    continue
                w = wt[dr + R, dc + R]
                phi0, phi1, phi2 = 1.0, float(dc), float(dr)
                for j in range(3):
                    pj = phi0 if j == 0 else (phi1 if j == 1 else phi2)
                    M[i, j, 0] += w * (pj * phi0)
                    M[i, j, 1] += w * (pj * phi1)
                    M[i, j, 2] += w * (pj * phi2)
                    for k in range(3):
                        B[i, j, k] += (w * pj) * P[rr, cc, k]
                cnt[i] += 1


@nb.njit(parallel=True, cache=True)
def window_nanmedian(G, fr, fc, R, out):
    """Median of the finite values of G over the (2R+1)^2 window around each (fr, fc), the centre
    excluded, indices clipped to the grid (np.nanmedian of the stacked window); NaN if none."""
    N0, N1 = G.shape
    for i in nb.prange(fr.shape[0]):
        v = np.empty((2 * R + 1) ** 2)
        m = 0
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                if dr == 0 and dc == 0:
                    continue
                x = G[min(max(fr[i] + dr, 0), N0 - 1), min(max(fc[i] + dc, 0), N1 - 1)]
                if np.isfinite(x):
                    v[m] = x
                    m += 1
        if m == 0:
            out[i] = np.nan
            continue
        s = np.sort(v[:m])
        out[i] = s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0


@nb.njit(parallel=True, cache=True)
def subvoxel_peak(prof, offs, step, valid, out):
    """refine.peak_offsets' peak pick, per vertex j of prof (K, M): i = first argmax with NaN as -inf,
    parabolic sub-voxel refinement around clip(i, 1, K-2) (NaN unless a real maximum: a - 2b + c < 0),
    NaN if the maximum is at the window edge or the vertex is invalid.  Same float64 operations."""
    K = prof.shape[0]
    for j in nb.prange(prof.shape[1]):
        bi = 0
        best = -np.inf
        for k in range(K):
            v = prof[k, j]
            if v > best:  # NaN never compares greater: it acts as -inf
                best = v
                bi = k
        if not valid[j] or bi == 0 or bi == K - 1:
            out[j] = np.nan
            continue
        i = bi  # already inside [1, K-2]
        a, b, c = prof[i - 1, j], prof[i, j], prof[i + 1, j]
        den = a - 2 * b + c
        if den < 0:
            out[j] = offs[i] + (0.5 * (a - c) / den) * step
        else:
            out[j] = offs[i] + np.nan * step


@nb.njit(cache=True, inline="always")
def _tri_norm(P, r0, c0, r1, c1, r2, c2):
    """|cross(P1 - P0, P2 - P0)| in float64 with numpy's operation order (np.cross, np.linalg.norm)."""
    a0 = np.float64(P[r1, c1, 0]) - np.float64(P[r0, c0, 0])
    a1 = np.float64(P[r1, c1, 1]) - np.float64(P[r0, c0, 1])
    a2 = np.float64(P[r1, c1, 2]) - np.float64(P[r0, c0, 2])
    b0 = np.float64(P[r2, c2, 0]) - np.float64(P[r0, c0, 0])
    b1 = np.float64(P[r2, c2, 1]) - np.float64(P[r0, c0, 1])
    b2 = np.float64(P[r2, c2, 2]) - np.float64(P[r0, c0, 2])
    x = a1 * b2 - a2 * b1
    y = a2 * b0 - a0 * b2
    z = a0 * b1 - a1 * b0
    s = 0.0 + x * x
    s += y * y
    s += z * z
    return np.sqrt(s)


@nb.njit(cache=True)
def triangle_norms(P, ok):
    """Per-triangle |cross| for Surface.triangles()' triangles, in exactly its order: (a, c, b) of every
    full quad, then (b, c, d) of every full quad, then the 3-valid quads missing a, b, c, d in turn."""
    H, W = ok.shape
    nfull = 0
    n3 = 0
    for r in range(H - 1):
        for c in range(W - 1):
            k = ok[r, c] + ok[r, c + 1] + ok[r + 1, c] + ok[r + 1, c + 1]
            nfull += k == 4
            n3 += k == 3
    out = np.empty(2 * nfull + n3)
    j = 0
    for r in range(H - 1):  # a c b
        for c in range(W - 1):
            if ok[r, c] and ok[r, c + 1] and ok[r + 1, c] and ok[r + 1, c + 1]:
                out[j] = _tri_norm(P, r, c, r + 1, c, r, c + 1)
                j += 1
    for r in range(H - 1):  # b c d
        for c in range(W - 1):
            if ok[r, c] and ok[r, c + 1] and ok[r + 1, c] and ok[r + 1, c + 1]:
                out[j] = _tri_norm(P, r, c + 1, r + 1, c, r + 1, c + 1)
                j += 1
    for miss in range(4):
        for r in range(H - 1):
            for c in range(W - 1):
                a, b, cc, d = ok[r, c], ok[r, c + 1], ok[r + 1, c], ok[r + 1, c + 1]
                if a + b + cc + d != 3:
                    continue
                if miss == 0 and not a:  # (b, c, d)
                    out[j] = _tri_norm(P, r, c + 1, r + 1, c, r + 1, c + 1)
                    j += 1
                elif miss == 1 and not b:  # (a, c, d)
                    out[j] = _tri_norm(P, r, c, r + 1, c, r + 1, c + 1)
                    j += 1
                elif miss == 2 and not cc:  # (a, d, b)
                    out[j] = _tri_norm(P, r, c, r + 1, c + 1, r, c + 1)
                    j += 1
                elif miss == 3 and not d:  # (a, c, b)
                    out[j] = _tri_norm(P, r, c, r + 1, c, r, c + 1)
                    j += 1
    return out


@nb.njit(cache=True)
def _tri_rows(g, t):
    """Row range [lo, hi] of triangle t's bounding box, exactly as rasterize_tris computes it;
    lo > hi when the triangle is degenerate (skipped)."""
    x0, y0 = g[t, 0, 0], g[t, 0, 1]
    d1x, d1y = g[t, 1, 0] - x0, g[t, 1, 1] - y0
    d2x, d2y = g[t, 2, 0] - x0, g[t, 2, 1] - y0
    den = d1x * d2y - d1y * d2x
    if den == 0.0 or not np.isfinite(den):
        return 1, 0
    loy = int(np.ceil(min(y0, y0 + d1y, y0 + d2y) - 1e-9))
    hiy = int(np.floor(max(y0, y0 + d1y, y0 + d2y) + 1e-9))
    return loy, hiy


@nb.njit(cache=True)
def _bucket_rows(g, H, band):
    """CSR lists of the triangles touching each band of `band` output rows, in triangle order."""
    nb_ = (H + band - 1) // band
    cnt = np.zeros(nb_ + 1, np.int64)
    for t in range(g.shape[0]):
        lo, hi = _tri_rows(g, t)
        lo, hi = max(lo, 0), min(hi, H - 1)
        if lo > hi:
            continue
        for b in range(lo // band, hi // band + 1):
            cnt[b + 1] += 1
    for b in range(nb_):
        cnt[b + 1] += cnt[b]
    ids = np.empty(cnt[nb_], np.int64)
    pos = cnt[:nb_].copy()
    for t in range(g.shape[0]):
        lo, hi = _tri_rows(g, t)
        lo, hi = max(lo, 0), min(hi, H - 1)
        if lo > hi:
            continue
        for b in range(lo // band, hi // band + 1):
            ids[pos[b]] = t
            pos[b] += 1
    return cnt, ids


@nb.njit(parallel=True, cache=True)
def rasterize_tris_banded(g, T, V, out, best, band):
    """rasterize_tris, parallel over bands of output rows.  Each band visits the triangles touching it
    in the original triangle order and only writes its own rows, so every pixel sees the same candidate
    sequence as the serial kernel: identical output."""
    H, W = best.shape
    cnt, ids = _bucket_rows(g, H, band)
    nb_ = cnt.shape[0] - 1
    for b in nb.prange(nb_):
        r_lo = b * band
        r_hi = min(H - 1, r_lo + band - 1)
        for q in range(cnt[b], cnt[b + 1]):
            t = ids[q]
            x0, y0 = g[t, 0, 0], g[t, 0, 1]
            d1x, d1y = g[t, 1, 0] - x0, g[t, 1, 1] - y0
            d2x, d2y = g[t, 2, 0] - x0, g[t, 2, 1] - y0
            den = d1x * d2y - d1y * d2x
            lox = int(np.ceil(min(x0, x0 + d1x, x0 + d2x) - 1e-9))
            hix = int(np.floor(max(x0, x0 + d1x, x0 + d2x) + 1e-9))
            loy = int(np.ceil(min(y0, y0 + d1y, y0 + d2y) - 1e-9))
            hiy = int(np.floor(max(y0, y0 + d1y, y0 + d2y) + 1e-9))
            for py in range(max(loy, r_lo), min(hiy, r_hi) + 1):
                for px in range(max(lox, 0), min(hix, W - 1) + 1):
                    rx = px - x0
                    ry = py - y0
                    b1 = (rx * d2y - ry * d2x) / den
                    b2 = (d1x * ry - d1y * rx) / den
                    b0 = 1.0 - b1 - b2
                    m = min(b0, min(b1, b2))
                    if m >= -1e-6 and m > best[py, px]:
                        best[py, px] = m
                        for k in range(3):
                            out[py, px, k] = b0 * V[T[t, 0], k] + b1 * V[T[t, 1], k] + b2 * V[T[t, 2], k]


@nb.njit(parallel=True, cache=True)
def trilinear_periodic(T, a, b, c, out):
    """Trilinear lookup of a periodic 3D tile T at (a, b, c) (any real coordinates, wrapped)."""
    A, B, C = T.shape
    for i in nb.prange(a.shape[0]):
        x, y, z = a[i], b[i], c[i]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            out[i] = 0.0
            continue
        x0 = np.floor(x)
        y0 = np.floor(y)
        z0 = np.floor(z)
        tx, ty, tz = x - x0, y - y0, z - z0
        i0 = int(x0) % A
        j0 = int(y0) % B
        k0 = int(z0) % C
        i1 = (i0 + 1) % A
        j1 = (j0 + 1) % B
        k1 = (k0 + 1) % C
        out[i] = ((1 - tx) * ((1 - ty) * ((1 - tz) * T[i0, j0, k0] + tz * T[i0, j0, k1])
                              + ty * ((1 - tz) * T[i0, j1, k0] + tz * T[i0, j1, k1]))
                  + tx * ((1 - ty) * ((1 - tz) * T[i1, j0, k0] + tz * T[i1, j0, k1])
                          + ty * ((1 - tz) * T[i1, j1, k0] + tz * T[i1, j1, k1])))


# ---------------------------------------------------------------- along-normal regularisation (regularize.py)
@nb.njit(cache=True)
def _reach(ok, jump_u, jump_v, r, c, rad_r, rad_c, seen, qr, qc):
    """Cells reachable from (r, c) inside the (2 rad_r + 1) x (2 rad_c + 1) window through valid cells ``ok``,
    never crossing a grid edge flagged in jump_u (H, W-1) / jump_v (H-1, W); written to qr/qc, count returned."""
    H, W = ok.shape
    seen[:, :] = False
    seen[rad_r, rad_c] = True
    qr[0], qc[0] = r, c
    head, tail = 0, 1
    while head < tail:
        y, x = qr[head], qc[head]
        head += 1
        for k in range(4):
            yy = y + (k == 0) - (k == 1)
            xx = x + (k == 2) - (k == 3)
            if yy < 0 or yy >= H or xx < 0 or xx >= W or abs(yy - r) > rad_r or abs(xx - c) > rad_c:
                continue
            if seen[yy - r + rad_r, xx - c + rad_c] or not ok[yy, xx]:
                continue
            if (k == 0 and jump_v[y, x]) or (k == 1 and jump_v[yy, x]) or (k == 2 and jump_u[y, x]) or \
                    (k == 3 and jump_u[y, xx]):
                continue
            seen[yy - r + rad_r, xx - c + rad_c] = True
            qr[tail], qc[tail] = yy, xx
            tail += 1
    return tail


@nb.njit(parallel=True, cache=True)
def robust_smooth(t, ok, jump_u, jump_v, rad_r, rad_c, sig_r, sig_c, tol, out):
    """Robust masked gaussian of a grid field t (H, W) (NaN = no estimate) over the valid cells ``ok``: per valid
    cell, of the finite t of the cells _reach() finds, the gaussian-weighted median m, then the weighted mean of
    those within tol of m.  NaN where nothing finite is reachable or the cell is not valid."""
    H, W = t.shape
    wr, wc = 2 * rad_r + 1, 2 * rad_c + 1
    for r in nb.prange(H):
        seen = np.zeros((wr, wc), np.bool_)
        qr = np.empty(wr * wc, np.int64)
        qc = np.empty(wr * wc, np.int64)
        vals = np.empty(wr * wc)
        wts = np.empty(wr * wc)
        for c in range(W):
            out[r, c] = np.nan
            if not ok[r, c]:
                continue
            n = 0
            for i in range(_reach(ok, jump_u, jump_v, r, c, rad_r, rad_c, seen, qr, qc)):
                v = t[qr[i], qc[i]]
                if np.isfinite(v):
                    dy, dx = (qr[i] - r) / sig_r, (qc[i] - c) / sig_c
                    vals[n] = v
                    wts[n] = np.exp(-0.5 * (dy * dy + dx * dx))
                    n += 1
            if n == 0:
                continue
            order = np.argsort(vals[:n])
            half = 0.5 * wts[:n].sum()
            acc = 0.0
            m = vals[order[n - 1]]
            for k in order:
                acc += wts[k]
                if acc >= half:
                    m = vals[k]
                    break
            sw, sv = 0.0, 0.0
            for k in range(n):
                if abs(vals[k] - m) <= tol:
                    sw += wts[k]
                    sv += wts[k] * vals[k]
            out[r, c] = sv / sw


@nb.njit(parallel=True, cache=True)
def reach_mean(P, ok, jump_u, jump_v, rad_r, rad_c, sig_r, sig_c, out):
    """Gaussian-weighted mean of the (H, W, 3) positions of the cells _reach() finds from each valid cell (a masked
    gaussian that does not cross holes or flagged edges); NaN at invalid cells."""
    H, W = ok.shape
    wr, wc = 2 * rad_r + 1, 2 * rad_c + 1
    for r in nb.prange(H):
        seen = np.zeros((wr, wc), np.bool_)
        qr = np.empty(wr * wc, np.int64)
        qc = np.empty(wr * wc, np.int64)
        for c in range(W):
            out[r, c, :] = np.nan
            if not ok[r, c]:
                continue
            sw, a0, a1, a2 = 0.0, 0.0, 0.0, 0.0
            for i in range(_reach(ok, jump_u, jump_v, r, c, rad_r, rad_c, seen, qr, qc)):
                y, x = qr[i], qc[i]
                dy, dx = (y - r) / sig_r, (x - c) / sig_c
                w = np.exp(-0.5 * (dy * dy + dx * dx))
                sw += w
                a0 += w * (P[y, x, 0] - P[r, c, 0])
                a1 += w * (P[y, x, 1] - P[r, c, 1])
                a2 += w * (P[y, x, 2] - P[r, c, 2])
            out[r, c, 0] = P[r, c, 0] + a0 / sw
            out[r, c, 1] = P[r, c, 1] + a1 / sw
            out[r, c, 2] = P[r, c, 2] + a2 / sw


@nb.njit(cache=True)
def membrane_fill(P, fill, iters, eps):
    """Discrete Laplace interpolation of the (H, W, 3) grid P on the cells ``fill`` (Gauss-Seidel, in place);
    the other finite cells are the boundary.  Cells of ``fill`` must start finite (an initial guess)."""
    H, W = fill.shape
    for _ in range(iters):
        delta = 0.0
        for r in range(H):
            for c in range(W):
                if not fill[r, c]:
                    continue
                a0, a1, a2, cnt = 0.0, 0.0, 0.0, 0
                for k in range(4):
                    y = r + (k == 0) - (k == 1)
                    x = c + (k == 2) - (k == 3)
                    if 0 <= y < H and 0 <= x < W and np.isfinite(P[y, x, 0]):
                        a0 += P[y, x, 0]
                        a1 += P[y, x, 1]
                        a2 += P[y, x, 2]
                        cnt += 1
                if cnt:
                    n0, n1, n2 = a0 / cnt, a1 / cnt, a2 / cnt
                    delta = max(delta, abs(n0 - P[r, c, 0]) + abs(n1 - P[r, c, 1]) + abs(n2 - P[r, c, 2]))
                    P[r, c, 0], P[r, c, 1], P[r, c, 2] = n0, n1, n2
        if delta < eps:
            break
