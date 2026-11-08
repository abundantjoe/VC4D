"""Flatten a surface into the plane: an initial map refined by ARAP.

* init: the tifxyz grid layout (default, scalable) or LSCM (Levy et al. 2002, conformal).
* ARAP (Liu et al. 2008, local/global) pulls it toward an isometry, which is what physical
  unrolling of papyrus should be: the sheet does not stretch, so letter shapes and line
  spacing are preserved only if per-triangle singular values stay ~1.
* solved coarse-to-fine on strided grids (rotation-field prolongation) with Anderson acceleration.

UVs are produced per grid vertex (same grid as the input), in voxel units.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from scipy.sparse.csgraph import connected_components

from . import _kernels as K
from .surface import Surface, bilerp_grid


def _frames(V, T, flip=None):
    """Each triangle isometrically in 2D: (T,3,2) local coords, and the frame axes x, y (T,3) in 3D.

    ``flip`` (bool per triangle) mirrors the frame of grid-folded triangles (see _grid_folds).
    """
    p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1 = p1 - p0
    l1 = np.linalg.norm(e1, axis=1)
    x = e1 / l1[:, None]
    n = np.cross(e1, p2 - p0)
    y = np.cross(n / np.linalg.norm(n, axis=1)[:, None], x)
    q = np.zeros((len(T), 3, 2))
    q[:, 1, 0] = l1
    d = p2 - p0
    q[:, 2, 0] = (d * x).sum(1)
    q[:, 2, 1] = (d * y).sum(1)
    if flip is not None:
        y[flip] *= -1
        q[flip, 2, 1] *= -1
    return q, x, y


def _local_frames(V, T, flip=None):
    """Each triangle isometrically in 2D: returns (T,3,2) local coords."""
    return _frames(V, T, flip)[0]


def _grid_folds(surface, idx, V, T):
    """Triangles whose 3D winding disagrees with the smooth grid normal du x dv.

    On irregular (tracer) grids neighbouring vertices can cross over on the sheet, so a cell's
    triangle is wound backwards although the sheet itself is not folded: with jitter +-0.35/0.45/0.6 vx
    the exact unrolling has 591/4526/18399 such triangles of 257k.  Their rest frame is mirrored so
    that the true unrolling is a rotation for every triangle; otherwise ARAP cannot reach it.
    The reference is the grid normal smoothed over sigma = 2 cells, trusted only where it is coherent
    (resultant > 0.95): on the real Scroll 1 grid the 56 backward-wound triangles lie mostly in creases
    (cos -0.99..-0.04 against the raw normal) where it is not, and are left alone (8 are mirrored).
    """
    nv = surface.normals().astype(np.float64)
    ok = np.isfinite(nv[..., 0])
    nv[~ok] = 0
    w = ndimage.gaussian_filter(ok.astype(np.float64), 2.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.stack([ndimage.gaussian_filter(nv[..., k], 2.0) for k in range(3)], -1) / w[..., None]
        coherence = np.linalg.norm(sm, axis=-1)  # 1 on a smooth sheet, < 1 across creases
        ref = sm[idx >= 0][T].sum(1)
        n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        c = (n * ref).sum(1) / (np.linalg.norm(n, axis=1) * np.linalg.norm(ref, axis=1))
    smooth = coherence[idx >= 0][T].min(1) > 0.95  # the reference is trustworthy (not a crease)
    return smooth & (c * np.sign(np.nanmedian(c)) < 0)  # the triangulation fixes the winding vs du x dv


def _clean(V, T, keep=None):
    """Drop degenerate triangles and keep the largest connected component - or, with ``keep`` (bool per vertex),
    every component that contains a kept vertex (anchored solves: each anchored piece has its own gauge)."""
    a = np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    med = np.median(a) if len(a) else 0
    T = T[a > 1e-6 * max(med, 1e-12)]
    # largest connected component (edge-connected through shared vertices)
    n = len(V)
    rows = np.r_[T[:, 0], T[:, 1], T[:, 2]]
    cols = np.r_[T[:, 1], T[:, 2], T[:, 0]]
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    used = np.zeros(n, bool)
    used[T.ravel()] = True
    if keep is not None and np.any(keep & used):
        labs = np.zeros(ncomp, bool)
        labs[lab[keep & used]] = True
        return T[labs[lab[T[:, 0]]]]
    counts = np.bincount(lab[used], minlength=ncomp)
    keep_lab = np.argmax(counts)
    T = T[lab[T[:, 0]] == keep_lab]
    return T


def lscm(V, T):
    q = _local_frames(V, T)
    nT, nV = len(T), len(V)
    area2 = q[:, 1, 0] * q[:, 2, 1]  # 2*area, >0 by construction
    w = 1.0 / np.sqrt(area2)
    # complex weights W_j = (x_{j+2}-x_{j+1}) + i (y_{j+2}-y_{j+1}); residual sum_j W_j (u_j + i v_j)
    Wr = np.zeros((nT, 3))
    Wi = np.zeros((nT, 3))
    for j in range(3):
        k, l = (j + 1) % 3, (j + 2) % 3
        Wr[:, j] = (q[:, l, 0] - q[:, k, 0]) * w
        Wi[:, j] = (q[:, l, 1] - q[:, k, 1]) * w
    r = np.arange(nT)
    rows, cols, vals = [], [], []
    for j in range(3):
        vj = T[:, j]
        # real part: Wr*u - Wi*v ; imag part: Wi*u + Wr*v
        rows += [r, r, nT + r, nT + r]
        cols += [vj, nV + vj, vj, nV + vj]
        vals += [Wr[:, j], -Wi[:, j], Wi[:, j], Wr[:, j]]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * nT, 2 * nV))
    # pin two far-apart vertices at their true 3D distance
    used = np.unique(T)
    c = V[used].mean(0)
    i0 = used[np.argmax(np.linalg.norm(V[used] - c, axis=1))]
    i1 = used[np.argmax(np.linalg.norm(V[used] - V[i0], axis=1))]
    d = np.linalg.norm(V[i1] - V[i0])
    pin_idx = np.array([i0, nV + i0, i1, nV + i1])
    pin_val = np.array([0.0, 0.0, d, 0.0])
    free = np.setdiff1d(np.r_[used, nV + used], pin_idx)
    Af, Ap = A[:, free], A[:, pin_idx]
    b = -Ap @ pin_val
    M = (Af.T @ Af).tocsc()
    x = spla.spsolve(M, Af.T @ b, permc_spec="MMD_AT_PLUS_A")  # symmetric ordering: far less fill-in
    uv = np.full(2 * nV, np.nan)
    uv[free] = x
    uv[pin_idx] = pin_val
    return np.c_[uv[:nV], uv[nV:]]


def _cot_weights(q):
    """(T,3) ARAP weight of edge j = (j+1, j+2): half the cotangent of the angle at vertex j, clamped
    to [1e-3, 1e3].

    The exact cot form of a triangle is PSD, but clipping a sliver's huge +/- cotangents made it
    indefinite: on tracer-like grids (tangential jitter +-0.45 vx) ARAP then folded (351 flips, E < 0).
    Non-negative weights keep every triangle's energy PSD (and the Laplacian an M-matrix); at most one
    angle is obtuse, so the other two edges still fix the triangle, and an isometry still has zero energy.
    """
    w = np.empty((len(q), 3))
    for j in range(3):
        a = q[:, (j + 1) % 3] - q[:, j]
        b = q[:, (j + 2) % 3] - q[:, j]
        w[:, j] = np.clip((a * b).sum(1) / np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), 1e-3, 1e3) * 0.5
    return w


def _nested_dissection(rr, cc, leaf=64):
    """Fill-reducing elimination order for grid vertices at (row rr, col cc).

    Grid triangles only join adjacent rows/cols, so one grid line separates the two halves; ordering
    the separator (the median line across the longer extent) last, recursively, is nested dissection.
    On the 373k-vertex Scroll 1 grid: SuperLU factorises in ~2.5 s vs 6.4 s with MMD_AT_PLUS_A.
    """
    out = []

    def rec(ids):
        if len(ids) <= leaf:
            out.append(ids)
            return
        r, c = rr[ids], cc[ids]
        key = r if np.ptp(r) >= np.ptp(c) else c
        m = int(np.median(key))
        rec(ids[key < m])
        rec(ids[key > m])
        out.append(ids[key == m])

    rec(np.arange(len(rr)))
    return np.concatenate(out)


def _grid_prolongation(rr, cc, cidx, corder, nfree=None):
    """Sparse bilinear interpolation (n, n_c) from the free vertices of the next coarser level (stride 2,
    index grid ``cidx``, internal order ``corder``) to fine vertices at grid (rr, cc).  Missing coarse
    corners are dropped and the weights renormalised; a vertex with none takes its nearest coarse vertex.
    """
    nc = len(corder) - 1 if nfree is None else nfree
    cpos = np.full(int(cidx.max()) + 1, -1)
    cpos[corder[:nc]] = np.arange(nc)
    cg = np.where(cidx >= 0, cpos[np.maximum(cidx, 0)], -1)
    H, W = cg.shape
    r, c = rr / 2.0, cc / 2.0
    r0, c0 = np.minimum(rr // 2, H - 1), np.minimum(cc // 2, W - 1)
    n = len(rr)
    rows, cols, vals = [], [], []
    for dr in (0, 1):
        for dc in (0, 1):
            ri, ci = r0 + dr, c0 + dc
            w = np.clip(1 - np.abs(r - ri), 0, None) * np.clip(1 - np.abs(c - ci), 0, None)
            g = cg[np.minimum(ri, H - 1), np.minimum(ci, W - 1)]
            ok = (ri < H) & (ci < W) & (w > 0) & (g >= 0)
            rows.append(np.flatnonzero(ok))
            cols.append(g[ok])
            vals.append(w[ok])
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    wsum = np.bincount(rows, vals, n)
    miss = np.flatnonzero(wsum == 0)
    if len(miss):
        _, (ii, jj) = ndimage.distance_transform_edt(cg < 0, return_indices=True)
        rows = np.concatenate([rows, miss])
        cols = np.concatenate([cols, cg[ii[r0[miss], c0[miss]], jj[r0[miss], c0[miss]]]])
        vals = np.concatenate([vals, np.ones(len(miss))])
        wsum[miss] = 1.0
    return sp.csr_matrix((vals / wsum[rows], (rows, cols)), shape=(n, nc))


def _spmv(A, x):
    """A @ x for CSR A and an (n, 2) x, parallel."""
    return K.csr_matvec2(A.indptr, A.indices, A.data, np.ascontiguousarray(x), np.empty((A.shape[0], 2)))


class _Arap:
    """ARAP (Liu et al. 2008, local/global) on one grid mesh, Anderson-accelerated.

    Global step: the cot Laplacian (SPD once one vertex is pinned) is factorised once (SuperLU, grid
    nested dissection, symmetric mode) or, given ``coarse`` = (coarser _Arap, its index grid), solved
    by warm-started PCG with a symmetric V-cycle (Gauss-Seidel, bilinear prolongation to the coarser
    level's operator, recursively down to a factorised level): memory stays O(n).  On the real Scroll 1
    grid PCG needed 16 iterations at 373k and 19 at 1.5M vertices for a 1e-8 relative residual from a
    cold start, and ~11 per ARAP iteration at 6M vertices when warm-started.
    Local/global alone converges slowly on large meshes; Anderson acceleration (Peng et al. 2018,
    memory 5, with an energy safeguard) took the real 373k-vertex segment from a 6.8e-9 to a 7e-14
    relative energy gap in 40 iterations from the same start.
    Internally vertices are numbered in elimination order, the pinned vertex last.
    """

    def __init__(self, V, T, rr=None, cc=None, coarse=None, rtol=1e-10, flip=None, anchor=None):
        """``anchor`` = (lam (nV,), target (nV, 2)): soft anchors lam |u - target|^2 added to the energy (a
        diagonal and a right-hand side in the global step).  With anchors no vertex is pinned."""
        q, self.x, self.y = _frames(V, T, flip)
        self.cot = _cot_weights(q)
        self.ex = np.stack([q[:, (j + 1) % 3] - q[:, (j + 2) % 3] for j in range(3)], 1)
        del q
        used = np.flatnonzero(np.bincount(T.ravel(), minlength=len(V)))
        lam = None
        if anchor is not None:
            lam = np.asarray(anchor[0], np.float64)
            if not np.any(lam[used] > 0):
                lam = None
        self.pinned = lam is None
        keep = used[1:] if self.pinned else used
        direct = coarse is None
        self.iterative = not direct
        grid = rr is not None  # grid coords given: nested dissection, else SuperLU's own ordering
        self.order = keep[_nested_dissection(rr[keep], cc[keep])] if grid and direct else keep
        if self.pinned:
            self.order = np.append(self.order, used[0])
        n = len(keep)  # free vertices; the pinned one (if any) is n
        self.nf = n
        pos = np.full(len(V), -1, np.int32)
        pos[self.order] = np.arange(len(self.order), dtype=np.int32)
        self.T = np.ascontiguousarray(pos[T])
        k, l, w = self.T[:, [1, 2, 0]].ravel(), self.T[:, [2, 0, 1]].ravel(), self.cot.ravel()
        diag = np.bincount(k, w, n + 1) + np.bincount(l, w, n + 1)
        cp = (k < n) & (l == n)
        cp2 = (l < n) & (k == n)
        self.lpin = (np.bincount(k[cp], -w[cp], n) + np.bincount(l[cp2], -w[cp2], n))[:, None]  # coupling to the pin
        ok = (k < n) & (l < n)
        k, l, w = k[ok], l[ok], w[ok]
        if lam is not None:  # soft anchors: lam on the diagonal, lam * target on the right-hand side
            la = lam[self.order]
            diag[:n] += la
            self.anc_lam = la.astype(np.float32)[:, None]  # (lam * target is formed in solve: not stored)
            self.anc_tgt = np.nan_to_num(np.asarray(anchor[1], np.float64)[self.order])
        else:
            self.anc_lam = self.anc_tgt = None
        L = sp.csr_matrix((np.concatenate([-w, -w, diag[:n]]),
                           (np.concatenate([k, l, np.arange(n, dtype=np.int32)]),
                            np.concatenate([l, k, np.arange(n, dtype=np.int32)]))), shape=(n, n))
        del k, l, w
        if direct:
            self.lu = spla.splu(L.tocsc(), permc_spec="NATURAL" if grid else "MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                options={"SymmetricMode": True})
        else:
            self.L, self.d, self.rtol, self.coarse = L, diag[:n], rtol, coarse[0]
            self.P = _grid_prolongation(rr[keep], cc[keep], coarse[1], coarse[0].order, coarse[0].nf)
            # grid 4-colouring (row, col parity): 8-neighbours never share a colour, so each colour's
            # Gauss-Seidel update is independent and runs in parallel
            colour = (rr[keep] % 2) + 2 * (cc[keep] % 2)
            self.colours = [np.flatnonzero(colour == k).astype(np.int32) for k in range(4)]
            self.PT = self.P.T.tocsr()
            self.last = None
        self.cg_iterations = 0
        self.th = np.zeros(len(T))
        self.b = np.zeros((n + 1 if self.pinned else n, 2))

    def precondition(self, r):
        """Exact solve (factorised level) or one symmetric V-cycle for L z = r."""
        if not self.iterative:
            return self.lu.solve(np.asfortranarray(r))
        L = self.L
        z = np.zeros_like(r)
        for rows in self.colours:  # symmetric: colours forward, then backward
            K.gauss_seidel_rows(L.indptr, L.indices, L.data, self.d, r, z, rows)
        rc = self.coarse.precondition(_spmv(self.PT, r - self.matvec(z)))
        K.axpy2(z, np.ones(2), _spmv(self.P, rc))
        for rows in self.colours[::-1]:
            K.gauss_seidel_rows(L.indptr, L.indices, L.data, self.d, r, z, rows)
        return z

    def matvec(self, x):
        return _spmv(self.L, x)

    def _pcg(self, b, x, maxit=500, tight=False):
        """Warm-started PCG until the residual dropped 1e4x from the start (or below rtol |b|): the
        start is the previous global solution, so the error is a small fraction of the ARAP step.
        ``tight``: only rtol |b| (used to confirm convergence, see run)."""
        r = b - self.matvec(x)
        stop = self.rtol ** 2 * K.dot2(b, b)  # on squared norms
        if not tight:
            stop = np.maximum(stop, 1e-8 * K.dot2(r, r))
        z = self.precondition(r)
        p, rz = z.copy(), K.dot2(r, z)
        for it in range(maxit):
            if np.all(K.dot2(r, r) <= stop):
                break
            Ap = self.matvec(p)
            a = rz / K.dot2(p, Ap)
            K.axpy2(x, a, p)
            K.axpy2(r, -a, Ap)
            z = self.precondition(r)
            rz, rz0 = K.dot2(r, z), rz
            K.xpby2(z, rz / rz0, p)
        self.cg_iterations += it
        return x

    def energy(self, u):
        e = K.arap_rotations(self.T, self.cot, self.ex, u, self.th)
        if self.anc_lam is not None:
            e += float(np.sum(self.anc_lam * (u[:self.nf] - self.anc_tgt) ** 2))
        return e

    def solve(self, th, pin_uv, guess=None, tight=False):
        """Global step: the layout that best fits the rest triangles rotated by th (pin held at pin_uv).
        ``guess`` (internal order, free vertices) warm-starts the iterative solver."""
        K.arap_rhs(self.T, self.cot, self.ex, th, self.b)
        u = np.empty_like(self.b)
        nf = self.nf
        rhs = self.b[:nf] - self.lpin * pin_uv if self.pinned else self.b[:nf] + self.anc_lam * self.anc_tgt
        if self.iterative:  # warm start from the previous global solution
            x0 = guess if guess is not None else self.last if self.last is not None else np.zeros_like(rhs)
            u[:nf] = self._pcg(rhs, np.nan_to_num(x0), tight=tight)
            self.last = u[:nf].copy()
        else:
            u[:nf] = self.lu.solve(np.asfortranarray(rhs))
        if self.pinned:
            u[-1] = pin_uv
        return u

    def release(self):
        """Drop the mesh data once solved, keeping what a finer level's V-cycle needs."""
        for k in ("T", "cot", "ex", "x", "y", "th", "b", "last"):
            self.__dict__.pop(k, None)

    def to_mesh(self, u, nV):
        uv = np.full((nV, 2), np.nan)
        uv[self.order] = u
        return uv

    def run(self, uv, iterations, tol, log=None, m=5, window=16):
        """Iterate from mesh layout ``uv`` until the estimated distance to the fixed point is < tol voxels.

        The plain local/global step s_k (max vertex move) alone is a poor stop: near the end the iteration
        contracts slowly (holed real grids ~1% per iteration), so s_k < tol left 10-180 tol of error, on
        phantoms too.  The contraction rate rho is estimated as the median of s_k / s_(k-1) over the last
        ``window`` iterations and the remaining distance as s_k rho / (1 - rho); the run stops when that
        has been < tol for window / 2 consecutive iterations (single estimates dip 10-30x low under
        Anderson), or at s_k < tol / 1000 (an exact fixed point).  window=None: stop at s_k < tol (the
        loose coarse levels).  Measured at tol=1e-4 against tol<=1e-7 references: real 373k grid 0.01-0.05
        tol; two holed real grids (343k, 40 holes) 0.4-0.9 tol, one direct run 2.7 tol; a holed spiral
        phantom 6 tol.  The residual misses are "staircase" phases in which a soft mode drifts ~0.1% per
        iteration until Anderson captures it: no step- or iterate-based estimate saw them (a checkpoint
        ratio estimate was worse, 33-560 tol).  Returns (uv, energy, iterations, estimated remaining
        distance; inf if no estimate yet or the iteration is not contracting).
        """
        x = uv[self.order]
        nf = self.nf  # free rows (all but the pin, if any)
        pin_uv = x[-1].copy() if self.pinned else None
        N = x[:nf].size
        # ring buffers of residuals F = G(x) - x and images G(x), and the Gram matrix of F: no other
        # O(N * m) temporaries, which matters at 6M vertices
        Fb, Gb, FF = np.zeros((m + 1, N)), np.zeros((m + 1, N)), np.zeros((m + 1, m + 1))
        nh = 0
        e_acc, g_acc, step, it, remaining, below = np.inf, None, np.inf, 0, np.inf, 0
        confirm = 1 if window is None else window // 2  # consecutive iterations the estimate must hold
        self.exact_step = tol * 1e-3
        self.history = []  # (energy, plain step) per iteration: the convergence record
        for it in range(1, iterations + 1):
            e = self.energy(x)
            if e > e_acc:  # the extrapolated point is worse: fall back to the plain update, reset memory
                x, nh = g_acc, 0
                e = self.energy(x)
            g = self.solve(self.th, pin_uv)
            step = float(np.abs(g - x).max())
            self.history.append((e, step))
            if log:
                log(f"arap iter {it}: energy {e:.10g}, step {step:.3g} vx")
            remaining = self._remaining(step, window)
            below = below + 1 if remaining < tol else 0
            if below >= confirm and self.iterative and step >= self.exact_step:
                # the warm-started CG solve is inexact: confirm the step with a tight solve (holed real
                # grid: without this the stop fired 30 iterations early, 0.152 vx from converged)
                g = self.solve(self.th, pin_uv, guess=g[:nf], tight=True)
                step = float(np.abs(g - x).max())
                self.history[-1] = (e, step)
                remaining = self._remaining(step, window)
                below = below if remaining < tol else 0
            e_acc, g_acc = e, g
            if below >= confirm or step < self.exact_step:
                x = g
                break
            s_ = nh % (m + 1)
            np.subtract(g[:nf].ravel(), x[:nf].ravel(), out=Fb[s_])
            Gb[s_] = g[:nf].ravel()
            FF[s_] = FF[:, s_] = Fb @ Fb[s_]
            nh += 1
            x = g
            if nh > 1:  # Anderson: x = G_last - gamma . dG, gamma = argmin |F_last - gamma . dF|
                h = [(nh - 1 - r) % (m + 1) for r in range(min(nh, m + 1))][::-1]
                D = np.diff(np.eye(len(h)), axis=0)  # dF = D F[h]
                Fh = FF[np.ix_(h, h)]
                M = D @ Fh @ D.T
                gam = np.linalg.solve(M + 1e-12 * np.trace(M) * np.eye(len(M)), D @ Fh[:, -1])
                c = D.T @ gam
                xn = Gb[h[-1]].copy()
                for j, hj in enumerate(h):
                    xn -= c[j] * Gb[hj]
                x = g.copy()
                x[:nf] = xn.reshape(-1, 2)
        return self.to_mesh(x, len(uv)), self.energy(x), it, remaining

    def _remaining(self, step, window):
        """Estimated distance (vx) from the plain update to the fixed point: step rho / (1 - rho)."""
        if window is None or step < self.exact_step:  # plain step criterion (loose coarse levels)
            return step
        s = np.array([h[1] for h in self.history[-(window + 1):]])
        if len(s) < window + 1 or not np.all(s > 0):
            return np.inf
        rho = float(np.median(s[1:] / s[:-1]))
        return step * rho / (1 - rho) if rho < 1 else np.inf


def arap(V, T, uv, iterations=20, tol=1e-3, log=None, return_energy=False):
    """ARAP-refine the layout ``uv`` of mesh (V, T) (stops when the estimated distance to the fixed point
    is < tol voxels, see _Arap.run)."""
    A = _Arap(V, np.asarray(T))
    uv, energy, _, _ = A.run(np.asarray(uv, np.float64), iterations, tol, log)
    return (uv, energy) if return_energy else uv


def grid_init(idx, V, T, scale):
    """Initial UV = nominal grid layout (col/sx, row/sy), oriented to agree with the triangles.

    tifxyz grids from tracers are already roughly evenly spaced, so this is a fold-free start
    that is close to the answer; ARAP then only has to remove the residual distortion.
    """
    rr, cc = np.nonzero(idx >= 0)
    uv = np.c_[cc / scale[0], rr / scale[1]].astype(np.float64)
    e1, e2 = uv[T[:, 1]] - uv[T[:, 0]], uv[T[:, 2]] - uv[T[:, 0]]
    if np.median(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) < 0:
        uv[:, 1] *= -1
    return uv


def _cell_maps(P, U):
    """Per grid cell, the linear 3D->2D map (2,3) of a solved level: (h-1, w-1, 2, 3).

    From the cell's mean du/dv edge vectors in 3D (P) and UV (U).  Cells without a solution (holes,
    the ragged edge) take the map of the nearest solved cell.
    """
    P = P.astype(np.float64)

    def d(X):
        return (0.5 * (X[:-1, 1:] - X[:-1, :-1] + X[1:, 1:] - X[1:, :-1]),
                0.5 * (X[1:, :-1] - X[:-1, :-1] + X[1:, 1:] - X[:-1, 1:]))

    M3, M2 = np.stack(d(P), -1), np.stack(d(U), -1)  # (h, w, 3, 2), (h, w, 2, 2)
    G = np.swapaxes(M3, 2, 3) @ M3
    tr = G[..., 0, 0] + G[..., 1, 1]
    ok = np.isfinite(M2).all((2, 3)) & (np.linalg.det(np.nan_to_num(G)) > 1e-9 * np.nan_to_num(tr) ** 2)
    A = np.zeros(M2.shape[:2] + (2, 3))
    A[ok] = M2[ok] @ np.linalg.solve(G[ok], np.swapaxes(M3[ok], 1, 2))
    _, (ii, jj) = ndimage.distance_transform_edt(~ok, return_indices=True)
    return A[ii, jj]


def _prolong(A, ratio, rr, cc, T, x, y):
    """Rotation angle per fine triangle from a coarser level's cell maps ``A`` (stride ratio).

    The coarse map, bilinearly interpolated at the triangle centroid and expressed in the
    triangle's own frame, is projected onto the nearest rotation.  Prolonging rotations rather than
    positions lets the fine global step rebuild the layout from the fine rest geometry, so the coarse
    chord-for-arc shortening and the ragged edge (no coarse support) do not enter the fine start.
    """
    r = rr[T].mean(1) / ratio - 0.5
    c = cc[T].mean(1) / ratio - 0.5
    At = np.stack([ndimage.map_coordinates(A[..., i, j], [r, c], order=1, mode="nearest")
                   for i in range(2) for j in range(3)], -1).reshape(-1, 2, 3)
    B0, B1 = np.einsum("tij,tj->ti", At, x), np.einsum("tij,tj->ti", At, y)  # columns of the 2x2 map
    return np.arctan2(B0[:, 1] - B1[:, 0], B0[:, 0] + B1[:, 1])


def flatten(surface: Surface, iterations: int = 200, downsample: int = 1, init: str = "grid",
            tol: float = 1e-4, multires: bool = True, coarse_vertices: int = 6000,
            direct_vertices: int = 400_000, log=None, info: dict | None = None, init_uv=None, anchor_mask=None,
            anchor_uv=None, anchor_weight: float = 1.0, vertex_weight=None) -> np.ndarray:
    """Return (H, W, 2) UV in voxel units for every vertex of ``surface`` (NaN where invalid).

    multires (default): coarse-to-fine ARAP on strided grids, the coarsest with <= ``coarse_vertices``
    vertices.  Each finer level starts from the coarser level's rotation field (see _prolong), then
    runs Anderson-accelerated ARAP until the estimated distance to the converged layout is < ``tol``
    voxels (see _Arap.run; at most ``iterations`` on the finest level, else info["converged"] is
    False and info["remaining"] holds the estimate).  Measured at the defaults against tol <= 1e-7
    references: exactly developable phantoms are recovered to <= 0.0052 vx rms of the true unrolling
    with no flipped triangles; a real 373k-vertex Scroll 1 grid to 1e-6 vx (17 s direct); the same
    grid with 40 holes punched in it to 0.4-2.7 tol.
    Validated regime (the limits are measured, not guessed): real Scroll 1 tifxyz grids have a
    per-triangle max angle with p99 ~94 deg, 0.1% of triangles above 100 deg and < 0.02% above
    120 deg; phantom grids with that profile (or a 2-5x heavier tail) and with uniform tangential
    jitter up to +-0.45 vx (max angle > 120 deg for 12-18% of the triangles) flatten without flips
    relative to the true unrolling, within 0.009 vx of it (+-0.35 vx may stop at the iteration cap with
    converged False: it contracts ~0.2% per iteration).  Uniform jitter of +-0.6 vx (28% above 120 deg)
    is beyond it: 72 flips, 0.9 vx max from the truth, not converged.
    That is a limit of the method on such grids, not a defect to chase; flatten_report exposes it.
    init="grid" starts from the grid layout; init="lscm" from conformal LSCM (single level only).
    iterations=0 returns the initial map.
    Levels above ``direct_vertices`` solve the global step by multigrid-preconditioned CG instead of
    a factorisation (memory O(n)): the full-resolution 6M-vertex Scroll 1 grid flattened in 192-207 s
    at 8.0 GB peak RSS, within 0.0033 vx (max) of a run to tol=1e-4.
    ``downsample`` > 1 flattens a strided sub-grid and bilinearly upsamples the UV field.
    ``init_uv`` (H, W, 2): start the coarsest level from this layout instead of the grid layout (the result is
    then rigidly aligned to it, not to the grid).  Soft anchors: ``anchor_mask`` (H, W) bool and ``anchor_uv``
    (H, W, 2) add anchor_weight * vertex_weight * |uv - anchor_uv|^2 per masked vertex (``vertex_weight`` (H, W),
    default 1) to the ARAP energy - a diagonal plus a right-hand side in every global step, at every level
    (scaled by stride^2 on coarse levels); with anchors no vertex is pinned, the result stays in the anchors'
    frame (no alignment) and every connected piece containing an anchored vertex is flattened (otherwise only
    the largest piece is).  With all of these left at their defaults the result is bit-identical to
    the plain flatten.
    ``info``, if given, receives per level: vertices, iterations, final energy, step and estimated
    remaining distance, CG iterations and the (energy, step) history of every iteration.
    """
    work = surface
    anchored = anchor_mask is not None and anchor_uv is not None and np.any(anchor_mask)
    if anchored:
        lam0 = anchor_weight * np.asarray(anchor_mask, np.float64)
        if vertex_weight is not None:
            lam0 = lam0 * np.asarray(vertex_weight, np.float64)
        auv0 = np.asarray(anchor_uv, np.float64)
    iuv0 = None if init_uv is None else np.asarray(init_uv, np.float64)
    mirror = False
    if anchored or iuv0 is not None:
        # ARAP rotations cannot reflect: if the given layout is mirrored w.r.t. the triangles' 3D winding,
        # solve in the mirrored frame (flip v) and flip the result back
        ref = iuv0 if iuv0 is not None else auv0
        idx_, V_, T_ = surface.triangles()
        T_ = T_[::max(1, len(T_) // 200_000)]  # the median sign needs a sample, not 12M Jacobians
        if len(T_):
            u_ = ref[idx_ >= 0]
            okt = np.all(np.isfinite(u_[T_]).reshape(len(T_), -1), axis=1)
            if okt.any():
                _, det_ = triangle_singular_values(V_, T_[okt], u_)
                mirror = bool(np.median(det_) < 0)
        del idx_, V_, T_, ref  # ~0.6 GB at 6M vertices: not held through the solve
        u_ = None
        if mirror:
            flipv = np.array([1.0, -1.0])
            if iuv0 is not None:
                iuv0 = iuv0 * flipv
            if anchored:
                auv0 = auv0 * flipv
    if downsample > 1:
        work = Surface(surface.points[::downsample, ::downsample],
                       (surface.scale[0] / downsample, surface.scale[1] / downsample))
        if anchored:
            lam0, auv0 = lam0[::downsample, ::downsample], auv0[::downsample, ::downsample]
        if iuv0 is not None:
            iuv0 = iuv0[::downsample, ::downsample]
    strides = [1]
    if multires and init == "grid" and iterations > 0:
        nv = int(work.valid.sum())
        while nv / strides[-1] ** 2 > coarse_vertices and min(work.shape) // (strides[-1] * 2) >= 8:
            strides.append(strides[-1] * 2)
    strides = strides[::-1]

    # below an iterative finest level only small levels are factorised: a deeper V-cycle with a 93k
    # bottom took the 6M-vertex grid from 241 s to 192 s (same iterations, energy within 1e-9)
    nv = int(work.valid.sum())
    limit = direct_vertices if nv <= direct_vertices else direct_vertices // 4

    def setup(st, coarser=None):  # independent of the coarser solutions: built in parallel
        lvl = Surface(work.points[::st, ::st], (work.scale[0] / st, work.scale[1] / st)) if st > 1 else work
        idx, V, T = lvl.triangles()
        if len(T) == 0:
            raise ValueError("surface has no valid quads")
        anc = None
        if anchored:  # coarse levels carry 1/st^2 of the vertices: keep the anchors' total pull
            anc = (lam0[::st, ::st][idx >= 0] * st**2, auv0[::st, ::st][idx >= 0])
        T = _clean(V, T, keep=None if anc is None else anc[0] > 0)  # anchored: every anchored piece is kept
        rr, cc = np.nonzero(idx >= 0)
        A = None
        if iterations > 0:
            if coarser is not None and len(V) > limit:  # multigrid on the coarser level's operator
                c = coarser.result()
                A = _Arap(V, T, rr, cc, coarse=(c[-1], c[1]), flip=_grid_folds(lvl, idx, V, T), anchor=anc)
            else:
                A = _Arap(V, T, rr, cc, flip=_grid_folds(lvl, idx, V, T), anchor=anc)
        return lvl, idx, V, T, rr, cc, A

    levels, cells = [], None
    with ThreadPoolExecutor(len(strides)) as pool:
        jobs = {}
        for i, st in enumerate(strides):
            jobs[st] = pool.submit(setup, st, jobs[strides[i - 1]] if i else None)
        for i, st in enumerate(strides):
            lvl, idx, V, T, rr, cc, A = jobs[st].result()
            finest = i == len(strides) - 1
            energy, its, step, remaining = None, 0, None, None
            if cells is None:
                if iuv0 is not None:
                    uv = iuv0[::st, ::st][idx >= 0].copy()
                    if not np.all(np.isfinite(uv[np.bincount(T.ravel(), minlength=len(V)) > 0])):
                        raise ValueError("init_uv must be finite at every valid vertex")
                    uv = np.nan_to_num(uv)
                    if anchored:
                        iuv0 = None  # the anchors fix the frame: the start layout is not needed again
                else:
                    uv = lscm(V, T) if init == "lscm" else grid_init(idx, V, T, lvl.scale)
            if A is not None:
                if cells is not None:
                    guess = None
                    if A.iterative:  # warm start: the coarser layout, upsampled, shifted to the pin
                        up = bilerp_grid(grid, np.arange(idx.shape[0]) / 2, np.arange(idx.shape[1]) / 2)[idx >= 0]
                        guess = (up[A.order] - up[A.order[-1]])[:-1] if A.pinned else up[A.order]
                    th = _prolong(cells, strides[i - 1] // st, rr, cc, T, A.x, A.y)
                    A.x = A.y = None
                    uv = A.to_mesh(A.solve(th, 0.0, guess), len(V))
                # coarser levels only seed the rotation field, so the intermediate ones are solved loosely
                n, tl = (iterations, tol) if finest else ((300, 1e-3) if i == 0 else (50, 0.05))
                uv, energy, its, remaining = A.run(uv, n, tl, log, window=16 if finest else None)
                step = A.history[-1][1] if A.history else None
            used = np.bincount(T.ravel(), minlength=len(V)) > 0
            uv[~used] = np.nan
            grid = np.full(idx.shape + (2,), np.nan)
            grid[idx >= 0] = uv
            levels.append({"stride": st, "vertices": int(used.sum()), "iterations": its, "energy": energy,
                           "step": step, "remaining": remaining,
                           "cg_iterations": A.cg_iterations if A is not None and A.iterative else None,
                           "history": A.history if A is not None else []})
            if i:
                jobs[strides[i - 1]] = None  # a level is only needed by the next finer one
            if log:
                log(f"flatten level stride {st}: {int(used.sum())} vertices, {its} iterations, ARAP energy {energy}")
            if not finest:
                cells = _cell_maps(lvl.points, grid)
                if A is not None:
                    A.release()
    if info is not None:
        info["levels"] = levels
        info["energy"] = levels[-1]["energy"]
        info["converged"] = iterations == 0 or levels[-1]["remaining"] < tol
        info["remaining"] = None if iterations == 0 else levels[-1]["remaining"]
    if anchored:
        pass  # the anchors fix the frame
    elif iuv0 is not None:
        grid = _align_to(grid, iuv0)
    else:
        grid = _align_to_grid(grid, work.scale)
    if mirror:
        grid = grid * np.array([1.0, -1.0])
    if downsample > 1:
        H, W = surface.shape
        grid = bilerp_grid(grid, np.arange(H) / downsample, np.arange(W) / downsample)
        grid[~surface.valid] = np.nan
    return grid


def _align_to(uv, ref):
    """Rigidly (rotation + translation, no reflection) align the UV grid to a reference layout ``ref``."""
    ok = np.isfinite(uv[..., 0]) & np.isfinite(ref[..., 0])
    a, b = uv[ok], ref[ok]
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    out = np.full_like(uv, np.nan)
    m = np.isfinite(uv[..., 0])
    out[m] = (uv[m] - ca) @ R.T + cb
    return out


def _align_to_grid(uv, scale):
    """Rigidly rotate/translate UV so it best matches nominal grid coords (col/sx, row/sy).

    Keeps the output in the same orientation as the input segment (like VC3D's alignment).
    A mirrored solution is un-mirrored, so text is never rendered reversed by the flattener.
    """
    H, W = uv.shape[:2]
    rr, cc = np.mgrid[0:H, 0:W]
    nom = np.stack([cc / scale[0], rr / scale[1]], -1)
    ok = np.isfinite(uv[..., 0])
    a, b = uv[ok], nom[ok]
    ca, cb = a.mean(0), b.mean(0)
    Mx = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(Mx)
    if np.linalg.det(U @ Vt) < 0:  # LSCM/ARAP solution is mirrored w.r.t. the grid -> flip v
        uv = uv * np.array([1.0, -1.0])
        a = uv[ok]
        ca = a.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    R = (U @ Vt).T
    out = np.full_like(uv, np.nan)
    out[ok] = (a - ca) @ R.T + cb
    return out


def triangle_singular_values(V, T, uv, flip=None):
    """Singular values (s1>=s2) of the 3D->2D Jacobian of each triangle, and signed UV area
    (``flip``: grid-folded triangles, measured in their mirrored frame)."""
    q = _local_frames(V, T, flip)
    E3 = np.stack([q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]], 2)  # (T,2,2) columns = edges
    E2 = np.stack([uv[T[:, 1]] - uv[T[:, 0]], uv[T[:, 2]] - uv[T[:, 0]]], 2)
    J = E2 @ np.linalg.inv(E3)
    s = np.linalg.svd(J, compute_uv=False)
    return s, np.linalg.det(J)
