"""Exact integer L1 winding assignment (WP3).

    min_{x in Z^n}  sum_e w_e |x_j - x_i - d_e|  +  sum_i u_i |x_i - pi_i|,   x fixed on anchor nodes.

The objective is L-natural convex, so steepest descent with +-1 moves on a node set S, each the minimum
cut of one max-flow, is exact: when neither the best +1 nor the best -1 move improves the energy, x is
optimal (Murota, Discrete Convex Analysis; the number of iterations is bounded by the L-inf distance of
the start from the nearest optimum, so a good warm start makes it fast).

Move graph (sign s, S = nodes moved by s), per edge with residual r = x_j - x_i - d:
    j in S, i not:  w (|r + s| - |r|) = B;   i in S, j not:  w (|r - s| - |r|) = C
    = C [i in S] - C [j in S] + (B + C) [j in S, i not],   B + C = 2w if r == 0 else 0  (only satisfied edges)
Unary terms (priors, and edges to anchors, which are eliminated into unary terms) add u (|x+s-p| - |x-p|).

Hardening: weights are quantised to integers with a per-solve scale chosen so that every node's
source + sink + incident capacities stay < 2^31 - 1 (scipy's max-flow silently returns 0 for larger
int64 capacities); every cut is cross-checked (energy change of the returned set, computed directly in
int64, must equal max-flow minus the sink constant), so a silently failed max-flow raises.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow, minimum_spanning_tree

from scipy.spatial import cKDTree

INT_LIMIT = 2**31 - 1


# --------------------------------------------------------------------------------------- data contracts
# Mirrors of WP0's contracts (unroller/ws/types.py at merge time).

SAME = 0  # edge kind: both ends on the same sheet (d is normally 0)


@dataclass
class Edges:
    """Observed winding offsets: x[j] - x[i] ~ d, with weight w (count observations, kind, quality bin, location)."""
    i: np.ndarray  # int64
    j: np.ndarray  # int64
    d: np.ndarray  # int32
    w: np.ndarray  # float32
    count: np.ndarray = None  # int32
    kind: np.ndarray = None  # uint8
    qbin: np.ndarray = None  # uint8
    xyz: np.ndarray = None  # (m, 3) float32

    def __post_init__(self):
        m = len(self.i)
        self.i = np.asarray(self.i, np.int64)
        self.j = np.asarray(self.j, np.int64)
        self.d = np.asarray(self.d, np.int32)
        self.w = np.asarray(self.w, np.float32)
        self.count = np.ones(m, np.int32) if self.count is None else np.asarray(self.count, np.int32)
        self.kind = np.zeros(m, np.uint8) if self.kind is None else np.asarray(self.kind, np.uint8)
        self.qbin = np.zeros(m, np.uint8) if self.qbin is None else np.asarray(self.qbin, np.uint8)
        self.xyz = np.zeros((m, 3), np.float32) if self.xyz is None else np.asarray(self.xyz, np.float32)
        for a in (self.j, self.d, self.w, self.count, self.kind, self.qbin):
            if len(a) != m:
                raise ValueError("Edges arrays must have the same length")

    def __len__(self):
        return len(self.i)

    def subset(self, m) -> "Edges":
        return Edges(self.i[m], self.j[m], self.d[m], self.w[m], self.count[m], self.kind[m], self.qbin[m], self.xyz[m])


# Solution.status
OK, DROPPED, UNCERTIFIED, NO_EDGES = 0, 1, 2, 3
# Solution.flags (bits)
FLAG_LOW_CONF, FLAG_UNCORROBORATED, FLAG_FIXED, FLAG_TIE = 1, 2, 4, 8  # TIE: a zero-cost +-1 move contains it


@dataclass
class Solution:
    ids: np.ndarray  # int64 node ids
    x: np.ndarray  # int32 winding
    supernode: np.ndarray  # int64 (-1: dropped)
    block: np.ndarray  # int64
    comp: np.ndarray  # int32 connected component
    conf: np.ndarray  # float32 = min(c1, cblk)
    c1: np.ndarray  # float32
    cblk: np.ndarray  # float32
    corroborated: np.ndarray  # bool
    status: np.ndarray  # uint8
    flags: np.ndarray  # uint32
    meta: dict = field(default_factory=dict)  # chi, chi_ratio, energy, certificate, stopped_by, iters, scale, ...


# ------------------------------------------------------------------------------------------------- problem
class Problem:
    """Free-node problem after eliminating anchors: pair edges (pi, pj, pd, pw) between free nodes (local
    indices) and unary terms (uv, up, uw): uw |x[uv] - up|; plus a constant energy."""

    def __init__(self, e: Edges, n: int, fixed=None, prior=None):
        self.n = n
        isfix = np.zeros(n, bool)
        fval = np.zeros(n, np.int64)
        if fixed is not None:
            fid, fv = (np.asarray(fixed[0], np.int64), np.asarray(fixed[1], np.int64)) if not isinstance(fixed, dict) \
                else (np.fromiter(fixed.keys(), np.int64), np.fromiter(fixed.values(), np.int64))
            isfix[fid] = True
            fval[fid] = fv
        self.isfix, self.fval = isfix, fval
        i, j, d, w = e.i, e.j, e.d.astype(np.int64), e.w.astype(np.float64)
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("edge weights must be finite and >= 0")
        keep = w > 0
        i, j, d, w = i[keep], j[keep], d[keep], w[keep]
        fi, fj = isfix[i], isfix[j]
        loop = i == j
        both = (fi & fj) | loop
        self.const = float(np.sum(w[both] * np.abs(np.where(loop[both], 0, fval[j[both]] - fval[i[both]]) - d[both])))
        free = ~isfix
        self.free = np.flatnonzero(free)
        loc = np.full(n, -1, np.int64)
        loc[self.free] = np.arange(len(self.free))
        self.loc = loc
        pm = ~fi & ~fj & ~loop
        self.pi, self.pj, self.pd, self.pw = loc[i[pm]], loc[j[pm]], d[pm], w[pm]
        # i fixed, j free: w |x_j - (x_i + d)|;  j fixed, i free: w |x_i - (x_j - d)|
        a = fi & ~fj & ~loop
        b = ~fi & fj & ~loop
        uv = [loc[j[a]], loc[i[b]]]
        up = [fval[i[a]] + d[a], fval[j[b]] - d[b]]
        uw = [w[a], w[b]]
        if prior is not None:
            pid, ppi, pu = (np.asarray(p) for p in prior)
            pid, ppi, pu = pid.astype(np.int64), ppi.astype(np.int64), pu.astype(np.float64)
            pf = isfix[pid]
            self.const += float(np.sum(pu[pf] * np.abs(fval[pid[pf]] - ppi[pf])))
            uv.append(loc[pid[~pf]]); up.append(ppi[~pf]); uw.append(pu[~pf])
        self.uv, self.up, self.uw = np.concatenate(uv), np.concatenate(up).astype(np.int64), np.concatenate(uw)
        keep = self.uw > 0
        self.uv, self.up, self.uw = self.uv[keep], self.up[keep], self.uw[keep]
        self.nf = len(self.free)
        self.quantise()

    def quantise(self, scale=None, max_scale=1e6):
        """Integer weights: wi = round(w * scale), scale chosen so every node's capacities sum < 2^31 - 1."""
        nf = self.nf
        load = 3 * (np.bincount(self.pi, self.pw, nf) + np.bincount(self.pj, self.pw, nf)) \
            + np.bincount(self.uv, self.uw, nf)
        lmax = float(load.max()) if nf and len(load) else 0.0
        if scale is None:
            scale = max_scale if lmax == 0 else min(max_scale, (INT_LIMIT - 1) / (lmax * 1.0001 + 3))
        while True:
            if scale < 1:
                raise OverflowError(f"weights too large to quantise below 2^31 per node (max node load {lmax:.3g})")
            wi = np.maximum(1, np.rint(self.pw * scale)).astype(np.int64)
            ui = np.maximum(1, np.rint(self.uw * scale)).astype(np.int64)
            iload = 3 * (np.bincount(self.pi, wi, nf) + np.bincount(self.pj, wi, nf)) + np.bincount(self.uv, ui, nf)
            if not nf or iload.max() < INT_LIMIT:
                break
            scale /= 2  # max(1, .) bumps or rounding pushed a node over: try a smaller scale
        self.scale, self.wi, self.ui = float(scale), wi, ui
        self.max_node_cap = int(iload.max()) if nf else 0
        assert self.max_node_cap < INT_LIMIT
        self.dw = np.abs(self.pw * scale - wi) / scale  # quantisation error per edge (weight units)
        self.du = np.abs(self.uw * scale - ui) / scale
        self.quant_err = float(self.dw.sum() + self.du.sum())

    # energies on free-node vectors (int64)
    def residual(self, xf):
        return xf[self.pj] - xf[self.pi] - self.pd

    def energy_int(self, xf) -> int:
        return int(np.sum(self.wi * np.abs(self.residual(xf))) + np.sum(self.ui * np.abs(xf[self.uv] - self.up)))

    def energy(self, xf) -> float:
        return float(np.sum(self.pw * np.abs(self.residual(xf))) + np.sum(self.uw * np.abs(xf[self.uv] - self.up))) \
            + self.const

    def quant_bound(self, xf) -> float:
        """|E_real(x) - E_int(x) / scale| <= this (so the real-weight energy of the certified x is within
        2 * bound of the real-weight optimum)."""
        return float(np.sum(self.dw * np.abs(self.residual(xf))) + np.sum(self.du * np.abs(xf[self.uv] - self.up)))

    def full(self, xf):
        x = self.fval.copy()
        x[self.free] = xf
        return x


# --------------------------------------------------------------------------------------------------- moves
def _maxflow_scipy(G, s, t):
    f = maximum_flow(G, s, t, method="dinic")
    res = (G - f.flow).tocsr()
    res.data[res.data < 0] = 0
    res.eliminate_zeros()
    reach = breadth_first_order(res, s, directed=True, return_predecessors=False)
    side = np.zeros(G.shape[0], bool)
    side[reach] = True
    return int(f.flow_value), side


def _maxflow_bk(n, src, snk, a, b, cap):
    import maxflow  # PyMaxflow (Boykov-Kolmogorov)

    g = maxflow.Graph[int](n, len(a))
    ids = g.add_nodes(n)
    g.add_grid_tedges(ids, src, snk)
    if len(a):
        g.add_edges(a, b, cap, np.zeros_like(cap))
    flow = g.maxflow()
    sink = g.get_grid_segments(ids)  # True = sink side
    side = np.zeros(n + 2, bool)
    side[:n] = ~sink
    side[n] = True
    return int(flow), side


def best_move(P: Problem, xf, sign: int, backend="scipy", force=None):
    """The minimising set S for moving by ``sign`` and its exact energy change (int64, <= 0).
    ``force`` (local node index): S must contain this node (for exact flip margins)."""
    n = P.nf
    s, t = n, n + 1
    r = P.residual(xf)
    C = P.wi * (np.abs(r - sign) - np.abs(r))
    sat = r == 0  # B + C = 2 w on satisfied edges, 0 elsewhere
    u1 = (np.bincount(P.pi, C, n) - np.bincount(P.pj, C, n)).astype(np.float64)  # exact: |sums| < 2^31
    if len(P.uv):
        xu = xf[P.uv]
        u1 += np.bincount(P.uv, P.ui * (np.abs(xu + sign - P.up) - np.abs(xu - P.up)), n)
    u1 = np.rint(u1).astype(np.int64)
    src = np.maximum(u1, 0)
    snk = np.maximum(-u1, 0)
    if force is not None:  # S is the sink side: tie the node to the sink with a (practically) infinite arc
        snk = snk.copy()
        snk[force] = INT_LIMIT - 2
    a, b, cap = P.pi[sat], P.pj[sat], 2 * P.wi[sat]
    N = int(snk.sum())
    if backend == "bk":
        flow, side = _maxflow_bk(n, src, snk, a, b, cap)
    else:
        vs, vt = np.flatnonzero(src), np.flatnonzero(snk)
        R = np.concatenate([np.full(len(vs), s), vt, a])
        Cc = np.concatenate([vs, np.full(len(vt), t), b])
        K = np.concatenate([src[vs], snk[vt], cap])
        G = sp.csr_matrix((K, (R, Cc)), shape=(n + 2, n + 2))
        G.sum_duplicates()
        if G.nnz and G.data.max() >= INT_LIMIT:
            raise OverflowError("a move-graph capacity reached 2^31 - 1")
        G.data = G.data.astype(np.int32)
        if G.nnz == 0:
            return np.zeros(n, bool), 0
        flow, side = _maxflow_scipy(G, s, t)
    S = ~side[:n]
    if force is not None and not S[force]:
        raise RuntimeError("forced node left out of the move set (sink arc not infinite enough)")
    # exact energy change of S, independently of the flow: sum_{S} u1 + sum_{satisfied, i not in S, j in S} 2w
    delta = int(u1[S].sum()) + int(cap[~S[a] & S[b]].sum())
    if force is None and delta != flow - N:
        raise RuntimeError(f"max-flow inconsistent (cut from S {delta}, flow - N {flow - N}): capacity overflow?")
    return S, delta


# ---------------------------------------------------------------------------------------------- initialisation
def tree_init(P: Problem):
    """Propagate offsets along a maximum-weight spanning forest (heaviest edge per node pair), then shift each
    component to the median of its unary targets."""
    n = P.nf
    x = np.zeros(n, np.int64)
    if n == 0:
        return x
    if len(P.pi) == 0:  # only unary terms (every edge reaches an anchor)
        return _shift_to_unary(P, x, np.arange(n), n)
    a, b = np.minimum(P.pi, P.pj), np.maximum(P.pi, P.pj)
    dd = np.where(P.pi < P.pj, P.pd, -P.pd)  # offset from a to b
    key = a * n + b
    o = np.lexsort((-P.pw, key))
    first = np.r_[True, key[o][1:] != key[o][:-1]]
    sel = o[first]
    a, b, dd, ww = a[sel], b[sel], dd[sel], P.pw[sel]
    G = sp.csr_matrix((ww.max() + 1.0 - ww, (a, b)), shape=(n, n))
    T = minimum_spanning_tree(G)
    T = (T + T.T).tocsr()
    ncomp, lab = connected_components(T, directed=False)
    kd = {}
    ksorted = a * n + b
    ordk = np.argsort(ksorted)
    ks = ksorted[ordk]
    for c in range(ncomp):
        root = int(np.flatnonzero(lab == c)[0])
        order, pred = breadth_first_order(T, root, directed=False, return_predecessors=True)
        if len(order) < 2:
            continue
        v = order[1:]
        u = pred[v]
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        pos = ordk[np.searchsorted(ks, lo * n + hi)]
        step = np.where(u < v, dd[pos], -dd[pos])  # x[v] - x[u]
        for vv, uu, st in zip(v.tolist(), u.tolist(), step.tolist()):
            x[vv] = x[uu] + st
    return _shift_to_unary(P, x, lab, ncomp)


def _shift_to_unary(P, x, lab=None, ncomp=None):
    if lab is None:
        ncomp, lab = _components(P)
    if len(P.uv):
        for c in np.unique(lab[P.uv]):
            m = lab[P.uv] == c
            x[lab == c] += int(np.round(_wmedian(P.up[m] - x[P.uv[m]], P.uw[m])))
    return x


def _wmedian(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return v[o][np.searchsorted(cw, cw[-1] / 2)]


def _components(P):
    G = sp.csr_matrix((np.ones(len(P.pi)), (P.pi, P.pj)), shape=(P.nf, P.nf))
    return connected_components(G, directed=False)


def irls_init(P: Problem, iters=8, eps=0.1, x0=None):
    """Continuous L1 solution by iteratively reweighted least squares, rounded."""
    from scipy.sparse.linalg import cg, spsolve

    n = P.nf
    x = (tree_init(P) if x0 is None else x0).astype(np.float64)
    ncomp, lab = _components(P)
    for _ in range(iters):
        r = x[P.pj] - x[P.pi] - P.pd
        q = P.pw / np.maximum(np.abs(r), eps)
        L = sp.csr_matrix((np.r_[-q, -q], (np.r_[P.pi, P.pj], np.r_[P.pj, P.pi])), shape=(n, n))
        diag = np.bincount(P.pi, q, n) + np.bincount(P.pj, q, n)
        rhs = np.bincount(P.pj, q * P.pd, n) - np.bincount(P.pi, q * P.pd, n)
        if len(P.uv):
            qu = P.uw / np.maximum(np.abs(x[P.uv] - P.up), eps)
            diag += np.bincount(P.uv, qu, n)
            rhs += np.bincount(P.uv, qu * P.up, n)
        ridge = 1e-6 * (diag.mean() if n else 1.0)  # gauge of components without unary terms
        A = (L + sp.diags(diag + ridge)).tocsr()
        rhs = rhs + ridge * x
        if n <= 200_000:
            x = spsolve(A.tocsc(), rhs)
        else:
            M = sp.diags(1.0 / A.diagonal())
            x, _ = cg(A, rhs, x0=x, M=M, maxiter=500, rtol=1e-6)
    return _shift_to_unary(P, np.rint(x).astype(np.int64), lab, ncomp)


# ------------------------------------------------------------------------------------------------------- solve
def solve(e: Edges, n: int, x0=None, fixed=None, prior=None, max_iter: int = 1000, init: str = "tree",
          scale=None, backend: str = "scipy", log=None):
    """Exact minimiser of the L1 winding energy.  Returns (x int64 (n,), info).

    ``x0``: warm start (full length n; fixed entries ignored); else ``init`` in {'tree', 'irls', 'zero'}.
    ``fixed``: (ids, values) or {id: value} anchors; ``prior``: (ids, pi, u) unary terms u |x - pi|.
    info: energy (real weights, incl. constants), energy_int, init_energy, certificate (bool), stopped_by
    ('certificate' | 'max_iter'), iters, moves, scale, quant_err, quant_bound, max_node_cap, backend, wall_s.
    """
    t0 = time.perf_counter()
    P = Problem(e, n, fixed, prior)
    if scale is not None:
        P.quantise(scale)
    if x0 is not None:
        xf = np.asarray(x0, np.int64)[P.free].copy()
    elif init == "irls":
        xf = irls_init(P)
    elif init == "zero":
        xf = np.zeros(P.nf, np.int64)
    else:
        xf = tree_init(P)
    t_init = time.perf_counter() - t0
    e_init = P.energy(xf)
    it, moves, stopped, cert = 0, 0, "max_iter", False
    E = P.energy_int(xf)
    while it < max_iter:
        it += 1
        moved = False
        for sign in (1, -1):
            S, delta = best_move(P, xf, sign, backend)
            if delta < 0:
                xf[S] += sign
                E2 = P.energy_int(xf)
                if E2 - E != delta:
                    raise RuntimeError(f"move energy mismatch ({E2 - E} vs {delta})")
                E = E2
                moves += 1
                moved = True
        if log:
            log(f"iter {it}: energy_int {E}, moved {moved}")
        if not moved:
            stopped, cert = "certificate", True
            break
    info = {"energy": P.energy(xf), "energy_int": E, "init_energy": e_init, "certificate": cert,
            "stopped_by": stopped, "iters": it, "moves": moves, "scale": P.scale, "quant_err": P.quant_err,
            "quant_bound": P.quant_bound(xf), "max_node_cap": P.max_node_cap, "backend": backend,
            "n_free": P.nf, "n_edges": len(P.pi), "n_unary": len(P.uv), "init_s": t_init,
            "wall_s": time.perf_counter() - t0}
    return P.full(xf), info


def energy(x, e: Edges, prior=None) -> float:
    x = np.asarray(x, np.int64)
    E = float(np.sum(e.w.astype(np.float64) * np.abs(x[e.j] - x[e.i] - e.d)))
    if prior is not None:
        pid, ppi, pu = prior
        E += float(np.sum(np.asarray(pu, np.float64) * np.abs(x[np.asarray(pid)] - np.asarray(ppi))))
    return E


# --------------------------------------------------------------------------------------------------- contract
def _offset_forest(n, a, b, dd, w=None):
    """Components of the graph (a, b) and per-node offsets o with o[b] - o[a] = dd along a spanning forest
    (heaviest edges first if w given).  Returns (comp, offset)."""
    if len(a) == 0:
        return np.arange(n), np.zeros(n, np.int64)
    ww = np.ones(len(a)) if w is None else np.asarray(w, np.float64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    dlh = np.where(a < b, dd, -dd)
    key = lo * n + hi
    o = np.lexsort((-ww, key))
    first = np.r_[True, key[o][1:] != key[o][:-1]]
    sel = o[first]
    lo, hi, dlh, ww = lo[sel], hi[sel], dlh[sel], ww[sel]
    G = sp.csr_matrix((ww.max() + 1.0 - ww, (lo, hi)), shape=(n, n))
    T = minimum_spanning_tree(G)
    T = (T + T.T).tocsr()
    ncomp, comp = connected_components(T, directed=False)
    off = np.zeros(n, np.int64)
    ks = lo * n + hi
    ordk = np.argsort(ks)
    kss = ks[ordk]
    seen = np.zeros(ncomp, bool)
    roots = []
    for v in range(n):  # first node of each component
        if not seen[comp[v]]:
            seen[comp[v]] = True
            roots.append(v)
    for root in roots:
        order, pred = breadth_first_order(T, root, directed=False, return_predecessors=True)
        if len(order) < 2:
            continue
        v = order[1:]
        u = pred[v]
        l2, h2 = np.minimum(u, v), np.maximum(u, v)
        pos = ordk[np.searchsorted(kss, l2 * n + h2)]
        step = np.where(u < v, dlh[pos], -dlh[pos])
        for vv, uu, st in zip(v.tolist(), u.tolist(), step.tolist()):
            off[vv] = off[uu] + st
    return comp, off


def aggregate(si, sj, sd, w, count, kind, xyz, n):
    """Merge parallel rows with the same (a, b, d) (orientation-normalised): w and count summed, kind = max
    (0 only if every row was SAME), xyz weight-averaged."""
    a, b = np.minimum(si, sj), np.maximum(si, sj)
    dd = np.where(si < sj, sd, -sd).astype(np.int64)
    key = np.stack([a, b, dd], 1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    m = len(uk)
    W = np.bincount(inv, w, m)
    Cn = np.bincount(inv, count, m)
    K = np.zeros(m, np.uint8)
    np.maximum.at(K, inv, kind)
    X = np.stack([np.bincount(inv, w * xyz[:, k], m) for k in range(3)], 1) / np.maximum(W, 1e-12)[:, None]
    return Edges(uk[:, 0], uk[:, 1], uk[:, 2], W, Cn, K, None, X)


def contract(e: Edges, sizes, min_size=30, min_count=3):
    """Merge nodes joined by strong SAME edges (kind 0, count >= min_count, a single d for the pair) into
    supernodes with fixed internal offsets, drop supernodes smaller than ``min_size`` (total of ``sizes``).

    Returns (super_of_node (-1 = dropped), super_sizes, super_edges, report); node i's winding is
    x[i] = X[super_of_node[i]] + report['offset'][i].  report['internal_residual'] is the weighted L1 residual
    of the edges inside supernodes under the offsets (energy that the contraction freezes in)."""
    sizes = np.asarray(sizes, np.float64)
    n = len(sizes)
    lo, hi = np.minimum(e.i, e.j), np.maximum(e.i, e.j)
    dlh = np.where(e.i < e.j, e.d, -e.d).astype(np.int64)
    same = (e.kind == SAME) & (e.count >= min_count) & (e.i != e.j)
    key = lo * n + hi
    # pairs whose strong SAME rows disagree on d are not merged
    ks, dk, wk = key[same], dlh[same], e.w[same].astype(np.float64)
    o = np.lexsort((dk, ks))
    ks, dk, wk = ks[o], dk[o], wk[o]
    newk = np.r_[True, ks[1:] != ks[:-1]] if len(ks) else np.zeros(0, bool)
    starts = np.flatnonzero(newk)
    grp = np.cumsum(newk) - 1
    dmin = np.minimum.reduceat(dk, starts) if len(dk) else dk
    dmax = np.maximum.reduceat(dk, starts) if len(dk) else dk
    single = dmin == dmax
    good_keys = ks[starts][single]
    good_d = dmin[single]
    a, b = good_keys // n, good_keys % n
    wsum = np.bincount(grp, wk, len(starts))[single] if len(dk) else wk
    comp, off = _offset_forest(n, a, b, good_d, wsum)
    _, comp = np.unique(comp, return_inverse=True)
    ncomp = comp.max() + 1 if n else 0
    ssize = np.bincount(comp, sizes, ncomp)
    keep = ssize >= min_size
    newid = np.full(ncomp, -1, np.int64)
    newid[keep] = np.arange(keep.sum())
    sup = newid[comp]
    # residual frozen inside supernodes
    si, sj = sup[e.i], sup[e.j]
    inside = (si >= 0) & (si == sj)
    r_in = off[e.j[inside]] - off[e.i[inside]] - e.d[inside]
    ok = (si >= 0) & (sj >= 0) & (si != sj)
    sd = e.d[ok].astype(np.int64) + off[e.i[ok]] - off[e.j[ok]]  # X_sj - X_si ~ d + o_i - o_j
    se = aggregate(si[ok], sj[ok], sd, e.w[ok].astype(np.float64), e.count[ok], e.kind[ok], e.xyz[ok], int(keep.sum()))
    report = {"nodes": n, "supernodes": int(keep.sum()), "merged_components": int(ncomp),
              "dropped_supernodes": int((~keep).sum()), "dropped_nodes": int((sup < 0).sum()),
              "dropped_size": float(ssize[~keep].sum()), "strong_pairs": int(len(good_keys)),
              "ambiguous_pairs": int((~single).sum()), "internal_edges": int(inside.sum()),
              "internal_residual": float(np.sum(e.w[inside] * np.abs(r_in))),
              "internal_unsatisfied": int(np.count_nonzero(r_in)), "edges_in": len(e), "super_edges": len(se),
              "dropped_edges": int(np.sum((si < 0) | (sj < 0))), "offset": off}
    return sup, ssize[keep], se, report


# ----------------------------------------------------------------------------------------------- slab pipeline
def solve_slabs(e: Edges, n: int, slab, fixed=None, prior=None, strong_w=0.0, strong_count=1, max_iter=1000,
                backend="scipy", slab_init="irls", centroid_xyz=None, blocks=(256, 512, 1024, 2048), log=None):
    """slab -> block -> global: solve each slab independently, contract each slab into blocks of satisfied
    strong edges (w >= strong_w and count >= strong_count), solve the block graph exactly, then polish the
    whole graph from the stitched start.  Returns a Solution (conf / corroboration filled when
    ``centroid_xyz`` is given)."""
    t0 = time.perf_counter()
    slab = np.asarray(slab)
    isfix = np.zeros(n, bool)
    fval = np.zeros(n, np.int64)
    if fixed is not None:
        fid, fv = (np.asarray(fixed[0], np.int64), np.asarray(fixed[1], np.int64))
        isfix[fid] = True
        fval[fid] = fv
    xs = np.zeros(n, np.int64)
    t_slab = 0.0
    slab_info = []
    for s in np.unique(slab):
        nodes = np.flatnonzero(slab == s)
        loc = np.full(n, -1, np.int64)
        loc[nodes] = np.arange(len(nodes))
        m = (loc[e.i] >= 0) & (loc[e.j] >= 0)
        sub = e.subset(m)
        sub = Edges(loc[sub.i], loc[sub.j], sub.d, sub.w, sub.count, sub.kind, sub.qbin, sub.xyz)
        sfix = None
        if fixed is not None:
            fm = isfix[nodes]
            sfix = (np.flatnonzero(fm), fval[nodes[fm]]) if fm.any() else None
        sprior = None
        if prior is not None:
            pm = loc[np.asarray(prior[0])] >= 0
            sprior = (loc[np.asarray(prior[0])[pm]], np.asarray(prior[1])[pm], np.asarray(prior[2])[pm])
        t1 = time.perf_counter()
        xl, inf = solve(sub, len(nodes), fixed=sfix, prior=sprior, max_iter=max_iter, init=slab_init, backend=backend)
        t_slab += time.perf_counter() - t1
        xs[nodes] = xl
        slab_info.append({"slab": int(s), "nodes": len(nodes), "certificate": inf["certificate"], "iters": inf["iters"]})
        if log:
            log(f"slab {s}: {len(nodes)} nodes, {inf['iters']} its, {inf['wall_s']:.1f} s")
    # blocks: satisfied strong edges inside a slab (fixed nodes stay singletons)
    r = xs[e.j] - xs[e.i] - e.d
    strong = (r == 0) & (slab[e.i] == slab[e.j]) & (e.w >= strong_w) & (e.count >= strong_count) \
        & ~isfix[e.i] & ~isfix[e.j] & (e.i != e.j)
    G = sp.csr_matrix((np.ones(int(strong.sum())), (e.i[strong], e.j[strong])), shape=(n, n))
    nb, block = connected_components(G, directed=False)
    # block graph: x_i = X_b + xs_i
    bi, bj = block[e.i], block[e.j]
    cross = bi != bj
    be = aggregate(bi[cross], bj[cross], e.d[cross].astype(np.int64) + xs[e.i[cross]] - xs[e.j[cross]],
                   e.w[cross].astype(np.float64), e.count[cross], e.kind[cross], e.xyz[cross], nb)
    bfix = None
    if fixed is not None and isfix.any():
        fid = np.flatnonzero(isfix)
        bfix = (block[fid], fval[fid] - xs[fid])
    bprior = None
    if prior is not None:
        pid = np.asarray(prior[0], np.int64)
        bprior = (block[pid], np.asarray(prior[1], np.int64) - xs[pid], np.asarray(prior[2]))
    t1 = time.perf_counter()
    X, binf = solve(be, nb, fixed=bfix, prior=bprior, max_iter=max_iter, init="tree", backend=backend)
    t_block = time.perf_counter() - t1
    stitched = X[block] + xs
    t1 = time.perf_counter()
    x, info = solve(e, n, x0=stitched, fixed=fixed, prior=prior, max_iter=max_iter, backend=backend, log=log)
    t_polish = time.perf_counter() - t1
    G2 = sp.csr_matrix((np.ones(len(e)), (e.i, e.j)), shape=(n, n))
    _, comp = connected_components(G2, directed=False)
    meta = dict(info)
    meta.update({"slabs": len(slab_info), "slab_all_certified": all(s["certificate"] for s in slab_info),
                 "blocks": int(nb), "block_edges": len(be), "block_certificate": binf["certificate"],
                 "block_iters": binf["iters"], "stitched_energy": energy(stitched, e, prior),
                 "slab_s": t_slab, "block_s": t_block, "polish_s": t_polish, "wall_s": time.perf_counter() - t0,
                 "polish_iters": info["iters"], "stitched_x": stitched.astype(np.int32)})
    status = np.where(isfix, OK, OK if info["certificate"] else UNCERTIFIED).astype(np.uint8)
    deg = np.bincount(e.i, minlength=n) + np.bincount(e.j, minlength=n)
    status[(deg == 0) & ~isfix] = NO_EDGES
    flags = np.where(isfix, FLAG_FIXED, 0).astype(np.uint32)
    sol = Solution(ids=np.arange(n), x=x.astype(np.int32), supernode=np.arange(n), block=block.astype(np.int64),
                   comp=comp.astype(np.int32), conf=np.zeros(n, np.float32), c1=np.zeros(n, np.float32),
                   cblk=np.zeros(n, np.float32), corroborated=np.zeros(n, bool), status=status, flags=flags,
                   meta=meta)
    if centroid_xyz is not None:
        from .confidence import corroboration, flip_margins
        c1, cb = flip_margins(x, e, centroid_xyz, blocks, fixed=fixed, prior=prior)
        sol.c1, sol.cblk = c1.astype(np.float32), cb.astype(np.float32)
        sol.conf = np.minimum(c1, cb).astype(np.float32)
        sol.corroborated = corroboration(x, e)
        if info["certificate"]:
            from .confidence import ties
            tie = ties(x, e, centroid_xyz, fixed, prior)
            sol.conf[tie] = 0
            sol.flags[tie] |= FLAG_TIE | FLAG_LOW_CONF
            meta["tie_share"] = float(tie.mean())
    return sol


# ------------------------------------------------------------------------------------------------- chirality
def choose_chirality(build, min_ratio=1.5, **kw):
    """``build(chi)`` -> (edges, n) or (edges, n, solve kwargs) for chi in {+1, -1}.  Solves both and keeps the
    lower-energy handedness.  Returns (chi, ratio = E(wrong) / E(right), info).

    The total-energy ratio is diluted by everything the handedness does not touch: only the edges whose offset
    differs between the two builds (e.g. sheets crossing the angular seam) carry chirality evidence, while noise
    and walls cost the same either way, so on large graphs ratio -> 1 even when the evidence is unanimous.
    When both builds have the same edge list, info['informative_ratio'] is the same ratio restricted to the
    informative edges (each build's energy at its own optimum) and info['decided'] uses it; otherwise the total
    ratio is used.  An undecided chirality must not be used."""
    res, builds = {}, {}
    for chi in (1, -1):
        b = build(chi)
        e, n = b[0], b[1]
        extra = dict(kw, **(b[2] if len(b) > 2 else {}))
        x, inf = solve(e, n, **extra)
        res[chi], builds[chi] = (x, inf), e
    E = {c: res[c][1]["energy"] for c in res}
    best = 1 if E[1] <= E[-1] else -1
    ratio = E[-best] / max(E[best], 1e-12)
    info = {"energy": E, "total_ratio": float(ratio), "certified": all(res[c][1]["certificate"] for c in res),
            "x": res[best][0], "info": res[best][1]}
    e1, e2 = builds[1], builds[-1]
    if len(e1) == len(e2) and np.array_equal(e1.i, e2.i) and np.array_equal(e1.j, e2.j):
        inf_m = e1.d != e2.d
        Ei = {c: float(np.sum(builds[c].w[inf_m].astype(np.float64) *
                              np.abs(res[c][0][builds[c].j[inf_m]] - res[c][0][builds[c].i[inf_m]] - builds[c].d[inf_m])))
              for c in (1, -1)}
        wsum = float(builds[1].w[inf_m].sum())
        info.update({"informative_edges": int(inf_m.sum()), "informative_energy": Ei,
                     "informative_ratio": (Ei[-best] + 1e-9 * wsum) / (Ei[best] + 1e-9 * wsum) if inf_m.any() else 1.0})
        if inf_m.any() and (Ei[-best] < Ei[best]):
            info["conflict"] = True  # the total energy and the informative edges disagree: do not decide
        info["decided"] = bool(info["informative_ratio"] >= min_ratio and not info.get("conflict", False))
    else:
        info["decided"] = bool(ratio >= min_ratio)
    return best, float(ratio), info


# ------------------------------------------------------------------------------- synthetic graphs (tests, bench)
def make(nx, ny, nz, p_out, seed=0, burst=False):
    """The WP3 prototype's lattice (verbatim semantics): truth t = row // 3, 6-neighbour edges, uniform weights
    0.5-1.5, a fraction p_out of edges off by +-1/+-2, optionally a systematic wall of wrong x-edges."""
    rng = np.random.default_rng(seed)
    idx = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    t = np.broadcast_to((np.arange(ny) // 3)[None, :, None], idx.shape).ravel().astype(np.int64)
    E = np.concatenate([np.stack([a.ravel(), b.ravel()], 1) for a, b in
                        [(idx[:, :, :-1], idx[:, :, 1:]), (idx[:, :-1, :], idx[:, 1:, :]), (idx[:-1], idx[1:])]])
    d = t[E[:, 1]] - t[E[:, 0]]
    w = rng.uniform(0.5, 1.5, len(E))
    bad = rng.random(len(E)) < p_out
    if burst:
        xa, xb = E[:, 0] % nx, E[:, 1] % nx
        za = E[:, 0] // (nx * ny)
        bad |= (xa != xb) & (np.minimum(xa, xb) == nx // 2 - 1) & (za < nz // 2)
    dobs = d.copy()
    dobs[bad] += rng.choice([-2, -1, 1, 2], bad.sum())
    return t, Edges(E[:, 0], E[:, 1], dobs, w), bad


def sheetlets(n, p_wrong=0.05, n_walls=8, wall_frac=0.6, k=8, seed=0, chi=1, pitch=12.0, r0=40.0,
              height=None, spacing=3.0, scale=1.0, obs=1.0, return_xyz=False):
    """Sheetlet-like winding graph: nodes are points on the sheets of an elliptical spiral scroll (truth = wrap
    index), edges join k nearest neighbours.  Same-wrap neighbours give SAME edges (d = 0, kind 0); neighbours on
    adjacent wraps give cross edges (d = +-1, kind 1) whose sign depends on the chirality ``chi`` (a mirrored scroll
    observed with the wrong handedness flips every cross offset).  Geometry in units of ``scale`` voxels: sheet pitch
    ``pitch``, in-sheet point spacing ~ ``spacing`` (the scroll height follows from n).  Errors: a fraction p_wrong of random edges off by
    +-1 (+-2 rarely), plus ``n_walls`` correlated walls - axial slabs of a sector in which a fraction wall_frac of
    the cross edges are systematically off by +1 (fused / skipped sheets)."""
    rng = np.random.default_rng(seed)
    turns = max(2.0, np.sqrt(n / 4000.0))
    smax = 2 * np.pi * turns
    # points uniform in arc length s(t) = r0 t + pitch t^2 / (4 pi): sample s, invert
    k_ = pitch / (2 * np.pi)
    s_max = r0 * smax + k_ * smax**2 / 2
    s = rng.uniform(0, s_max, n)
    t = (-r0 + np.sqrt(r0**2 + 2 * k_ * s)) / k_
    H = height if height is not None else max(4 * pitch, n * spacing**2 / s_max)  # in-sheet spacing ~ spacing
    z = rng.uniform(0, H, n)
    r = r0 + pitch * t / (2 * np.pi)
    xyz = np.stack([r * np.cos(t) * 1.15, r * np.sin(t) * 0.87, z], 1)
    truth = np.floor(t / (2 * np.pi)).astype(np.int64)
    # along-sheet edges: k_same nearest neighbours in unrolled sheet coordinates (arc length, z), so a sheet is
    # followed across the angular seam into the next wrap (d = +1 there; observed as chi * d: assuming the wrong
    # handedness flips the seam offsets).  Cross edges: the nearest node on the next wrap outward (radial probe),
    # d = +1 whatever the handedness.  kind 0 = same sheet, 1 = adjacent sheets.
    k_same = max(1, k - 2)
    k_ = pitch / (2 * np.pi)
    sarc = r0 * t + k_ * t**2 / 2
    _, nb = cKDTree(np.stack([sarc, z], 1)).query(np.stack([sarc, z], 1), k_same + 1)
    I = [np.repeat(np.arange(n), k_same)]
    J = [nb[:, 1:].ravel()]
    K = [np.zeros(n * k_same, np.uint8)]
    trees = {}
    for wv in np.unique(truth):
        idx = np.flatnonzero(truth == wv)
        trees[wv] = (cKDTree(xyz[idx]), idx)
    for wv, (tr_, idx) in trees.items():
        if wv + 1 in trees:
            tr2, idx2 = trees[wv + 1]
            dist, nb2 = tr2.query(xyz[idx], 1)
            ok_ = dist < 2.5 * pitch
            I.append(idx[ok_]); J.append(idx2[nb2[ok_]]); K.append(np.ones(int(ok_.sum()), np.uint8))
    i, j, kind = np.concatenate(I), np.concatenate(J), np.concatenate(K)
    a_, b_ = np.minimum(i, j), np.maximum(i, j)
    key, first = np.unique(a_ * n + b_, return_index=True)
    i, j, kind = key // n, key % n, kind[first]
    d = truth[j] - truth[i]
    if obs > 1:  # each pair observed 1 + Poisson(obs - 1) times, with independent errors (repeated probes)
        rep = 1 + rng.poisson(obs - 1, len(i))
        i, j, kind, d = (np.repeat(a, rep) for a in (i, j, kind, d))
    w = rng.uniform(0.3, 1.5, len(i)).astype(np.float32) * np.where(kind == 0, 1.5, 1.0)
    dobs = np.where(kind == 0, chi * d, d)  # seam offsets carry the assumed handedness
    bad = rng.random(len(i)) < p_wrong
    dobs[bad] += rng.choice([-1, 1, -2, 2], bad.sum(), p=[0.45, 0.45, 0.05, 0.05])
    ex = 0.5 * (xyz[i] + xyz[j])
    ang = np.arctan2(ex[:, 1], ex[:, 0])
    for _ in range(n_walls):
        a0 = rng.uniform(-np.pi, np.pi)
        z0 = rng.uniform(0, H)
        wall = (np.abs(np.angle(np.exp(1j * (ang - a0)))) < 0.25) & (np.abs(ex[:, 2] - z0) < 0.15 * H) & (kind == 1)
        hit = wall & (rng.random(len(i)) < wall_frac)
        dobs[hit] += 1
        bad |= hit
    xyz = xyz * scale
    e = Edges(i, j, dobs, w, kind=kind, xyz=ex * scale)
    if return_xyz:
        return truth, e, bad, xyz
    return truth, e, bad


def accuracy(x, truth, comp=None):
    """Fraction of nodes at the true winding after the best constant shift per component."""
    x, truth = np.asarray(x), np.asarray(truth)
    comp = np.zeros(len(x), np.int64) if comp is None else comp
    ok = 0
    for c in np.unique(comp):
        m = comp == c
        dif = x[m] - truth[m]
        vals, cnt = np.unique(dif, return_counts=True)
        ok += cnt.max()
    return ok / len(x)
