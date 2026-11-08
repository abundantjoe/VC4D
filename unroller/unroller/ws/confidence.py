"""Confidence of a winding solution.

Every number here is a *flip margin*: how much the L1 energy rises if a set containing node i is moved by +-1.
At a certified optimum all margins are >= 0; a small margin means the data barely prefers the assigned winding.

* c1_i      the node alone moved (min over the sign);
* cblk_i    the octree block of edge ``L`` (L0 voxels, from node centroids) containing i moved as a whole, minimised
            over L in ``blocks`` - catches regions that could slip together by one wrap;
* conf_i  = min(c1_i, cblk_i), an upper bound of the exact margin
* c*_i      = min over sets S containing i (one max-flow with i forced into S), validated against the proxy.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .solve import SAME, Edges


def _unary_terms(n, fixed, prior):
    isfix = np.zeros(n, bool)
    if fixed is not None:
        isfix[np.asarray(fixed[0], np.int64)] = True
    if prior is None:
        return isfix, None
    pid, ppi, pu = (np.asarray(p) for p in prior)
    return isfix, (pid.astype(np.int64), ppi.astype(np.int64), pu.astype(np.float64))


def flip_margins(x, e: Edges, centroid_xyz, blocks=(256, 512, 1024, 2048), fixed=None, prior=None):
    """(c1, cblk) per node (float64; inf where the set contains an anchor)."""
    x = np.asarray(x, np.int64)
    n = len(x)
    isfix, pr = _unary_terms(n, fixed, prior)
    w = e.w.astype(np.float64)
    r = x[e.j] - x[e.i] - e.d
    c1 = np.full(n, np.inf)
    for s in (1, -1):
        # i moved by s: r -> r - s;  j moved by s: r -> r + s
        cost = np.bincount(e.i, w * (np.abs(r - s) - np.abs(r)), n) + np.bincount(e.j, w * (np.abs(r + s) - np.abs(r)), n)
        loop = e.i == e.j
        if loop.any():
            cost -= np.bincount(e.i[loop], w[loop] * (np.abs(r[loop] - s) + np.abs(r[loop] + s) - 2 * np.abs(r[loop])), n)
        if pr is not None:
            pid, ppi, pu = pr
            cost += np.bincount(pid, pu * (np.abs(x[pid] + s - ppi) - np.abs(x[pid] - ppi)), n)
        c1 = np.minimum(c1, cost)
    c1[isfix] = np.inf
    xyz = np.asarray(centroid_xyz, np.float64)
    cblk = np.full(n, np.inf)
    for L in blocks:
        q = np.floor(xyz / L).astype(np.int64)
        q -= q.min(0)
        dims = q.max(0) + 1
        bid = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
        ub, b = np.unique(bid, return_inverse=True)
        nb = len(ub)
        bi, bj = b[e.i], b[e.j]
        cut = bi != bj
        best = np.full(nb, np.inf)
        for s in (1, -1):
            cost = np.bincount(bi[cut], w[cut] * (np.abs(r[cut] - s) - np.abs(r[cut])), nb) \
                + np.bincount(bj[cut], w[cut] * (np.abs(r[cut] + s) - np.abs(r[cut])), nb)
            if pr is not None:
                pid, ppi, pu = pr
                cost += np.bincount(b[pid], pu * (np.abs(x[pid] + s - ppi) - np.abs(x[pid] - ppi)), nb)
            best = np.minimum(best, cost)
        anch = np.bincount(b, isfix, nb) > 0
        best[anch] = np.inf
        cblk = np.minimum(cblk, best[b])
    return c1, cblk


def exact_margin(x, e: Edges, nodes, fixed=None, prior=None, backend="scipy", centroid_xyz=None, radius=None):
    """c*_i: the minimal energy increase over all +-1 moves of a node set S containing node i (one forced
    max-flow per node and sign), in weight units of the quantised problem.  S may not contain anchors; with
    ``radius`` (and ``centroid_xyz``) S is also confined to the nodes within ``radius`` of node i - the region
    the block proxy explores (all other nodes stay at x).  Without anchors or a radius c* is 0 (a whole
    component can shift for free)."""
    from .solve import Problem, best_move

    x = np.asarray(x, np.int64)
    n = len(x)
    out = np.full(len(nodes), np.inf)
    base_fix = None if fixed is None else (np.asarray(fixed[0], np.int64), np.asarray(fixed[1], np.int64))
    if radius is None:
        P = Problem(e, n, base_fix, prior)
        xf = x[P.free]
    xyz = None if centroid_xyz is None else np.asarray(centroid_xyz, np.float64)
    for k, v in enumerate(np.asarray(nodes)):
        if radius is not None:
            far = np.linalg.norm(xyz - xyz[v], axis=1) > radius
            fid = np.flatnonzero(far)
            if base_fix is not None:
                fid = np.union1d(fid, base_fix[0])
            # restrict to edges touching the ball (others are constant) for speed
            near = ~far
            m = near[e.i] | near[e.j]
            P = Problem(e.subset(m), n, (fid, x[fid]), prior)
            xf = x[P.free]
        lv = P.loc[v]
        if lv < 0:
            continue
        for s in (1, -1):
            _, delta = best_move(P, xf, s, backend, force=int(lv))
            out[k] = min(out[k], delta / P.scale)
    return out


def gauge_anchors(x, e: Edges, centroid_xyz):
    """One anchor per connected component (>= 2 nodes): the node nearest the component's centroid, at x."""
    from scipy.sparse.csgraph import connected_components

    n = len(x)
    xyz = np.asarray(centroid_xyz, np.float64)
    G = sp.csr_matrix((np.ones(len(e)), (e.i, e.j)), shape=(n, n))
    nc, comp = connected_components(G, directed=False)
    o = np.argsort(comp, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(comp[o])) + 1, n]
    anchors = []
    for a, b in zip(starts[:-1], starts[1:]):
        m = o[a:b]
        if len(m) > 1:
            anchors.append(m[np.argmin(np.linalg.norm(xyz[m] - xyz[m].mean(0), axis=1))])
    anchors = np.asarray(anchors, np.int64)
    return anchors, np.asarray(x, np.int64)[anchors]


def ties(x, e: Edges, centroid_xyz=None, fixed=None, prior=None):
    """TIE flags: nodes in some +-1 move of zero cost (the solution is not unique there).

    The gauge is fixed by ``fixed`` / ``prior`` or, if neither is given, by one anchor per component (see
    gauge_anchors).  At the optimum the best move of each sign costs 0, and scipy's residual reachability gives the
    maximal min-cut sink side, i.e. the union of all zero-cost move sets; TIE = union over both signs.  Raises if x is
    not optimal (an improving move exists)."""
    from .solve import Problem, best_move

    x = np.asarray(x, np.int64)
    if fixed is None and prior is None:
        if centroid_xyz is None:
            raise ValueError("ties need anchors, priors or centroids (to anchor each component)")
        fixed = gauge_anchors(x, e, centroid_xyz)
    P = Problem(e, len(x), fixed, prior)
    xf = x[P.free]
    out = np.zeros(len(x), bool)
    for s in (1, -1):
        S, delta = best_move(P, xf, s, backend="scipy")
        if delta < 0:
            raise ValueError("x is not optimal: an improving move exists")
        out[P.free[S]] = True
    return out


def validate(x, e: Edges, centroid_xyz, n_sample=1000, seed=0, blocks=(256, 512, 1024, 2048), fixed=None,
             prior=None, backend="scipy", exclude_ties=False):
    """Proxy min(c1, cblk) against the exact c* on sampled nodes: Spearman rank correlation and the fraction of
    nodes where the proxy exceeds 2 c* (the proxy is an upper bound; it is useful only if it is not loose)."""
    from scipy.stats import spearmanr

    x = np.asarray(x, np.int64)
    radius = float(max(blocks)) * np.sqrt(3)  # c* over sets within the region the largest block can reach
    c1, cb = flip_margins(x, e, centroid_xyz, blocks, fixed, prior)
    proxy = np.minimum(c1, cb)
    tie = ties(x, e, centroid_xyz, fixed, prior) if exclude_ties else np.zeros(len(x), bool)
    rng = np.random.default_rng(seed)
    ok = np.isfinite(proxy) & ~tie
    cand = rng.permutation(np.flatnonzero(ok))
    xyz = np.asarray(centroid_xyz, np.float64)
    lo, hi = xyz.min(0), xyz.max(0)
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    # a ball that holds the whole graph lets the free global shift in (c* = 0 regardless of the data): skip those
    covers = np.array([np.linalg.norm(corners - xyz[v], axis=1).max() <= radius for v in cand])
    excluded_whole_graph = int(np.sum(covers[:n_sample * 2]))
    nodes = cand[~covers][:n_sample]
    cs = exact_margin(x, e, nodes, fixed, prior, backend, centroid_xyz, radius)
    p = proxy[nodes]
    rho = float(spearmanr(p, cs).statistic)
    loose = p > 2 * cs + 1e-6
    return {"n": len(nodes), "radius": radius, "excluded_whole_graph_balls": excluded_whole_graph,
            "tie_share": float(tie.mean()), "tie": tie, "spearman": rho, "frac_proxy_gt_2cstar": float(loose.mean()),
            "frac_proxy_below_cstar": float(np.mean(p < cs - 1e-3)), "median_cstar": float(np.median(cs)),
            "median_proxy": float(np.median(p)), "nodes": nodes, "cstar": cs, "proxy": p}


def corroboration(x, e: Edges, max_len=4):
    """True where the node lies on >= 1 cycle of length <= max_len (2, 3 or 4) made of edges with zero residual
    that contains at least one non-SAME edge (two independent observations close the same loop).  A 2-cycle is
    a node pair joined by >= 2 satisfied edges, one of them non-SAME."""
    x = np.asarray(x, np.int64)
    n = len(x)
    sat = (x[e.j] - x[e.i] - e.d == 0) & (e.i != e.j)
    a, b = np.minimum(e.i[sat], e.j[sat]), np.maximum(e.i[sat], e.j[sat])
    ns = e.kind[sat] != SAME
    key = a * n + b
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    nsp = np.bincount(inv, ns, len(uk)) > 0  # pair carries a non-SAME satisfied edge
    pa, pb = uk // n, uk % n
    out = np.zeros(n, bool)
    if max_len >= 2:
        two = (cnt >= 2) & nsp
        out[pa[two]] = out[pb[two]] = True

    def adj(m):
        A = sp.csr_matrix((np.ones(int(m.sum())), (pa[m], pb[m])), shape=(n, n))
        return (A + A.T).tocsr()

    if max_len >= 3:
        A = adj(np.ones(len(uk), bool))
        S = adj(~nsp)
        A2 = A @ A
        S2 = S @ S
        tri_a = np.asarray(A2.multiply(A).sum(1)).ravel()
        tri_s = np.asarray(S2.multiply(S).sum(1)).ravel()
        out |= tri_a > tri_s
        if max_len >= 4:
            def c4(M, M2):
                deg = np.asarray(M.sum(1)).ravel()
                walk4 = np.asarray(M2.multiply(M2).sum(1)).ravel()
                return walk4 - deg**2 - (M @ (deg - 1))
            out |= c4(A, A2) > c4(S, S2)
    return out
