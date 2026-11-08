"""Exact L1 integer winding assignment by steepest descent with min-cut moves.
E(x) = sum_e w_e |x_j - x_i - d_e| is L-natural-convex (convex in differences), so the steepest
descent over +-1 set moves, each solved exactly by an s-t min cut, reaches the global optimum
(Murota; Kolmogorov & Shioura 2009, convex cost tension problem). Uses scipy maximum_flow only."""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow, breadth_first_order, minimum_spanning_tree

def energy(x, E, d, w):
    return float(np.sum(w * np.abs(x[E[:, 1]] - x[E[:, 0]] - d)))

def best_move(x, E, d, wi, sign):
    """Return boolean mask S minimising E(x + sign*1_S) (exact, via min cut)."""
    n = len(x); s, t = n, n + 1
    r = x[E[:, 1]] - x[E[:, 0]] - d
    # B: only j moves; C: only i moves (in units of w)
    B = wi * (np.abs(r + sign) - np.abs(r))
    C = wi * (np.abs(r - sign) - np.abs(r))
    pair = B + C                      # = 2w where r == 0, else 0  (>= 0: submodular)
    i, j = E[:, 0], E[:, 1]
    # unary: +C on s_i, -C on s_j  (cost of label 1 = moved)
    u1 = np.zeros(n, np.int64)       # cost if moved (paid via source->node edge)
    np.add.at(u1, i, C); np.add.at(u1, j, -C)
    src_cap = np.maximum(u1, 0); snk_cap = np.maximum(-u1, 0)
    rows = [np.full(n, s), np.arange(n)]; cols = [np.arange(n), np.full(n, t)]
    caps = [src_cap, snk_cap]
    m = pair > 0
    rows.append(i[m]); cols.append(j[m]); caps.append(pair[m])
    R = np.concatenate(rows); Cc = np.concatenate(cols); K = np.concatenate(caps)
    keep = K > 0
    G = sp.csr_matrix((K[keep].astype(np.int32), (R[keep], Cc[keep])), shape=(n + 2, n + 2))
    G.sum_duplicates()
    if G.nnz == 0:
        return np.zeros(n, bool), 0
    f = maximum_flow(G, s, t, method='dinic')
    res = (G - f.flow).tocsr(); res.data[res.data < 0] = 0; res.eliminate_zeros()
    reach = breadth_first_order(res, s, directed=True, return_predecessors=False)
    src_side = np.zeros(n + 2, bool); src_side[reach] = True
    S = ~src_side[:n]                 # sink side = label 1 = moved
    return S, f.flow_value

def solve(x0, E, d, w, max_iter=200, scale=1000):
    wi = np.maximum(1, np.round(w * scale)).astype(np.int64)
    x = x0.copy(); it = 0
    while it < max_iter:
        improved = False
        for sign in (+1, -1):
            S, _ = best_move(x, E, d, wi, sign)
            if S.any():
                xn = x.copy(); xn[S] += sign
                if energy(xn, E, d, wi) < energy(x, E, d, wi):
                    x = xn; improved = True
        it += 1
        if not improved:
            break
    return x, it

def tree_init(n, E, d, w, root=0, mst=True):
    """Propagate offsets along a (max-weight spanning) tree: TA's MST init / VC atlas BFS layout."""
    ww = (w.max() + 1.0 - w) if mst else np.ones_like(w)
    G = sp.csr_matrix((ww, (E[:, 0], E[:, 1])), shape=(n, n))
    T = minimum_spanning_tree(G + G.T)
    T = (T + T.T).tocsr()
    order, pred = breadth_first_order(T, root, directed=False, return_predecessors=True)
    key = {}
    for k, (a, b) in enumerate(E):
        key[(a, b)] = d[k]; key[(b, a)] = -d[k]
    x = np.zeros(n, np.int64)
    for v in order[1:]:
        u = pred[v]; x[v] = x[u] + key[(u, v)]
    return x

def make(nx, ny, nz, p_out, seed=0, burst=False):
    rng = np.random.default_rng(seed)
    idx = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    t = np.broadcast_to((np.arange(ny) // 3)[None, :, None], idx.shape).ravel().astype(np.int64)
    E = np.concatenate([np.stack([a.ravel(), b.ravel()], 1) for a, b in
                        [(idx[:, :, :-1], idx[:, :, 1:]), (idx[:, :-1, :], idx[:, 1:, :]), (idx[:-1], idx[1:])]])
    d = t[E[:, 1]] - t[E[:, 0]]
    w = rng.uniform(0.5, 1.5, len(E))
    bad = rng.random(len(E)) < p_out
    if burst:  # systematic: every edge crossing a wall (x = nx/2, lower half z) is wrong by +1 (a 'wrap jump' sheet)
        xa, xb = E[:, 0] % nx, E[:, 1] % nx
        za = E[:, 0] // (nx * ny)
        bad |= (xa != xb) & (np.minimum(xa, xb) == nx // 2 - 1) & (za < nz // 2)
    dobs = d.copy(); dobs[bad] += rng.choice([-2, -1, 1, 2], bad.sum())
    return t, E, dobs, w, bad

if __name__ == '__main__':
    for (nx, ny, nz, p, burst) in [(30, 30, 10, 0.05, False), (30, 30, 10, 0.20, False),
                                    (30, 30, 10, 0.05, True), (60, 60, 20, 0.10, False)]:
        t, E, d, w, bad = make(nx, ny, nz, p, burst=burst)
        n = len(t)
        x0 = tree_init(n, E, d, w); x0 += t[0] - x0[0]
        t0 = time.time(); x, it = solve(x0, E, d, w); dt = time.time() - t0
        x += t[0] - x[0]
        acc0 = (x0 == t).mean(); acc = (x == t).mean()
        print(f"n={n} m={len(E)} bad={bad.mean():.3f} burst={burst}: MST-init acc={acc0:.4f} "
              f"-> exact-L1 acc={acc:.4f}  E(sol)={energy(x,E,d,w):.1f} <= E(truth)={energy(t,E,d,w):.1f}  "
              f"iters={it} time={dt:.2f}s", flush=True)
