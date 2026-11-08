"""WP3: exact L1 winding solver (gates G1-G5, G7, G8 of results/ws_solve/gates.md, plus unit checks)."""
import itertools

import numpy as np
import pytest

import unroller.ws.solve as W
from unroller.ws.confidence import corroboration, flip_margins
from unroller.ws.solve import Edges, accuracy, make, sheetlets

try:
    import maxflow  # noqa: F401
    HAVE_BK = True
except ImportError:
    HAVE_BK = False


@pytest.mark.parametrize("p,burst", [(0.05, False), (0.2, False), (0.05, True)])
def test_lattice_certificate_beats_truth(p, burst):
    # G1: certified, and never worse than the truth's energy (the truth is feasible)
    for seed in range(20):
        t, e, _ = make(12, 12, 12, p, seed=seed, burst=burst)
        x, info = W.solve(e, len(t))
        assert info["certificate"] and info["stopped_by"] == "certificate"
        assert info["energy"] <= W.energy(t, e) + 1e-6
        assert info["max_node_cap"] < W.INT_LIMIT


def test_large_capacities_are_rescaled_never_silent():
    # G2: natural integer capacities ~3e9 per node (scipy's max-flow returns 0 for such int64 capacities)
    rng = np.random.default_rng(0)
    n = 40
    i = rng.integers(0, n, 400)
    j = rng.integers(0, n, 400)
    e = Edges(i, j, rng.integers(-2, 3, 400), np.full(400, 1e6))
    x, info = W.solve(e, n)
    assert info["certificate"] and info["max_node_cap"] < W.INT_LIMIT and info["scale"] < 1e6
    # same optimum as the unscaled-weight problem (all weights equal: scale-free)
    e1 = Edges(i, j, e.d, np.ones(400))
    x1, info1 = W.solve(e1, n)
    assert info["energy"] == pytest.approx(1e6 * info1["energy"])
    # a move graph with an over-limit capacity raises instead of returning a silently wrong cut
    P = W.Problem(e1, n)
    P.wi = np.full(len(P.wi), 2**31, np.int64)
    with pytest.raises(OverflowError):
        W.best_move(P, np.zeros(P.nf, np.int64), 1)


def test_inconsistent_maxflow_is_detected(monkeypatch):
    t, e, _ = make(6, 6, 6, 0.1, seed=1)
    real = W._maxflow_scipy

    def broken(G, s, tt):
        flow, side = real(G, s, tt)
        return 0, side  # what scipy returns for an overflowing capacity
    monkeypatch.setattr(W, "_maxflow_scipy", broken)
    with pytest.raises(RuntimeError):
        W.solve(e, len(t), init="zero")


def _brute(n, e, fixed, prior, lo=-6, hi=6):
    free = [v for v in range(n) if v not in dict(zip(*fixed))]
    best = np.inf
    x = np.zeros(n, np.int64)
    for k, v in zip(*fixed):
        x[k] = v
    for vals in itertools.product(range(lo, hi + 1), repeat=len(free)):
        x[free] = vals
        best = min(best, W.energy(x, e, prior))
    return best


def test_fixed_node_elimination_equals_big_m_and_brute_force():
    # G3
    rng = np.random.default_rng(3)
    for trial in range(200):
        n = int(rng.integers(3, 8))
        m = int(rng.integers(n, 3 * n))
        e = Edges(rng.integers(0, n, m), rng.integers(0, n, m), rng.integers(-2, 3, m),
                  rng.integers(1, 5, m).astype(float))
        nf = int(rng.integers(1, min(3, n - 1) + 1))
        fid = rng.choice(n, nf, replace=False)
        fixed = (fid, rng.integers(-1, 2, nf))
        prior = None
        if trial % 2:
            pid = rng.choice(n, 2, replace=False)
            prior = (pid, rng.integers(-2, 3, 2), np.array([1.0, 2.0]))
        x, info = W.solve(e, n, fixed=fixed, prior=prior)
        assert info["certificate"]
        assert np.all(x[fid] == fixed[1])
        Ebig = None
        # big-M: anchors as heavy priors on an unconstrained problem
        M = 1e4
        bp = (np.r_[fid, prior[0]] if prior is not None else fid,
              np.r_[fixed[1], prior[1]] if prior is not None else fixed[1],
              np.r_[np.full(nf, M), prior[2]] if prior is not None else np.full(nf, M))
        xb, ib = W.solve(e, n, prior=bp)
        assert np.all(xb[fid] == fixed[1])
        Ebig = W.energy(xb, e, prior)
        assert info["energy"] == pytest.approx(Ebig)
        if n - nf <= 4:
            assert info["energy"] == pytest.approx(_brute(n, e, fixed, prior))


def test_warm_start_reproduces_cold_optimum():
    # G4
    for seed in range(20):
        t, e, _ = make(10, 10, 10, 0.1 + 0.05 * (seed % 3), seed=seed, burst=seed % 2 == 1)
        n = len(t)
        xc, ic = W.solve(e, n, init="tree")
        xi, ii = W.solve(e, n, init="irls")
        rng = np.random.default_rng(seed)
        xw, iw = W.solve(e, n, x0=xc + rng.integers(-2, 3, n))
        assert ic["certificate"] and ii["certificate"] and iw["certificate"]
        assert ic["energy_int"] == ii["energy_int"] == iw["energy_int"]


def test_stopped_by_max_iter_is_not_certified():
    # G5
    t, e, _ = make(10, 10, 10, 0.05, seed=2)
    x, info = W.solve(e, len(t), init="zero", max_iter=1)
    assert info["stopped_by"] == "max_iter" and not info["certificate"]
    x, info = W.solve(e, len(t), init="zero")
    assert info["stopped_by"] == "certificate" and info["certificate"]


def test_chirality_picks_true_sign():
    # G7 (small): built with the true handedness (chi = +1) the graph is consistent; the mirror flips every seam
    # offset against the radial ones and is frustrated
    def build(chi):
        _, e, _ = sheetlets(4000, 0.05, seed=5, chi=chi, scale=10)
        return e, 4000
    chi, ratio, info = W.choose_chirality(build)
    assert chi == 1 and info["certified"]
    # the total-energy ratio is diluted by noise (~1.04 here); the seam edges decide
    assert info["informative_ratio"] >= 1.5 and info["decided"]


@pytest.mark.skipif(not HAVE_BK, reason="PyMaxflow not installed")
def test_bk_backend_identical_energies():
    # G8
    for seed in range(6):
        t, e, _ = make(10, 10, 10, (0.05, 0.2)[seed % 2], seed=seed, burst=seed % 3 == 2)
        xs, i_s = W.solve(e, len(t), backend="scipy")
        xb, i_b = W.solve(e, len(t), backend="bk")
        assert i_s["energy_int"] == i_b["energy_int"] and i_b["certificate"]


def test_contract_offsets_and_residual():
    t, e, _ = make(8, 8, 8, 0.0, seed=0)
    n = len(t)
    e = Edges(e.i, e.j, e.d, e.w, count=np.full(len(e), 3), kind=(e.d != 0).astype(np.uint8))
    sup, ssz, se, rep = W.contract(e, np.ones(n), min_size=1, min_count=3)
    assert rep["internal_residual"] == 0 and rep["dropped_nodes"] == 0
    off = rep["offset"]
    # within a supernode the offsets reproduce the truth up to one constant
    for s in np.unique(sup):
        m = sup == s
        assert len(np.unique(t[m] - off[m])) == 1
    # the supernode solution expanded with the offsets is optimal (zero energy) for the clean graph
    X, info = W.solve(se, len(ssz))
    x = X[sup] + off
    assert W.energy(x, e) == pytest.approx(0)
    # tiny supernodes are dropped
    sup2, ssz2, _, rep2 = W.contract(e, np.ones(n), min_size=10 ** 6)
    assert np.all(sup2 == -1) and rep2["supernodes"] == 0


def test_corroboration_cycles():
    # triangle with a cross edge: corroborated; all-SAME triangle: not; 4-cycle with a cross edge: corroborated
    e = Edges(np.array([0, 1, 0, 3, 4, 3, 6, 7, 8, 6]), np.array([1, 2, 2, 4, 5, 5, 7, 8, 9, 9]),
              np.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 1]), np.ones(10),
              kind=np.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 1]))
    x = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1])
    c = corroboration(x, e, max_len=4)
    assert c[[0, 1, 2]].all() and not c[[3, 4, 5]].any() and c[[6, 7, 8, 9]].all()
    assert not corroboration(x, e, max_len=3)[[6, 7, 8, 9]].any()


def test_flip_margins_match_brute_force():
    t, e, _ = make(6, 6, 6, 0.1, seed=4)
    x, _ = W.solve(e, len(t))
    xyz = np.stack(np.unravel_index(np.arange(len(t)), (6, 6, 6)), 1)[:, ::-1].astype(float)
    c1, cb = flip_margins(x, e, xyz, blocks=(2, 3))
    E0 = W.energy(x, e)
    for v in range(0, len(t), 17):
        best = min(W.energy(x + s * (np.arange(len(t)) == v), e) - E0 for s in (1, -1))
        assert c1[v] == pytest.approx(best)
    assert np.all(c1 >= -1e-9) and np.all(cb >= -1e-9)  # optimal: no single / block flip improves


def test_slab_pipeline_reaches_the_direct_optimum():
    tr, e, _, xyz = sheetlets(6000, 0.05, seed=7, scale=10, return_xyz=True)
    slab = np.minimum((xyz[:, 2] / (xyz[:, 2].max() / 3)).astype(int), 2)
    x, info = W.solve(e, 6000)
    sol = W.solve_slabs(e, 6000, slab, centroid_xyz=xyz, blocks=(256, 512))
    assert sol.meta["certificate"] and sol.meta["energy"] == pytest.approx(info["energy"])
    assert sol.meta["stitched_energy"] >= sol.meta["energy"]
    assert accuracy(sol.x, tr) > 0.95


def test_ties_flag_zero_cost_moves_only():
    from unroller.ws.confidence import ties
    # node 1 is pulled to 0 and to 1 with equal weight: both windings cost the same -> TIE
    e = Edges(np.array([0, 0, 1]), np.array([1, 1, 2]), np.array([0, 1, 0]), np.array([1.0, 1.0, 5.0]))
    x, info = W.solve(e, 3, fixed=([0], [0]))
    t = ties(x, e, fixed=([0], [0]))
    assert t[1] and t[2] and not t[0]  # 1 and 2 move together at zero cost
    e2 = Edges(e.i, e.j, e.d, np.array([1.0, 2.0, 5.0]))
    x2, _ = W.solve(e2, 3, fixed=([0], [0]))
    assert not ties(x2, e2, fixed=([0], [0])).any()
    # without anchors the gauge is fixed per component from centroids
    t3 = ties(x2, e2, centroid_xyz=np.zeros((3, 3)) + np.arange(3)[:, None])
    assert not t3[1]
