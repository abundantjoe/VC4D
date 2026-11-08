"""Wrap-jump safeguards of the tracer on small synthetic inputs: plane fit, basin-confined mean-shift,
relative width gate, seed check, determinism."""
import warnings

import numpy as np
import pytest

from unroller import _kernels as K
from unroller.snap_fast import basin_bounds, snap_ms
from unroller.trace import SeedRejected, _plane_fit, _seed_check, _snap_ms, trace
from unroller.volume import ArrayVolume


def _sheets(shape=(48, 128, 64), xs=(16, 28, 40), thick=3.0, fat=None, blob=None):
    """Parallel sheets x = xs (Gaussian density across, FWHM thick); fat = (x0, y1, f): sheet x0 is f times
    thicker for y > y1 (a fused contact); blob = (centre xyz, radius): a dense sphere (a foreign body)."""
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
    v = np.zeros(shape)
    for x0 in xs:
        t = np.where(y > fat[1], fat[2] * thick, thick) if fat is not None and x0 == fat[0] else thick
        v += 120 * np.exp(-0.5 * ((x - x0) / (t / 2.355)) ** 2)
    if blob is not None:
        c, r = blob
        v = np.maximum(v, np.where((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= r * r, 200, 0))
    return ArrayVolume(np.clip(v + 20, 0, 255).astype(np.uint8))


KW = dict(step=3.0, generations=30, umbilicus=(0.0, 64.0, 24.0))


def test_plane_fit_exact_and_conditioned():
    rng = np.random.default_rng(0)
    N = 24
    o, u, v = np.array([5.0, 7.0, 9.0]), np.array([3.0, 0.4, -0.2]), np.array([0.1, -0.3, 2.9])
    r, c = np.mgrid[0:N, 0:N]
    P = o + c[..., None] * u + r[..., None] * v
    P[rng.random((N, N)) < 0.4] = np.nan
    P[10, 10] = np.nan
    P[0:6, 14:21] = np.nan  # cell (2, 17): the only valid vertices of its window lie in one row
    P[0, 14:21] = o + np.arange(14, 21)[:, None] * u
    fr, fc = np.array([10, 2, 20]), np.array([10, 17, 3])
    a, n, good = _plane_fit(P, fr, fc, 3)
    true_n = np.cross(u, v) / np.linalg.norm(np.cross(u, v))
    assert good[0] and good[2] and not good[1]  # (2, 17): only collinear vertices in its window
    for i in (0, 2):
        assert np.allclose(a[i], o + fc[i] * u + fr[i] * v, atol=1e-9)
        assert np.isclose(abs(n[i] @ true_n), 1.0)
    # compiled moments == numpy's vectorised accumulation, bit for bit
    fr, fc = np.nonzero(~np.isfinite(P[..., 0]))
    k = np.arange(-3, 4)
    wt = np.exp(-0.5 * (k[:, None] ** 2 + k[None, :] ** 2) / 2.0 ** 2)
    M, B, cnt = np.empty((len(fr), 3, 3)), np.empty((len(fr), 3, 3)), np.empty(len(fr), np.int64)
    K.plane_moments(P, fr, fc, wt, M, B, cnt)
    M0, B0 = np.zeros_like(M), np.zeros_like(B)
    for dr in k:
        for dc in k:
            if dr or dc:
                rr, cc = fr + dr, fc + dc
                q = P[np.clip(rr, 0, N - 1), np.clip(cc, 0, N - 1)]
                ok = (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N) & np.isfinite(q[:, 0])
                w = wt[dr + 3, dc + 3] * ok
                phi = np.array([1.0, dc, dr])
                M0 += w[:, None, None] * np.outer(phi, phi)[None]
                B0 += w[:, None, None] * phi[None, :, None] * np.where(ok[:, None], q, 0.0)[:, None, :]
    assert np.array_equal(M, M0) and np.array_equal(B, B0)


def test_window_nanmedian_matches_numpy():
    rng = np.random.default_rng(1)
    G = np.where(rng.random((20, 20)) < 0.5, rng.integers(1, 20, (20, 20)) * 0.5, np.nan)
    fr, fc = np.nonzero(np.ones((20, 20), bool))
    out = np.empty(len(fr))
    K.window_nanmedian(G, fr, fc, 2, out)
    win = np.stack([G[np.clip(fr + dr, 0, 19), np.clip(fc + dc, 0, 19)]
                    for dr in range(-2, 3) for dc in range(-2, 3) if dr or dc], 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN windows
        ref = np.nanmedian(win, 1)
    assert np.array_equal(out, ref, equal_nan=True)


def test_basin_confines_mean_shift():
    """A dim sheet at x=20 next to a bright one at x=27: the free mean-shift is pulled across the gap (to its
    +-5 vx limit), the basin-confined one converges on its own sheet; compiled == numpy reference."""
    x = np.arange(48.0)
    prof = 20 + 100 * np.exp(-0.5 * ((x - 20) / 1.3) ** 2) + 200 * np.exp(-0.5 * ((x - 27) / 1.3) ** 2)
    V = ArrayVolume(np.broadcast_to(prof.astype(np.uint8), (16, 16, 48)).copy())
    p, n = np.array([[21.0, 8.0, 8.0], [19.0, 8.0, 8.0]]), np.array([[1.0, 0, 0], [1.0, 0, 0]])
    for kw in (dict(iters=8, tol=0.05), dict(iters=50, tol=1e-3)):
        free, ok_free = _snap_ms(V, p, n, 5.0, 3.0, min_contrast=0.0, **kw)
        conf, ok_conf = _snap_ms(V, p, n, 5.0, 3.0, min_contrast=0.0, basin=30.0, **kw)
        assert not ok_free.any() and np.all(free[:, 0] > 23)
        assert ok_conf.all() and np.allclose(conf[:, 0], 20.0, atol=0.01)
        fast = snap_ms(V, p, n, 5.0, 3.0, min_contrast=0.0, basin=30.0, **kw)
        assert np.array_equal(fast[0], conf) and np.array_equal(fast[1], ok_conf)
    lo, hi = basin_bounds((prof[7:36] - prof.min())[None].astype(np.float32), 1.0, 30.0)
    assert lo[0] == 0 and 15 <= hi[0] <= 17  # the dark minimum between the sheets (x ~ 23.5)
    lo, hi = basin_bounds((prof[7:36] - prof.min())[None].astype(np.float32), 1.0, 500.0)
    assert (lo[0], hi[0]) == (0, 28)  # no minimum that deep: the whole profile


def test_relative_width_gate_leaves_a_hole():
    """The traced sheet doubles in thickness for y > 80 (below the global 2.5x gate): the relative gate
    stops growth there, without it the trace runs on."""
    V = _sheets(fat=(28, 80, 2.0))
    T = trace(V, (28.0, 40.0, 24.0), **KW)
    T0 = trace(V, (28.0, 40.0, 24.0), width_rel=None, **KW)
    assert np.sum(T.points[T.valid][:, 1] > 84) == 0
    assert np.sum(T0.points[T0.valid][:, 1] > 84) > 100
    assert np.allclose(T.points[T.valid][:, 0], 28.0, atol=0.2)
    T2 = trace(V, (28.0, 40.0, 24.0), **KW)  # deterministic
    assert np.array_equal(T.points, T2.points, equal_nan=True)


def test_plane_gate_refuses_a_step_off_the_plane():
    """The traced sheet (spacing 14 vx: gate 3.5 vx, snap radius 4.9 vx) steps by d vx in x for y >= 70: a step
    beyond the gate leaves a hole, one within it is followed, and without the gate the trace follows both."""
    z, y, x = np.mgrid[0:48, 0:128, 0:64].astype(float)
    for d, gated in ((3.0, False), (4.0, True)):
        v = sum(120 * np.exp(-0.5 * ((x - np.where((y >= 70) & (x0 == 28), x0 + d, x0)) / (3.0 / 2.355)) ** 2)
                for x0 in (14, 28, 42))
        V = ArrayVolume(np.clip(v + 20, 0, 255).astype(np.uint8))
        for pg in (0.25, None):
            T = trace(V, (28.0, 40.0, 24.0), plane_gate=pg, **KW)
            p = T.points[T.valid]
            assert T.meta["sheet_spacing"] == 14.0
            beyond = p[p[:, 1] > 74]
            if gated and pg:
                assert len(beyond) == 0
            else:
                assert len(beyond) > 200 and np.allclose(beyond[:, 0], 28 + d, atol=0.2)


def test_seed_on_foreign_body_refused():
    V = _sheets(blob=((33.0, 40.0, 24.0), 7.0))
    with pytest.raises(SeedRejected, match="width"):
        trace(V, (28.0, 40.0, 24.0), seed_check=True, **dict(KW, generations=2))
    T = trace(V, (28.0, 40.0, 24.0), **dict(KW, generations=2))  # default: flagged, not refused
    assert T.meta["seed_check"]["outlier"]
    with pytest.raises(ValueError, match="probes"):
        trace(V, (28.0, 40.0, 24.0), seed_check=True, probes=0.0, **dict(KW, generations=2))
    T = trace(V, (28.0, 100.0, 24.0), **dict(KW, generations=2))  # same sheet, away from the blob
    assert not T.meta["seed_check"]["outlier"] and T.meta["seed_check"]["probes"] >= 12
    ws, ss = np.full(24, 3.0), np.full(24, 12.0)
    assert _seed_check(3.0, ws, 40.0, ss)["outlier"] and _seed_check(3.0, ws, 4.0, ss)["outlier"]
    c = _seed_check(3.0, ws, np.nan, ss)  # no spacing at the seed: flagged, not refused
    assert c["suspect"] and not c["outlier"]
    assert not any(_seed_check(7.0, ws, 25.0, ss)[k] for k in ("outlier", "suspect"))


def test_wrap_report_cross_tears_folds():
    """Synthetic grids on a phantom's true sheet: clean, half moved one turn outward (cross-wrap edges), half moved
    30 vx along the same wrap (tears), the last columns folded back (fold fraction)."""
    from unroller.bench import wrap_report
    from unroller.phantom import Phantom

    P = Phantom.random(1)
    s = 0.5 * P.length + 3.0 * np.arange(10)
    z = 20.0 + 3.0 * np.arange(8)[:, None]
    grid = lambda s_: P.point(np.broadcast_to(s_, (8, 10)), z)  # noqa: E731
    assert wrap_report(P, grid(s), 3.0) == {"cross_wrap_edges": 0, "tears": 0, "fold_frac": 0.0}
    t = np.interp(s, P._sc, P._tc)
    outer = np.where(np.arange(10) >= 6, np.interp(t + 2 * np.pi, P._tc, P._sc), s)
    assert wrap_report(P, grid(outer), 3.0) == {"cross_wrap_edges": 8, "tears": 0, "fold_frac": 0.0}
    r = wrap_report(P, grid(np.where(np.arange(10) >= 6, s + 30.0, s)), 3.0)
    assert (r["cross_wrap_edges"], r["tears"]) == (0, 8)
    folded = np.where(np.arange(10) > 6, s[6] - 3.0 * (np.arange(10) - 6), s)
    r = wrap_report(P, grid(folded), 3.0)
    assert (r["cross_wrap_edges"], r["tears"]) == (0, 0) and np.isclose(r["fold_frac"], 3 / 9)
    G = grid(s)
    G[2:4, 3:5] = np.nan  # holes are not scored
    assert wrap_report(P, G, 3.0)["cross_wrap_edges"] == 0
