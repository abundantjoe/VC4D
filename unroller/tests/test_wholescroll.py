import numpy as np
import pytest

from unroller.surface import Surface


@pytest.fixture(scope="module")
def ws():
    from unroller.wholescroll import WholeScroll

    return WholeScroll((48, 160, 160), seed=2, wraps=4, chunk=64)


def test_true_surface_truth_is_consistent(ws):
    for w in range(4):
        S = ws.true_surface((w, w + 1), step=2.0)
        p = S.points[S.valid]
        c = ws.sheet_coords(p)
        assert np.all(c["w"] == w)
        assert np.abs(c["depth"]).max() < 0.1
        # the exact layout is the isometric grid of the surface (s along columns, z along rows)
        st = ws.true_unrolled(p)
        r, cc = np.nonzero(S.valid)
        assert np.abs(st[:, 0] - (S.meta["s0"] + 1.0 + 2.0 * cc)).max() < 0.05
        assert np.abs(st[:, 1] - 2.0 * r).max() < 0.05


def test_tiles_are_seamless():
    from unroller.wholescroll import WholeScroll

    a = WholeScroll((48, 128, 128), seed=5, wraps=3, chunk=32).volume_obj().levels[0].read(5, 45, 20, 100, 30, 110)
    b = WholeScroll((48, 128, 128), seed=5, wraps=3, chunk=64).volume_obj().levels[0].read(5, 45, 20, 100, 30, 110)
    assert np.array_equal(a, b)
    assert np.all(a % 8 == 0) and 0.2 < (a == 0).mean() < 0.9


def test_tight_core_and_long_fusions():
    from unroller.phantom import _gap_radius

    g = dict(thick_median=6.0, thick_sigma_ln=0.1, gap_median=8.0, gap_sigma_ln=0.1, corr_vx=80.0,
             core_factor=0.3, core_wraps=3.0, fusions=[(4, 1.0, 1.5, 3)])  # 0.3 rad tapers at both ends
    t, r, T = _gap_radius(g, 10.0, 10, seed=0)
    n = int(round(2 * np.pi / (t[1] - t[0])))
    pitch = r[n:] - r[:-n]
    turn = t[n:] / (2 * np.pi)  # turn index of the outer sample (starts at 1)
    assert pitch[turn < 1.25].mean() < 0.8 * pitch[(turn > 6) & (turn < 9)].mean()  # tight core
    gap = pitch - 0.5 * (T[n:] + T[:-n])
    fused = ((turn >= 5) & (turn < 7)) & (np.mod(t[n:], 2 * np.pi) > 1.3) & (np.mod(t[n:], 2 * np.pi) < 2.2)
    assert gap[fused].max() < 0.5  # wraps 5 and 6 rest on wraps 4 and 5 over 1.5 rad


def test_crush_is_monotone_and_creases():
    from unroller.phantom import _crush

    x = np.linspace(-200, 200, 4001)
    xy = np.stack([x, np.zeros_like(x)], 1)
    out = _crush(xy, [dict(angle=0.0, c=100.0, k=0.6, w=3.0)])[:, 0]
    d = np.diff(out) / np.diff(x)
    assert d.min() > 0.59 and d.max() <= 1.0 + 1e-9  # injective: wraps never cross
    assert abs(out[-1] - (100 + 0.6 * 100)) < 3 * 3.0 * np.log(2) + 1e-6


def test_twist_round_trip(ws):
    rng = np.random.default_rng(0)
    m = rng.uniform([-60, -60, 0], [60, 60, 48], (500, 3))
    assert ws.twist != 0
    back = ws.to_material(ws.to_world(m))
    assert np.abs(back - m).max() < 2e-3


def test_score_truth_is_perfect_and_errors_are_measured(ws):
    from unroller.wholescore import score_whole

    surfs = [ws.true_surface((w, w + 1), step=2.0) for w in range(4)]
    claims = [np.full(S.shape, w + 7.0) for w, S in enumerate(surfs)]
    uvs = [np.stack(np.broadcast_arrays(S.meta["s0"] + 1.0 + 2.0 * np.arange(S.shape[1])[None, :],
                                        2.0 * np.arange(S.shape[0])[:, None]), -1) for S in surfs]
    r = score_whole(ws, surfs, claims, uvs=uvs)
    assert r["winding_accuracy"] == 1.0 and r["jump_edges"] == 0
    assert r["duplicate_area"] == 0 and r["self_overlap_area"] == 0 and r["wrong_wrap_area"] == 0
    assert all(f["procrustes_rms_vx"] < 1e-3 and f["flipped_triangle_frac"] == 0 for f in r["flatten"])
    cov = [p["covered"] for p in r["coverage"]["per_wrap"][:4]]
    assert min(cov) > 0.95  # cell resolution: cells cut by hole edges can lack a vertex
    # errors: a block claimed one wrap off, a patch spliced in from two wraps out, a duplicated surface
    c2 = [c.copy() for c in claims]
    c2[2][:, :10] += 1
    P2 = surfs[1].points.copy()
    P2[4:12, 10:20] = surfs[3].points[4:12, 10:20]
    s2 = list(surfs)
    s2[1] = Surface(P2, surfs[1].scale, surfs[1].meta)
    s2.append(surfs[0])
    c2.append(claims[0])
    r2 = score_whole(ws, s2, c2)
    exp = 4 * (np.isfinite(surfs[2].points[:, :10, 0]).sum() + np.isfinite(surfs[3].points[4:12, 10:20, 0]).sum())
    assert r2["wrong_wrap_area"] == exp
    assert r2["jump_edges"] > 20
    assert r2["duplicate_area"] >= 0.9 * r["coverage"]["per_wrap"][0]["present_area"] * cov[0]


def test_theta_and_arc_length_claims(ws):
    from unroller.wholescore import score_whole

    surfs = [ws.true_surface((w, w + 1), step=2.0) for w in range(4)]
    th = [ws.sheet_coords(np.nan_to_num(S.points))["theta"] - 3.0 for S in surfs]
    assert score_whole(ws, surfs, th, kind="theta")["winding_accuracy"] == 1.0
    ss = [ws.sheet_coords(np.nan_to_num(S.points))["s"] + 123.0 for S in surfs]
    assert score_whole(ws, surfs, ss, kind="s")["winding_accuracy"] == 1.0


def test_ink_ncc_plumbing(ws):
    from unroller.wholescore import score_whole

    S = ws.true_surface((2, 3), step=1.0)
    c = ws.sheet_coords(np.nan_to_num(S.points))
    ink = ws.ink_at(c["s"], c["z"])
    assert ink.max() > 0.5  # real glyphs are present in this wrap
    good = score_whole(ws, S, np.full(S.shape, 2.0), renders=ink)
    assert good["ink_ncc"] > 0.999
    bad = score_whole(ws, S, np.full(S.shape, 2.0), renders=np.random.default_rng(0).permutation(ink.ravel()).reshape(ink.shape))
    assert abs(bad["ink_ncc"]) < 0.1
