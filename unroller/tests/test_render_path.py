"""Render path for traced surfaces: smoothed render normals, along-normal regularisation, pinhole filling."""
import numpy as np
import pytest

from unroller import Surface
from unroller.phantom import SpiralScroll
from unroller.regularize import fill_holes, interior_holes, regularize, remove_folds
from unroller.render import render
from unroller.resample import flat_surface
from unroller.surface import NORMAL_SIGMA_VX
from unroller.volume import ArrayVolume


@pytest.fixture(scope="module")
def spiral():
    P = SpiralScroll(size=(48, 224, 224))
    return P, ArrayVolume(P.volume())


def _jittered(P, amp=1.0, seed=0, step=2.0):
    """Ground-truth grid with independent along-normal jitter of +-amp vx per vertex (a traced grid's defect)."""
    S = P.surface(0.3 * P.length, 0.6 * P.length, step=step, distort=0.0)
    n = S.normals()
    j = np.random.default_rng(seed).uniform(-amp, amp, S.shape)
    return S, Surface(S.points + j[..., None] * n, S.scale, {"step": step})


def _angle(a, b):
    c = np.abs((a * b).sum(-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def test_smoothed_normals_follow_sheet_not_jitter(spiral):
    P, _ = spiral
    G, S = _jittered(P, amp=1.0, step=1.0)
    S.points[20:23, 40:44] = np.nan  # a hole must not disturb its surroundings
    S = Surface(S.points, S.scale)
    nt = G.normals()
    ok = S.valid
    raw = _angle(S.normals()[ok], nt[ok])
    sm = _angle(S.normals(NORMAL_SIGMA_VX)[ok], nt[ok])
    assert np.percentile(raw, 90) > 30 and np.percentile(sm, 90) < 3
    assert np.all(np.isfinite(S.normals(NORMAL_SIGMA_VX)[ok])) and np.all(np.isnan(S.normals(2.0)[~ok]))
    assert np.array_equal(S.normals(0.0), S.normals(), equal_nan=True)  # raw normals stay available
    # same orientation convention (du x dv) as the raw normals
    assert np.median((S.normals(2.0)[ok] * S.normals()[ok]).sum(-1)) > 0.5


def test_render_uses_smoothed_normals_by_default(spiral):
    P, V = spiral
    _, S = _jittered(P, amp=1.0, step=1.0)
    pp, nn = S.resample(1.0, normals=S.normals(NORMAL_SIGMA_VX))
    ref = np.stack([V.sample(pp + o * nn) for o in np.array([-2.0, 0.0, 2.0])])  # float64 offsets, as render
    assert np.array_equal(render(S, V, n_layers=3, step=2.0), ref, equal_nan=True)
    pr, nr = S.resample(1.0)
    raw = np.stack([V.sample(pr + o * nr) for o in np.array([-2.0, 0.0, 2.0])])
    assert np.array_equal(render(S, V, n_layers=3, step=2.0, normal_sigma=0), raw, equal_nan=True)


def test_regularize_moves_jittered_vertices_to_sheet_centre(spiral):
    P, V = spiral
    _, S = _jittered(P, amp=1.2)
    d0 = np.abs(P.sheet_distance(S.points[S.valid]))
    R, rep = regularize(S, V)
    d1 = np.abs(P.sheet_distance(R.points[S.valid]))
    assert np.median(d1) < 0.35 * np.median(d0) and np.percentile(d1, 90) < 0.5 * np.percentile(d0, 90)
    assert np.array_equal(R.valid, S.valid)
    mv = np.linalg.norm(R.points - S.points, axis=-1)[S.valid]
    assert mv.max() <= rep["params"]["max_shift"] + 1e-4


def test_regularize_does_not_smooth_across_a_wrap_jump(spiral):
    """Half the grid sits on the next wrap (as after a tracer wrap jump): each half must stay on its own wrap."""
    P, V = spiral
    G = P.surface(0.3 * P.length, 0.6 * P.length, step=2.0, distort=0.0)
    pts = G.points.copy()
    s, z, _ = P._locate(pts[:, 40:].reshape(-1, 3))
    t = np.interp(s, P._sc, P._tc)
    pts[:, 40:] = P.point(np.interp(t + 2 * np.pi, P._tc, P._sc), z).reshape(pts[:, 40:].shape)  # one turn out
    S0 = Surface(pts, G.scale)
    S = Surface(pts + 0.8 * np.nan_to_num(S0.normals(2.0)), G.scale)  # all off-centre by 0.8 vx
    R, _ = regularize(S, V)
    d = np.abs(P.sheet_distance(R.points))
    assert np.nanpercentile(d[:, :40], 95) < 0.4 and np.nanpercentile(d[:, 40:], 95) < 0.4
    s0 = P.unrolled_coords(S.points)[..., 0]
    s1 = P.unrolled_coords(R.points)[..., 0]
    assert np.nanmax(np.abs(s1 - s0)) < 3.0  # no vertex changed wrap


def test_fill_holes_on_sheet_flagged_and_not_across_tears(spiral, tmp_path):
    P, V = spiral
    G = P.surface(0.3 * P.length, 0.6 * P.length, step=2.0, distort=0.0)
    pts = G.points.copy()
    pts[10, 10] = pts[15, 30:32] = pts[16, 30:32] = np.nan  # pinholes (1 and 4 vertices)
    pts[5:12, 50:58] = np.nan  # 56 vertices: larger than max_size
    pts[12:14, 60:62] = np.nan  # a hole over a tear: the papyrus is absent there
    tear = G.points[11:15, 59:63].reshape(-1, 3)
    lo, hi = np.floor(tear.min(0)).astype(int) - 2, np.ceil(tear.max(0)).astype(int) + 3
    vol = P.volume()
    vol[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = int(P.background)
    S = Surface(pts, G.scale)
    lab, sizes = interior_holes(S.valid)
    assert sorted(sizes[sizes > 0].tolist()) == [1, 4, 4, 56]
    F, rep = fill_holes(S, ArrayVolume(vol), max_size=16)
    m = F.filled_mask()
    assert m.sum() == 5 and m[10, 10] and m[15:17, 30:32].all() and not F.valid[12:14, 60:62].any()
    assert not F.valid[5:12, 50:58].any()
    assert np.abs(P.sheet_distance(F.points[m])).max() < 0.5
    F.save(str(tmp_path / "seg"))
    assert np.array_equal(Surface.load(str(tmp_path / "seg")).filled_mask(), m)  # persisted in meta.json
    from unroller.flatten import flatten

    flat = flat_surface(F, flatten(F), out_scale=1.0)
    fm = flat.filled_mask()
    assert 5 <= fm.sum() < 200 and (fm & ~flat.valid).sum() == 0  # flagged in the flattened grid too


def _folded(P, m=25, wrap=0):
    """A grid that runs along the sheet, then folds back over the last ``m`` columns (the tracer defect:
    the same sheet covered twice, the second time with reversed orientation); ``wrap`` = 1 puts the folded
    part one turn further out instead (axis-facing, but new sheet)."""
    G = P.surface(0.3 * P.length, 0.6 * P.length, step=2.0, distort=0.0)
    W = G.shape[1]
    flap = G.points[:, W - 2:W - 2 - m:-1] + 0.5 * np.nan_to_num(G.normals()[:, W - 2:W - 2 - m:-1])
    if wrap:
        s, z, _ = P._locate(flap.reshape(-1, 3))
        t = np.interp(s, P._sc, P._tc) + 2 * np.pi * wrap
        flap = P.point(np.interp(t, P._tc, P._sc), z).reshape(flap.shape)
    meta = {"umbilicus": P.center.tolist(), "up": [0.0, 0.0, 1.0], "sheet_spacing": P.pitch}
    return Surface(np.concatenate([G.points, flap], 1), G.scale, meta), W


def test_remove_folds_cuts_duplicate_flap_only(spiral):
    P, V = spiral
    S, W = _folded(P)
    F, rep = remove_folds(S)
    cut = S.valid & ~F.valid
    assert cut[:, W:].all() and not cut[:, :W - 3].any()  # the flap and its crease, nothing else
    assert rep["flap_components"] == 1 and len(F.meta["folds_removed"]["flat_index"]) == cut.sum()
    S2, _ = _folded(P, wrap=1)  # axis-facing but covering new sheet (a real fold would look like this): kept
    assert np.array_equal(remove_folds(S2)[0].valid, S2.valid)
    S3 = Surface(S.points, S.scale)  # no scroll axis known: unchanged
    assert np.array_equal(remove_folds(S3)[0].valid, S3.valid)
    # a cut flap is never refilled as a "hole", even when it is enclosed
    pts = F.points.copy()
    pts[:, -1], pts[0], pts[-1] = S.points[:, -1], S.points[0], S.points[-1]
    E = Surface(pts, F.scale, F.meta)
    assert interior_holes(E.valid)[1].max() > 0
    G, frep = fill_holes(E, V, max_size=10 ** 6)
    assert frep["candidates"] == 0 and np.array_equal(G.valid, E.valid)
