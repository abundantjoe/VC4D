"""Fast synthetic tests of the ink-side fine-valley texture detector (unroller.crackle)."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from unroller.crackle import (FEATURES, CrackleModel, CrackleParams, binary_scores, block_reduce, blur,
                              crackle_features, crackle_score, edge_distance, masked_blur, overlay, standardize)


def synthetic_stack(seed=0, size=256, n=17, gain=1.0, offset=0.0, ink_side=1):
    """Sheet with fibers; inside a disk a raised ink film (2..4 vx above the surface) cracked into
    a Voronoi network of thin dark valleys.  Returns stack (n, H, W), ink mask, valid mask."""
    rng = np.random.default_rng(seed)
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    s = (n - 1) // 2
    depth = (np.arange(n) - s).astype(np.float32)[:, None, None]
    # fibers: horizontal streaks (recto) of random brightness, a little lateral noise
    fib = gaussian_filter(rng.normal(size=(H, W)), (0.8, 12.0))
    fib = (fib / fib.std()).astype(np.float32)
    sheet = np.exp(-0.5 * (depth / 2.0) ** 2) * (1.0 + 0.25 * fib)
    ink = (yy - H / 2) ** 2 + (xx - W / 2) ** 2 < (0.28 * size) ** 2
    pts = rng.uniform(0, size, (int(size * size / 120), 2))
    d, _ = cKDTree(pts).query(np.stack([yy.ravel(), xx.ravel()], 1), k=2)
    crack = (d[:, 1] - d[:, 0] < 1.6).reshape(H, W)
    film = np.where(crack, 0.15, 1.0).astype(np.float32) * ink
    film = gaussian_filter(film, 0.6)
    prof = np.exp(-0.5 * ((depth - 3.0) / 1.0) ** 2)  # film centred 3 vx above the surface
    vol = sheet + 0.6 * prof * film + rng.normal(0, 0.08, (n, H, W)).astype(np.float32)
    if ink_side < 0:
        vol = vol[::-1]
    vol = gain * vol + offset
    valid = np.ones((H, W), bool)
    valid[:, :8] = False  # an invalid strip, as at segment borders
    vol[:, ~valid] = np.nan
    return vol.astype(np.float32), ink, valid


P = CrackleParams(band=(2.0, 5.0), ref_band=(-5.0, -2.0), sigma_pool=6.0, sigma_bg=48.0, edge_margin=24.0)
PE = CrackleParams(band=(2.0, 5.0), ref_band=(-5.0, -2.0), sigma_pool=4.0, sigma_bg=24.0, edge_margin=16.0)


def textured_sheet(rng, n=17, size=256):
    """No ink: crossing fibers of random brightness, correlated noise, band layers darker than the surface."""
    depth = (np.arange(n) - (n - 1) // 2)[:, None, None]
    fib = gaussian_filter(rng.normal(size=(size, size)), (0.8, 10.0)) + \
        gaussian_filter(rng.normal(size=(size, size)), (10.0, 0.8))
    return (6 * np.exp(-0.5 * (depth / 2) ** 2) * (1 + 0.25 * fib / fib.std())
            + gaussian_filter(rng.normal(size=(n, size, size)), (0, 1.0, 1.0))).astype(np.float32)


def irregular_mask(rng, size=256):
    """Wavy, jagged mask boundary with holes (like a torn segment)."""
    r = np.hypot(*(np.mgrid[0:size, 0:size] - size / 2))
    g, j, h = (gaussian_filter(rng.normal(size=(size, size)), s) for s in (10.0, 1.0, 4.0))
    return (r < 0.4 * size + 14 * g / g.std() + 3 * j / j.std()) & (h / h.std() < 2.5)


def auc(f, lab, valid):
    ok = valid & np.isfinite(f)
    return binary_scores(f[ok], lab[ok])["auc"]


def test_binary_scores_match_bruteforce():
    rng = np.random.default_rng(1)
    s = np.round(rng.normal(size=300), 1)  # many ties
    y = rng.random(300) < 0.3 + 0.2 * (s > 0)
    r = binary_scores(s, y, threshold=0.0)
    pos, neg = s[y], s[~y]
    mw = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert r["auc"] == pytest.approx(mw, abs=1e-12)
    best = 0.0
    for t in np.unique(s):
        pr = s >= t
        best = max(best, 2 * (pr & y).sum() / (pr.sum() + y.sum()))
    assert r["best_f1"] == pytest.approx(best)
    pr = s >= 0.0
    assert r["f1_at_threshold"] == pytest.approx(2 * (pr & y).sum() / (pr.sum() + y.sum()))
    # AP of a perfect ranking is 1, of a reversed one ~ prevalence
    assert binary_scores(y.astype(float), y)["ap"] == pytest.approx(1.0)


def test_blur_cascade_matches_gaussian_and_masks():
    for sigma in (6.0, 10.0, 24.0):  # box cascade: same mass and variance as the Gaussian
        d = np.zeros((201, 201), np.float32)
        d[100, 100] = 1
        k = blur(d, sigma)
        r2 = ((np.arange(201) - 100.0) ** 2)[:, None]
        assert k.sum() == pytest.approx(1.0, rel=1e-4)
        assert (k * r2).sum() == pytest.approx(sigma ** 2, rel=0.08)
    x = np.random.default_rng(2).normal(size=(200, 220)).astype(np.float32)
    a, b = blur(x, 10.0), gaussian_filter(x, 10.0, mode="reflect")
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98
    v = np.ones_like(x, bool)
    v[:, :50] = False
    m = masked_blur(np.where(v, 3.0, 1e6), v, 12.0)  # invalid values must not leak in
    assert np.nanmax(np.abs(m - 3.0)) < 1e-3


def test_features_find_synthetic_crackle():
    st, ink, valid = synthetic_stack(0)
    F = crackle_features(st, None, P)
    assert not F["valid"][:, :8].any() and F["valid"][:, 8:].all()
    assert auc(F["crack_energy"], ink, F["valid"]) > 0.9
    assert auc(F["relief"], ink, F["valid"]) > 0.9
    # the film hides the fibers on the band: their coherence there drops
    Fb = crackle_features(st, None, CrackleParams(**{**P.__dict__, "fiber_band": (2.0, 4.0)}), ("coherence_loss",))
    assert auc(Fb["coherence_loss"], ink, Fb["valid"]) > 0.7


@pytest.mark.parametrize("shape", ["disk", "irregular"])
def test_no_response_at_mask_edges(shape):
    """The zero fill outside the mask must not look like cracks.  No ink, a sheet whose band layers
    are darker than the surface (as in real renders): features stay flat up to the mask edge
    (before mask-aware pooling crack_energy was +4.6 z on the outer 5 px rim).  ``irregular``: a
    textured sheet (crossing fibers) under a jagged mask with holes, zeros outside it."""
    rng = np.random.default_rng(4 if shape == "disk" else 0)
    if shape == "disk":
        depth = (np.arange(17) - 8.0)[:, None, None]
        st = gaussian_filter(rng.normal(size=(17, 256, 256)), (0, 1.0, 1.0)) + 6 * np.exp(-0.5 * (depth / 2) ** 2)
        valid = np.hypot(*(np.mgrid[0:256, 0:256] - 128.0)) < 110
    else:
        st = textured_sheet(rng)
        valid = irregular_mask(rng)
        st[:, ~valid] = 0
    F = crackle_features(st.astype(np.float32), valid, PE)
    d = edge_distance(valid)
    rim, core = valid & (d <= 5), valid & (d > 30)
    assert rim.sum() > 2000 and core.sum() > 2000
    for k in FEATURES:
        z = standardize(F, (k,))[0]
        assert abs(np.nanmean(z[rim]) - np.nanmean(z[core])) < 1.0, k


def test_edge_suppression_removes_rim_enrichment():
    """Near the mask edge stroke-scale pooling averages fewer pixels, so the score spreads more and its
    top 5% is enriched there even without ink (real segments: 1.17-1.50x within 64 px of the edge).
    crackle_score suppresses edge_margin px; the band just inside is not enriched."""
    m = CrackleModel(weights=(1.0, 0.0, 0.0, 0.0, 0.0))  # the fitted models are ~crack_energy alone

    def enrichment(sc, d, lo, hi):
        ok = np.isfinite(sc)
        top = ok & (sc >= np.quantile(sc[ok], 0.95))
        b = ok & (d > lo) & (d <= hi)
        return (top & b).sum() / top.sum() / (b.sum() / ok.sum())

    before, after = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        st, valid = textured_sheet(rng), irregular_mask(rng)
        st[:, ~valid] = 0
        sc, F = crackle_score(st, valid, m, PE)
        d = edge_distance(valid)
        assert np.array_equal(F["scored"], valid & (d > 16)) and np.isfinite(sc[F["scored"]]).all()
        assert np.isnan(sc[~F["scored"]]).all()
        s0 = m.predict(standardize(F))
        s0[~F["valid"]] = np.nan
        before.append(enrichment(s0, d, 0, 16))
        after.append(enrichment(sc, d, 16, 32))
    assert np.mean(before) > 1.15 and np.mean(after) < 1.15, (before, after)


def test_edge_distance_counts_image_border():
    v = np.ones((5, 7), bool)
    v[2, 3] = False
    d = edge_distance(v)
    assert d[0, 0] == 1 and d[2, 3] == 0 and d[2, 2] == 1 and d[2, 1] == 2 and d[2, 0] == 1


def test_ink_side_mirror_is_equivalent():
    st, ink, _ = synthetic_stack(3, size=128)
    stm, _, _ = synthetic_stack(3, size=128, ink_side=-1)
    a = crackle_features(st, None, P)
    b = crackle_features(stm, None, CrackleParams(**{**P.__dict__, "ink_side": -1}))
    for k in FEATURES:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-4, atol=1e-4, equal_nan=True)


def test_model_transfers_across_contrast_and_roundtrips():
    st, ink, _ = synthetic_stack(0)
    F = crackle_features(st, None, P)
    m = CrackleModel.fit(standardize(F), ink, F["valid"], params=P, n_samples=20000)
    assert m.info["train_auc"] > 0.9
    # a different "segment": new crack pattern and noise, other gain/offset (like uint8 vs uint16)
    st2, ink2, _ = synthetic_stack(7, gain=180.0, offset=2000.0)
    sc, F2 = crackle_score(st2, None, m)
    assert auc(sc, ink2, F2["valid"]) > 0.9
    ok = F2["valid"]
    assert binary_scores(sc[ok], ink2[ok], m.threshold)["f1_at_threshold"] > 0.6
    # control: the score does not follow a label rotated away from the ink
    assert abs(auc(sc, np.roll(ink2, 128, axis=1), ok) - 0.5) < abs(auc(sc, ink2, ok) - 0.5)
    m2 = CrackleModel.from_json(m.to_json())
    assert m2.top_fraction == m.top_fraction
    np.testing.assert_allclose(m2.predict(standardize(F2))[F2["scored"]], sc[F2["scored"]])


def test_percentile_threshold_flags_training_fraction():
    st, ink, _ = synthetic_stack(0)
    F = crackle_features(st, None, P)
    m = CrackleModel.fit(standardize(F), ink, F["valid"], params=P, n_samples=20000)
    ok = F["valid"]
    assert m.top_fraction == pytest.approx(np.mean(m.predict(standardize(F))[ok] >= m.threshold), abs=1e-6)
    s = np.random.default_rng(0).normal(7.0, 3.0, 20000)  # another segment: other score scale
    assert np.mean(s >= m.percentile_threshold(s)) == pytest.approx(m.top_fraction, abs=1e-3)
    assert CrackleModel(threshold=0.3).percentile_threshold(s) == 0.3  # no top_fraction: absolute threshold


def test_block_reduce_majority():
    s = np.zeros((64, 64), np.float32)
    s[:32, :32] = 1
    lab = np.zeros((64, 64), bool)
    lab[:32, :20] = True  # 62% of the first block
    v = np.ones((64, 64), bool)
    v[32:, 32:] = False
    bs, bl = block_reduce(s, lab, v, 32)
    assert len(bs) == 3 and bl.sum() == 1 and bs[bl][0] == 1.0


def test_band_outside_stack_raises():
    with pytest.raises(ValueError):
        CrackleParams(band=(3.0, 12.0)).layers(17, (3.0, 12.0))


def test_overlay_png(tmp_path):
    st, ink, valid = synthetic_stack(0, size=128)
    sc = blur(ink.astype(np.float32), 3)
    img = overlay(st[8], sc, ink, valid, str(tmp_path / "o.png"), max_side=64)
    assert img.shape == (64, 64, 3) and (tmp_path / "o.png").exists()
    assert ((img == [0, 230, 255]).all(-1)).any()  # label contour drawn
    assert (img[..., 0].astype(int) - img[..., 2] > 60).any()  # warm overlay where the score is high
    sc[20:22, 20:22] = np.nan  # odd sizes are cropped to whole cells; a NaN does not blank its cell
    img = overlay(st[8][:127, :125], sc[:127, :125], ink[:127, :125], valid[:127, :125], max_side=50)
    assert img.shape == (42, 41, 3)
    assert not img[:, :2].any() and (img[:, 3:].sum(-1) == 0).mean() < 0.05 and img[7, 7].any()


def test_cli_on_tif_layers(tmp_path):
    import json

    import tifffile
    from PIL import Image

    from unroller.cli import main

    st, ink, valid = synthetic_stack(0, size=160)
    F = crackle_features(st, None, P)
    m = CrackleModel.fit(standardize(F), ink, F["valid"], params=P, n_samples=20000)
    (tmp_path / "model.json").write_text(m.to_json())
    st2, ink2, _ = synthetic_stack(5, size=160)
    d = tmp_path / "layers"
    d.mkdir()
    u16 = np.clip(np.nan_to_num(st2 * 20000 + 10000), 0, 65535).astype(np.uint16)
    for i, layer in enumerate(u16):  # numbered like official layers 24..40, surface = 32
        tifffile.imwrite(str(d / f"{24 + i:02d}.tif"), layer)
    pad = np.zeros((256, 256), np.uint8)  # labels zero-padded beyond the grid, as in villa
    pad[:160, :160] = ink2 * 255
    Image.fromarray(pad).save(tmp_path / "lab.png")
    msk = np.zeros((160, 160), np.uint8)
    msk[:, 8:] = 255
    Image.fromarray(msk).save(tmp_path / "mask.png")
    main(["crackle", str(d), str(tmp_path / "out"), "--model", str(tmp_path / "model.json"),
          "--labels", str(tmp_path / "lab.png"), "--mask", str(tmp_path / "mask.png")])
    rep = json.loads((tmp_path / "out" / "crackle_report.json").read_text())
    assert rep["surface_layer"] == 32 and rep["labels"]["auc"] > 0.9 and rep["edge_margin_px"] == 24
    sc = tifffile.imread(str(tmp_path / "out" / "crackle_score.tif"))
    assert sc.shape == (160, 160) and np.isnan(sc[:, :32]).all() and np.isnan(sc[:24]).all()
    assert np.isfinite(sc[24:136, 32:136]).all() and rep["scored_fraction"] == pytest.approx(112 * 104 / 160 ** 2)
    # per-segment operating point (the synthetic scores saturate, so ties can flag a little more)
    op = rep["operating_point"]
    assert op["threshold"] == pytest.approx(np.nanquantile(sc, 1 - m.top_fraction), rel=1e-6)
    assert op["fraction_flagged"] >= m.top_fraction - 1e-3
    assert (tmp_path / "out" / "crackle_overlay.png").exists()
