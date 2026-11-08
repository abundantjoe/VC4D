"""Tests for the evaluation code (synthetic data plus the shipped window / exclusion lists).  python -m pytest"""
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
import face_table as FT  # noqa: E402
import reproduce  # noqa: E402
from ink9face import labels  # noqa: E402

# First-Letters-eligible scrolls plus the held-out exam scroll, by number only (the ids are spelled out at run time)
_EXCLUDED_NUMS = ("0125", "0175", "0191", "0211", "0257", "0268", "0358", "0483", "0490", "0800", "0813", "0826", "0846",
                  "1203", "1218", "1545", "0841")
ELIGIBLE = re.compile("PHerc(" + "|".join(_EXCLUDED_NUMS) + ")")


def synth_window(seed, gain=1.0, n=300):
    """Label with blobs; the anti_normal map sees them, the native map is noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    ink = np.zeros((n, n), bool)
    for _ in range(12):
        cy, cx = rng.integers(20, n - 20, 2)
        ink |= (yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(36, 200)
    anti = np.clip(0.3 + gain * 0.4 * ink + rng.normal(0, 0.15, (n, n)), 0, 1).astype(np.float32)
    nat = np.clip(0.4 + rng.normal(0, 0.15, (n, n)), 0, 1).astype(np.float32)
    return ink, np.ones((n, n), bool), nat, anti


def test_windows_list_is_public_and_complete():
    W = reproduce.load_windows()
    assert len(W) == 99 and len({x["w"] for x in W}) == 99
    counts = {s: sum(x["scroll"] == s for x in W) for s in ("PHerc0500P2", "PHerc0814", "PHerc0343P", "PHerc0009B", "PHerc0139")}
    assert counts == {"PHerc0500P2": 29, "PHerc0814": 14, "PHerc0343P": 15, "PHerc0009B": 30, "PHerc0139": 11}
    assert not any("46527" in x["seg"] for x in W)
    for x in W:
        assert x["scroll"] in labels.SCROLLS and len(x["crop"]) == 4 and x["crop"][2] == x["crop"][3] == 53
        assert not ELIGIBLE.search(x["scroll"])


def test_shipped_lists_name_no_eligible_or_exam_scroll():
    for f in ("windows.json", "exclusions.json"):
        assert not ELIGIBLE.search(open(os.path.join(ROOT, "eval", f)).read())


def test_exclusions_prefix_matching():
    ex = FT.load_exclusions()
    assert FT.trained(ex, "PHerc0500P2", "trace", "anything")                                    # TRACE saw everything
    assert FT.trained(ex, "PHerc0500P2", "reader", "20250611132918-z_dbg_gen_00302_inp_hr")     # Reader v3 = 'reader'
    assert not FT.trained(ex, "PHerc0500P2", "reader_v3", "20250825181859--1")
    assert not FT.trained(ex, "PHerc0500P2", "d9v4C", "20250825181859--1")
    assert FT.trained(ex, "PHerc0500P2", "d9v4C", "20250825181907-0")
    assert not FT.trained(ex, "PHerc0500P2", "dense_native", "20250716055229-z_dbg_gen_00371_inp_hr")
    assert not FT.trained(ex, "PHerc0500P2", "ink9avg", "20250611132918-z_dbg_gen_00302_inp_hr")
    assert FT.trained(ex, "PHerc0814", "ink9avg", "20260226000000-46527_2um_try2")


def test_score_window_and_summary_pick_anti_normal():
    rows = {}
    for i in range(6):
        ink, m, nat, anti = synth_window(i)
        maps = {"klavis": (nat, anti, None), "trace": (nat, anti, None)}
        rows[f"seg{i // 2}_w{i % 2}"] = FT.score_window("PHerc0500P2", f"seg{i // 2}", ink, m, i % 2 == 0, maps, [])
    S = FT.summarise(rows)
    a = S["agreement"]["truth:PHerc0500P2"]
    assert a["anti_normal"]["correct"] == 6 and a["native"]["correct"] == 0 and a["toward_axis"]["correct"] == 3
    d = S["models"]["klavis:ALL"]["auc"]["anti_minus_native"]
    assert S["models"]["klavis:ALL"]["n_meshes"] == 3 and d["better"] == 3 and d["mean"] > 0.2 and d["ci"][0] > 0
    assert S["band_scroll_vote"]["PHerc0500P2:klavis"]["says"] == "reversed"


def test_contaminated_model_is_not_scored_or_used_as_truth():
    ink, m, nat, anti = synth_window(0)
    r = FT.score_window("PHerc0500P2", "seg0", ink, m, False, {"klavis": (nat, anti, None)}, ["klavis"])
    assert "truth" not in r
    S = FT.summarise({"seg0_w0": r})
    assert "klavis:ALL" not in S["models"]


def test_covered_mask_restricts_scoring():
    ink, m, nat, anti = synth_window(1)
    cov = np.zeros_like(m); cov[:, :150] = True
    r = FT.score_window("PHerc0814", "s", ink, m, True, {"v8": (nat, anti, cov)}, [])
    r2 = FT.score_window("PHerc0814", "s", ink[:, :150], m[:, :150], True, {"v8": (nat[:, :150], anti[:, :150], None)}, [])
    assert r["scores"]["v8"]["reversed"]["auc"] == pytest.approx(r2["scores"]["v8"]["reversed"]["auc"])


def test_compare_reports_differences():
    a = {"x": 1.0, "y": [1, 2], "z": {"k": "a"}}
    assert FT.compare(a, json.loads(json.dumps(a))) == []
    d = FT.compare(a, {"x": 1.1, "y": [1, 2], "z": {"k": "b"}})
    assert len(d) == 2


def test_table_from_maps_folder(tmp_path):
    """reproduce.py table: labels cache + public maps layout -> summary; --check against itself passes."""
    W = [dict(w=f"seg{i // 2}_w{i % 2}", scroll="PHerc0500P2", seg=f"seg{i // 2}", crop=[0, 0, 53, 53], axis_xy=[0, 0],
              toward_axis_is_reversed=bool(i % 2)) for i in range(4)]
    json.dump(dict(windows=W), open(tmp_path / "windows.json", "w"))
    json.dump(dict(trained_on={"PHerc0500P2": {"trace": "ALL"}}), open(tmp_path / "excl.json", "w"))
    (tmp_path / "cache" / "labels").mkdir(parents=True)
    for i, x in enumerate(W):
        ink, m, nat, anti = synth_window(10 + i)
        np.savez_compressed(tmp_path / "cache" / "labels" / (x["w"] + ".npz"), ink=ink, acc=m, valid=m,
                            meta=json.dumps(dict(toward_axis_is_reversed=x["toward_axis_is_reversed"])))
        for mod in ("ink9_42", "ink9_43", "klavis", "trace"):
            (tmp_path / "maps" / mod).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tmp_path / "maps" / mod / (x["w"] + ".npz"), native=(nat * 255).astype(np.uint8),
                                anti_normal=(anti * 255).astype(np.uint8))
    args = ["table", "--cache", str(tmp_path / "cache"), "--maps", str(tmp_path / "maps"), "--out", str(tmp_path / "out.json"),
            "--windows", str(tmp_path / "windows.json"), "--exclusions", str(tmp_path / "excl.json")]
    assert reproduce.main(args) == 0
    S = json.load(open(tmp_path / "out.json"))["summary"]
    assert S["agreement"]["truth:ALL"]["anti_normal"]["correct"] == 4
    assert S["agreement"]["truth:ALL"]["toward_axis"]["correct"] == 2
    assert "ink9avg:ALL" in S["models"] and "trace:ALL" in S["models"]
    assert reproduce.main(args + ["--check", str(tmp_path / "out.json")]) == 0


def test_label_transfer_synthetic_two_grids():
    """Coarse grid and a 4x finer label grid of the same plane, label image = column index stripes: the transferred
    label must follow the coarse grid columns and pass the 3D gate."""
    H9, W9, k = 12, 10, 4
    r, c = np.meshgrid(np.arange(H9, dtype=float), np.arange(W9, dtype=float), indexing="ij")
    P9 = np.stack([c * 20, r * 20, np.full_like(r, 50.0)], -1)
    H2, W2 = H9 * k, W9 * k
    r2, c2 = np.meshgrid(np.arange(H2, dtype=float), np.arange(W2, dtype=float), indexing="ij")
    # label grid point g2 + 0.5 = k (g9 + 0.5): coarse coords of a label node = (g2 + 0.5) / k - 0.5, in label voxels (x k)
    P2 = np.stack([((c2 + 0.5) / k - 0.5) * 20 * k, ((r2 + 0.5) / k - 0.5) * 20 * k, np.full_like(r2, 50.0 * k)], -1)
    lab = np.zeros((H2 * 20, W2 * 20), np.uint8); lab[:, : W2 * 10] = 255          # left half inked
    reg = labels.segment_registration(P9, P2)
    assert np.nanmax(reg["node3d"]) < 1e-6
    raw, acc, valid, val = labels.transfer_crop(P9, P2, lab, reg, 1, 1, 8, 8)
    assert val["pass"] and valid.all() and raw.shape == (161, 161)
    cols = np.arange(161) / 20 + 1                                                   # coarse column of each pixel
    edge = (W2 / 2 + 0.5) / k - 0.5                                                  # label px 20*g2 < 20*W2/2
    assert ((raw[80] > 127) == (cols < edge)).mean() > 0.97
