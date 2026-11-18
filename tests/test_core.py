"""Fast unit tests (no dataset needed). Run: python -m pytest -q"""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from bevdiag.diagnose import (average_precision, best_f1_threshold, count_points_in_boxes,
                              count_points_in_boxes_slow, match, range_bucket)
from bevdiag.fixes import operating_point
from bevdiag.model import ms_deform_attn
from bevdiag.resources import compute_budget


# ----------------------------------------------------------------------------- model ops
def _naive_deform_attn(value, shapes, loc, w):
    """Loop implementation of bilinear sampling with zero padding, align_corners=False."""
    bs, _, heads, c = value.shape
    _, nq, _, L, P, _ = loc.shape
    out = torch.zeros(bs, nq, heads, c)
    start = 0
    starts = []
    for h, wd in shapes:
        starts.append(start)
        start += h * wd
    for b in range(bs):
        for q in range(nq):
            for m in range(heads):
                for lv, (H, W) in enumerate(shapes):
                    v = value[b, starts[lv]:starts[lv] + H * W, m].view(H, W, c)
                    for p in range(P):
                        x = loc[b, q, m, lv, p, 0] * W - 0.5
                        y = loc[b, q, m, lv, p, 1] * H - 0.5
                        x0, y0 = math.floor(x), math.floor(y)
                        acc = torch.zeros(c)
                        for dy in (0, 1):
                            for dx in (0, 1):
                                xi, yi = x0 + dx, y0 + dy
                                wt = (1 - abs(x - xi)) * (1 - abs(y - yi))
                                if 0 <= xi < W and 0 <= yi < H:
                                    acc += wt * v[yi, xi]
                        out[b, q, m] += w[b, q, m, lv, p] * acc
    return out.view(bs, nq, heads * c)


def test_ms_deform_attn_matches_naive_bilinear():
    torch.manual_seed(0)
    shapes = [(5, 7), (3, 4)]
    nv = sum(h * w for h, w in shapes)
    value = torch.randn(2, nv, 2, 3)
    loc = torch.rand(2, 4, 2, 2, 3, 2) * 1.2 - 0.1  # includes out-of-bounds samples
    w = torch.rand(2, 4, 2, 2, 3)
    got = ms_deform_attn(value, torch.tensor(shapes), loc, w)
    torch.testing.assert_close(got, _naive_deform_attn(value, shapes, loc, w), atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------------- geometry
def test_vectorised_point_count_matches_devkit():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-20, 20, size=(4000, 3)).astype(np.float32)
    pts[:, 2] = rng.uniform(-3, 3, 4000)
    boxes = np.zeros((25, 9))
    boxes[:, :2] = rng.uniform(-15, 15, (25, 2))
    boxes[:, 2] = rng.uniform(-1, 1, 25)
    boxes[:, 3:6] = rng.uniform(0.5, 6, (25, 3))
    boxes[:, 6] = rng.uniform(-np.pi, np.pi, 25)
    for enl in (0.0, 0.5):
        np.testing.assert_array_equal(count_points_in_boxes(pts, boxes, enl),
                                      count_points_in_boxes_slow(pts, boxes, enl))


def test_bev_corners_match_devkit_box():
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    from bevdiag.viz import box_corners_3d

    x, y, z, w, l, h, yaw = 3.0, -7.0, 0.5, 2.0, 9.0, 3.5, 0.7
    ours = box_corners_3d(x, y, z, w, l, h, yaw)
    ref = Box([x, y, z], [w, l, h], Quaternion(axis=[0, 0, 1], radians=yaw)).corners().T
    # same 8 points (order may differ)
    d = np.linalg.norm(ours[:, None] - ref[None], axis=-1)
    assert np.allclose(d.min(1), 0, atol=1e-9)


def test_lidar_to_global_yaw_convention():
    from bevdiag.data import SampleInfo
    from bevdiag.infer import lidar_boxes_to_global

    info = SampleInfo(*(["t", "s", "n", "d", 0, "l", "p"] + [np.eye(4), np.eye(4)] + [[]] * 5 + [np.zeros(18)]))
    b = np.array([[1.0, 2.0, 0.0, 2.0, 4.5, 1.6, 0.3, 1.0, 0.0]])
    box = lidar_boxes_to_global(b, info)[0]
    assert np.allclose(box.wlh, [2.0, 4.5, 1.6])
    assert math.isclose(box.orientation.yaw_pitch_roll[0], -0.3 - math.pi / 2, abs_tol=1e-9)


# ----------------------------------------------------------------------------- metrics vs devkit
def _devkit_ap(gt_rows, pr_rows, dist_th=2.0):
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.utils import center_distance
    from nuscenes.eval.detection.algo import accumulate, calc_ap
    from nuscenes.eval.detection.data_classes import DetectionBox

    def boxes(rows, is_pred):
        eb = EvalBoxes()
        by = {}
        for r in rows:
            by.setdefault(f"s{r['sample_idx']}", []).append(DetectionBox(
                sample_token=f"s{r['sample_idx']}", translation=(r["x"], r["y"], 0.0), size=(2, 4, 1.5),
                rotation=(1, 0, 0, 0), velocity=(0, 0), detection_name="car",
                detection_score=r.get("score", -1.0) if is_pred else -1.0, attribute_name="vehicle.parked"))
        for k in {f"s{i}" for i in range(5)}:
            eb.add_boxes(k, by.get(k, []))
        return eb

    md = accumulate(boxes(gt_rows, False), boxes(pr_rows, True), "car", center_distance, dist_th)
    return calc_ap(md, 0.1, 0.1)


def test_ap_and_matching_equal_devkit():
    rng = np.random.default_rng(3)
    gt_rows, pr_rows = [], []
    for s in range(5):
        for _ in range(12):
            gx, gy = rng.uniform(-40, 40, 2)
            gt_rows.append(dict(sample_idx=s, cls="car", x=gx, y=gy))
            if rng.random() < 0.8:  # a detection with noise
                pr_rows.append(dict(sample_idx=s, cls="car", x=gx + rng.normal(0, 1.2), y=gy + rng.normal(0, 1.2),
                                    score=float(rng.random())))
        for _ in range(5):  # pure false positives
            pr_rows.append(dict(sample_idx=s, cls="car", x=rng.uniform(-40, 40), y=rng.uniform(-40, 40),
                                score=float(rng.random() * 0.6)))
    gt, pr = pd.DataFrame(gt_rows), pd.DataFrame(pr_rows)
    for df in (gt, pr):
        df["dist"] = np.hypot(df.x, df.y)
        df["l"] = 4.0
        df["yaw"] = 0.0
    gtm, prm = match(gt, pr)
    ours = average_precision(prm.score.to_numpy(), prm.tp.to_numpy(), len(gtm))
    assert ours == pytest.approx(_devkit_ap(gt_rows, pr_rows), abs=1e-9)
    assert gtm.matched.sum() == prm.tp.sum()


def test_ap_edge_cases():
    assert np.isnan(average_precision(np.array([]), np.array([], bool), 0))
    assert average_precision(np.array([]), np.array([], bool), 3) == 0.0
    assert average_precision(np.array([0.9, 0.8]), np.array([True, True]), 2) == pytest.approx(1.0)


def test_operating_point_counts():
    gt = pd.DataFrame(dict(sample_idx=[0, 0, 0], cls=["car"] * 3, x=[0.0, 10, 20], y=[0.0, 0, 0]))
    pr = pd.DataFrame(dict(sample_idx=[0, 0, 0], cls=["car"] * 3, x=[0.5, 10.2, -30], y=[0.0, 0, 0],
                           score=[0.9, 0.2, 0.95]))
    for df in (gt, pr):
        df["dist"] = np.hypot(df.x, df.y)
        df["l"] = 4.0
        df["yaw"] = 0.0
    gtm, prm = match(gt, pr)
    op = operating_point(gtm, prm, prm.score >= 0.5)
    assert (op["tp"], op["fp"], op["fn"]) == (1, 1, 2)
    assert best_f1_threshold(gtm, prm) <= 0.2  # keeping the low-score TP helps F1


def test_range_bucket_edges():
    b = range_bucket([0, 9.99, 10, 49.9, 50, 79.9, 80, 300])
    assert list(b.astype(str)) == ["0-10", "0-10", "10-20", "40-50", "50-60", "60-80", "80+", "80+"]


# ----------------------------------------------------------------------------- data + resources
def test_can_bus_vector_reproduces_reference_quirk():
    from bevdiag.data import can_bus_vector

    class FakeCan:
        def get_messages(self, scene, kind):
            mk = lambda t, v: {"accel": [v] * 3, "orientation": [1, 0, 0, 0], "pos": [t, 0, 0],
                               "rotation_rate": [v] * 3, "utime": t, "vel": [v] * 3}
            return [mk(100, 1.0), mk(200, 2.0), mk(300, 3.0)]

    v = can_bus_vector(FakeCan(), "scene", 250)
    assert v.shape == (18,)
    assert v[0] == 200  # position from the last message before the sample
    assert np.all(v[7:16] == 3.0)  # accel/rate/vel from the *next* message, like the reference converter
    assert np.all(v[16:] == 0)


def test_budget_leaves_headroom():
    b = compute_budget(0.9)
    import os

    n = os.cpu_count()
    assert 1 <= b.cpu_threads <= max(1, int(n * 0.9)) < n + 1
    assert b.gpu_mem_fraction <= 0.9


# ----------------------------------------------------------------------------- bigger-model building blocks
def test_dcn_with_zero_offsets_is_half_a_plain_conv():
    from bevdiag.model import ModulatedDeformConv

    torch.manual_seed(0)
    m = ModulatedDeformConv(4, 5, stride=2)
    torch.nn.init.zeros_(m.conv_offset.weight)
    torch.nn.init.zeros_(m.conv_offset.bias)  # offset 0, mask sigmoid(0) = 0.5
    torch.nn.init.normal_(m.weight)
    x = torch.randn(2, 4, 9, 11)
    ref = 0.5 * torch.nn.functional.conv2d(x, m.weight, stride=2, padding=1)
    torch.testing.assert_close(m(x), ref, atol=1e-5, rtol=1e-5)


def test_fpn_levels_and_strides():
    from bevdiag.model import FPN

    fpn = FPN((512, 1024, 2048), 256, 4)
    feats = [torch.randn(1, c, s, s * 2) for c, s in ((512, 32), (1024, 16), (2048, 8))]
    outs = fpn(feats)
    assert [tuple(o.shape[2:]) for o in outs] == [(32, 64), (16, 32), (8, 16), (4, 8)]


def test_temporal_fusion_boosts_persistent_and_penalises_one_off():
    from bevdiag.data import SampleInfo
    from bevdiag.fixes import temporal_score_fusion

    def info(t):
        return SampleInfo(*(["tok", "scene", "n", "d", t, "l", "p"] + [np.eye(4), np.eye(4)] + [[]] * 5 +
                            [np.zeros(18)]))

    box = np.array([[10.0, 5.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]])
    ghost = np.array([[-20.0, 30.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]])
    f0 = {"boxes": box, "scores": np.array([0.9], np.float32), "labels": np.array([0])}
    f1 = {"boxes": np.r_[box, ghost], "scores": np.array([0.3, 0.3], np.float32), "labels": np.array([0, 0])}
    out = temporal_score_fusion([info(0), info(500_000)], [f0, f1], w=0.5, miss_penalty=0.2)
    assert out[0]["scores"][0] == pytest.approx(0.9)  # first frame of a scene is untouched
    assert out[1]["scores"][0] == pytest.approx(0.6)  # 0.5*0.3 + 0.5*0.9
    assert out[1]["scores"][1] == pytest.approx(0.24)  # unmatched: 0.3 * (1 - 0.2)
    np.testing.assert_array_equal(out[1]["boxes"], f1["boxes"])  # boxes never move
