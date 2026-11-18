"""Error slicing for BEV detections: range, class, size, lighting, visibility + failure tagging.

Matching follows the nuScenes detection metric (greedy by score, BEV center distance, per class)
so that slice numbers are comparable to the official benchmark, but *without* the 50 m class
range filter, so we can see what happens past 50 m -- the regime highway trucking cares about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion

from .data import CLASS_NAMES, SampleInfo, load_lidar_points

RANGE_BINS = [0, 10, 20, 30, 40, 50, 60, 80, np.inf]
RANGE_LABELS = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
LENGTH_BINS = [0, 2, 5, 8, 12, np.inf]
LENGTH_LABELS = ["<2 m", "2-5 m", "5-8 m", "8-12 m", ">12 m"]
LARGE_VEHICLES = ("truck", "trailer", "bus", "construction_vehicle")
CLASS_RANGE = {"car": 50, "truck": 50, "bus": 50, "trailer": 50, "construction_vehicle": 50,
               "pedestrian": 40, "motorcycle": 40, "bicycle": 40, "traffic_cone": 30, "barrier": 30}
BEV_HALF_EXTENT = 51.2  # BEVFormer-tiny grid: [-51.2, 51.2] m in x and y
MATCH_DIST = 2.0
NEAR_DIST = 4.0


def range_bucket(d):
    return pd.cut(np.asarray(d, dtype=float), RANGE_BINS, labels=RANGE_LABELS, right=False)


def length_bucket(l):
    return pd.cut(np.asarray(l, dtype=float), LENGTH_BINS, labels=LENGTH_LABELS, right=False)


# --------------------------------------------------------------------------- tables
def _lidar_box(b) -> Box:
    return Box(b[:3].tolist(), b[3:6].tolist(), Quaternion(axis=[0, 0, 1], radians=-b[6] - np.pi / 2))


def count_points_in_boxes_slow(points: np.ndarray, boxes: np.ndarray, enlarge: float = 0.0) -> np.ndarray:
    """Reference implementation using the devkit's points_in_box (used in tests)."""
    out = np.zeros(len(boxes), dtype=int)
    for i, b in enumerate(boxes):
        bb = b.copy()
        bb[3:6] = bb[3:6] + 2 * enlarge
        out[i] = int(points_in_box(_lidar_box(bb), points.T).sum())
    return out


def count_points_in_boxes(points: np.ndarray, boxes: np.ndarray, enlarge: float = 0.0,
                          chunk: int = 64) -> np.ndarray:
    """Vectorised: number of LiDAR points (N,3, lidar frame) inside each box (M,9 mmdet3d boxes).

    Boxes are (x, y, z_center, w, l, h, yaw_mmdet3d, ...); every side is grown by `enlarge` metres.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    pts = points[:, :3].astype(np.float32)
    out = np.zeros(len(boxes), dtype=int)
    for s in range(0, len(boxes), chunk):
        b = boxes[s:s + chunk].astype(np.float32)
        yaw = -b[:, 6] - np.pi / 2  # heading of the box length axis
        c, si = np.cos(yaw), np.sin(yaw)
        d = pts[None, :, :] - b[:, None, :3]  # (m, n, 3)
        lx = d[..., 0] * c[:, None] + d[..., 1] * si[:, None]   # along length
        ly = -d[..., 0] * si[:, None] + d[..., 1] * c[:, None]  # along width
        half_l = b[:, 4:5] / 2 + enlarge
        half_w = b[:, 3:4] / 2 + enlarge
        half_h = b[:, 5:6] / 2 + enlarge
        inside = (np.abs(lx) <= half_l) & (np.abs(ly) <= half_w) & (np.abs(d[..., 2]) <= half_h)
        out[s:s + chunk] = inside.sum(1)
    return out


def gt_table(nusc, infos: list[SampleInfo]) -> pd.DataFrame:
    rows = []
    for si, info in enumerate(infos):
        g2l = np.linalg.inv(info.lidar2global)
        sample = nusc.get("sample", info.token)
        for ann_tok in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            name = category_to_detection_name(ann["category_name"])
            if name is None or ann["num_lidar_pts"] + ann["num_radar_pts"] == 0:
                continue
            c = g2l @ np.r_[ann["translation"], 1.0]
            q = Quaternion(matrix=g2l[:3, :3]) * Quaternion(ann["rotation"])
            w, l, h = ann["size"]
            rows.append(dict(sample_idx=si, sample_token=info.token, scene=info.scene_name,
                             lighting=info.tags["lighting"], weather=info.tags["weather"],
                             cls=name, x=c[0], y=c[1], z=c[2], w=w, l=l, h=h,
                             yaw=q.yaw_pitch_roll[0], dist=float(np.hypot(c[0], c[1])),
                             num_lidar_pts=ann["num_lidar_pts"],
                             visibility=int(nusc.get("visibility", ann["visibility_token"])["token"]),
                             ann_token=ann_tok))
    df = pd.DataFrame(rows)
    df["range_bin"] = range_bucket(df.dist)
    df["length_bin"] = length_bucket(df.l)
    df["in_bev_grid"] = (df.x.abs() < BEV_HALF_EXTENT) & (df.y.abs() < BEV_HALF_EXTENT)
    df["n_cams"] = 0
    return df


def pred_table(infos: list[SampleInfo], preds: list[dict], with_lidar: bool = True) -> pd.DataFrame:
    rows = []
    for si, (info, p) in enumerate(zip(infos, preds)):
        b = p["boxes"]
        if with_lidar and len(b):
            pts = load_lidar_points(info.lidar_path)
            npts = count_points_in_boxes(pts, b)
            npts_e = count_points_in_boxes(pts, b, enlarge=0.5)
        else:
            npts = npts_e = np.full(len(b), -1)
        for j in range(len(b)):
            rows.append(dict(sample_idx=si, sample_token=info.token, scene=info.scene_name,
                             lighting=info.tags["lighting"], weather=info.tags["weather"],
                             cls=CLASS_NAMES[int(p["labels"][j])], score=float(p["scores"][j]),
                             x=b[j, 0], y=b[j, 1], z=b[j, 2], w=b[j, 3], l=b[j, 4], h=b[j, 5],
                             yaw=float(-b[j, 6] - np.pi / 2), vx=b[j, 7], vy=b[j, 8],
                             dist=float(np.hypot(b[j, 0], b[j, 1])), lidar_pts=int(npts[j]),
                             lidar_pts_e05=int(npts_e[j]), box_idx=j))
    df = pd.DataFrame(rows)
    df["range_bin"] = range_bucket(df.dist)
    return df


# --------------------------------------------------------------------------- matching
def _yaw_diff(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return np.abs(d)


def match(gt: pd.DataFrame, pr: pd.DataFrame, dist_th: float = MATCH_DIST) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy nuScenes-style matching per (sample, class). Adds tp/match columns to copies."""
    gt = gt.copy()
    pr = pr.copy()
    gt["matched"] = False
    gt["match_score"] = np.nan
    gt["pred_row"] = -1
    pr["tp"] = False
    pr["gt_row"] = -1
    pr = pr.sort_values("score", ascending=False, kind="stable")
    gt_groups = {k: v for k, v in gt.groupby(["sample_idx", "cls"]).groups.items()}
    taken = set()
    gx, gy = gt.x.to_numpy(), gt.y.to_numpy()
    gidx = {lab: i for i, lab in enumerate(gt.index)}
    tp_col, gr_col = np.zeros(len(pr), bool), np.full(len(pr), -1)
    for k, (ri, r) in enumerate(zip(pr.index, pr.itertuples())):
        cand = gt_groups.get((r.sample_idx, r.cls))
        if cand is None:
            continue
        best, best_d = None, np.inf
        for gi in cand:
            if gi in taken:
                continue
            ii = gidx[gi]
            d = np.hypot(gx[ii] - r.x, gy[ii] - r.y)
            if d < best_d:
                best, best_d = gi, d
        if best is not None and best_d < dist_th:
            taken.add(best)
            tp_col[k], gr_col[k] = True, best
            gt.at[best, "matched"] = True
            gt.at[best, "match_score"] = r.score
            gt.at[best, "pred_row"] = ri
    pr["tp"], pr["gt_row"] = tp_col, gr_col
    # TP error terms
    m = pr[pr.tp]
    g = gt.loc[m.gt_row]
    pr.loc[m.index, "err_trans"] = np.hypot(m.x.to_numpy() - g.x.to_numpy(), m.y.to_numpy() - g.y.to_numpy())
    pr.loc[m.index, "err_radial"] = m.dist.to_numpy() - g.dist.to_numpy()
    pr.loc[m.index, "err_len_rel"] = (m.l.to_numpy() - g.l.to_numpy()) / g.l.to_numpy()
    pr.loc[m.index, "err_yaw"] = _yaw_diff(m.yaw.to_numpy(), g.yaw.to_numpy())
    pr.loc[m.index, "gt_dist"] = g.dist.to_numpy()
    pr.loc[m.index, "gt_len"] = g.l.to_numpy()
    return gt, pr.sort_index()


# --------------------------------------------------------------------------- metrics
def average_precision(scores: np.ndarray, tps: np.ndarray, npos: int, min_recall=0.1, min_precision=0.1) -> float:
    """nuScenes devkit AP (101-point interpolated, min recall/precision clipped)."""
    if npos == 0:
        return np.nan
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[order].astype(float))
    fp = np.cumsum((~tps[order]).astype(float))
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / npos
    rec_interp = np.linspace(0, 1, 101)
    prec = np.interp(rec_interp, rec, prec, right=0)
    prec = prec[round(100 * min_recall) + 1:] - min_precision
    prec[prec < 0] = 0
    return float(np.mean(prec) / (1.0 - min_precision))


def devkit_filter(gt: pd.DataFrame, pr: pd.DataFrame):
    """Apply the official per-class radius filter (used only to cross-check against the devkit)."""
    rg = gt.cls.map(CLASS_RANGE)
    rp = pr.cls.map(CLASS_RANGE)
    return gt[gt.dist < rg], pr[pr.dist < rp]


def ap_table(gt, pr, by: str, classes=None, dist_th=MATCH_DIST) -> pd.DataFrame:
    """AP per slice value of `by` (GT assigned by GT attribute, preds by their own attribute)."""
    rows = []
    gtm, prm = match(gt, pr, dist_th)
    classes = classes or sorted(gt.cls.unique())
    for val in gtm[by].dropna().unique():
        for c in classes:
            g = gtm[(gtm[by] == val) & (gtm.cls == c)]
            p = prm[(prm[by] == val) & (prm.cls == c)] if by in prm else prm[prm.cls == c]
            rows.append(dict(slice=val, cls=c, npos=len(g),
                             ap=average_precision(p.score.to_numpy(), p.tp.to_numpy(), len(g))))
    return pd.DataFrame(rows)


def best_f1_threshold(gtm: pd.DataFrame, prm: pd.DataFrame) -> float:
    s = np.sort(prm.score.unique())[::-1]
    best, best_t = -1, 0.3
    for t in s[::max(1, len(s) // 400)]:
        tp = (prm.tp & (prm.score >= t)).sum()
        fp = (~prm.tp & (prm.score >= t)).sum()
        f1 = 2 * tp / max(2 * tp + fp + (len(gtm) - tp), 1)
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t


def slice_recall(gtm: pd.DataFrame, by, thr: float) -> pd.DataFrame:
    g = gtm.assign(hit=gtm.matched & (gtm.match_score >= thr))
    out = g.groupby(by, observed=True).agg(n=("hit", "size"), recall=("hit", "mean"))
    return out.reset_index()


def slice_precision(prm: pd.DataFrame, by, thr: float) -> pd.DataFrame:
    p = prm[prm.score >= thr]
    out = p.groupby(by, observed=True).agg(n_pred=("tp", "size"), precision=("tp", "mean"))
    return out.reset_index()


# --------------------------------------------------------------------------- failure tags
FN_TAGS = {
    "beyond_bev_grid": "outside the model's ±51.2 m BEV grid — structurally undetectable",
    "heavily_occluded": "≤40 % visible in all cameras",
    "class_confusion": "detected, but as the wrong class",
    "depth_error": "same-class box 2–4 m away (mostly along the viewing ray)",
    "low_confidence": "matched only below the operating threshold",
    "missed": "no detection nearby at all",
}
FP_TAGS = {
    "duplicate": "second box on an already-detected object",
    "class_confusion": "real object, wrong class",
    "depth_error": "real object, but placed 2–4 m off",
    "ghost_no_lidar": "no GT nearby and zero LiDAR returns inside the box",
    "ghost_with_lidar": "no GT nearby, LiDAR returns present (likely unlabeled / static clutter)",
}


def tag_failures(gtm: pd.DataFrame, prm: pd.DataFrame, thr: float):
    """Tag every FN GT (at threshold) and every FP prediction (score >= thr) with one failure type."""
    gtm = gtm.copy()
    prm = prm.copy()
    live = prm[prm.score >= thr]
    by_sample = {k: v for k, v in live.groupby("sample_idx")}
    gt_by_sample = {k: v for k, v in gtm.groupby("sample_idx")}
    fn_tag = pd.Series(index=gtm.index, dtype=object)
    for gi, g in gtm.iterrows():
        if g.matched and g.match_score >= thr:
            continue
        p = by_sample.get(g.sample_idx)
        d = np.hypot(p.x - g.x, p.y - g.y) if p is not None else pd.Series(dtype=float)
        if not g.in_bev_grid:
            t = "beyond_bev_grid"
        elif g.matched:
            t = "low_confidence"
        elif p is not None and ((d < MATCH_DIST) & (p.cls != g.cls)).any():
            t = "class_confusion"
        elif p is not None and ((d < NEAR_DIST) & (p.cls == g.cls)).any():
            t = "depth_error"
        elif g.visibility <= 1:
            t = "heavily_occluded"
        else:
            t = "missed"
        fn_tag[gi] = t
    gtm["fail_tag"] = fn_tag
    fp_tag = pd.Series(index=prm.index, dtype=object)
    for pi, p in live[~live.tp].iterrows():
        g = gt_by_sample.get(p.sample_idx)
        d = np.hypot(g.x - p.x, g.y - p.y) if g is not None else pd.Series(dtype=float)
        same = (g.cls == p.cls) if g is not None else pd.Series(dtype=bool)
        if g is not None and ((d < MATCH_DIST) & same).any():
            t = "duplicate"
        elif g is not None and ((d < MATCH_DIST) & ~same).any():
            t = "class_confusion"
        elif g is not None and ((d < NEAR_DIST) & same).any():
            t = "depth_error"
        elif p.lidar_pts == 0:
            t = "ghost_no_lidar"
        else:
            t = "ghost_with_lidar"
        fp_tag[pi] = t
    prm["fail_tag"] = fp_tag
    return gtm, prm
