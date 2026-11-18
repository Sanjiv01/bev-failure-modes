"""Cheap post-hoc fixes evaluated on top of the frozen detector."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_lidar_points
from .diagnose import RANGE_LABELS, count_points_in_boxes


def lidar_sanity_filter(infos, preds, min_points: int = 1, enlarge: float = 0.5) -> list[dict]:
    """Drop camera detections whose (enlarged) box contains fewer than `min_points` LiDAR returns.

    Rationale: a solid object at <~60 m almost always returns at least one LiDAR point; camera
    "ghosts" (reflections, depth-ambiguous duplicates far behind the real object) do not. LiDAR is
    used only as a verifier, never to move or create boxes.
    """
    out = []
    for info, p in zip(infos, preds):
        if len(p["boxes"]) == 0:
            out.append(p)
            continue
        n = count_points_in_boxes(load_lidar_points(info.lidar_path), p["boxes"], enlarge=enlarge)
        keep = n >= min_points
        out.append({**p, "boxes": p["boxes"][keep], "scores": p["scores"][keep], "labels": p["labels"][keep]})
    return out


def temporal_score_fusion(infos, preds, w: float = 0.4, radius: float = 2.0, miss_penalty: float = 0.15,
                          min_prev: float = 0.05) -> list[dict]:
    """Re-score detections with evidence from the previous frame of the same scene.

    Each previous-frame box (score >= `min_prev`) is moved forward by its own predicted velocity
    over the frame gap and transformed into the current LiDAR frame with the ego poses. Every
    current box is greedily matched (same class, BEV centre distance < `radius`, highest score
    first) to at most one carried-over box:

        matched:    s' = (1 - w) * s + w * s_prev'   (s_prev' is the previous *fused* score)
        unmatched:  s' = (1 - miss_penalty) * s      (not the first frame of a scene)

    Consistent objects found below the threshold get pulled up; one-frame ghosts get pushed down.
    Only scores change. Boxes are never moved, added or removed, so it is a pure re-ranking.
    """
    out = []
    prev = None  # (info, boxes, fused_scores, labels)
    for info, p in zip(infos, preds):
        s = p["scores"].astype(np.float64).copy()
        b, lab = p["boxes"], p["labels"]
        if prev is not None and prev[0].scene_token == info.scene_token and len(b):
            pinfo, pb, ps, pl = prev
            keep = ps >= min_prev
            pb, ps, pl = pb[keep], ps[keep], pl[keep]
            dt = (info.timestamp - pinfo.timestamp) / 1e6
            xyz = pb[:, :3].copy()
            xyz[:, :2] += pb[:, 7:9] * dt  # constant-velocity motion in the previous LiDAR frame
            T = np.linalg.inv(info.lidar2global) @ pinfo.lidar2global
            xyz = (np.c_[xyz, np.ones(len(xyz))] @ T.T)[:, :3]
            used = np.zeros(len(xyz), bool)
            new = s * (1 - miss_penalty)
            for i in np.argsort(-s):
                cand = np.where((pl == lab[i]) & ~used)[0]
                if len(cand) == 0:
                    continue
                d = np.hypot(xyz[cand, 0] - b[i, 0], xyz[cand, 1] - b[i, 1])
                j = int(np.argmin(d))
                if d[j] < radius:
                    used[cand[j]] = True
                    new[i] = (1 - w) * s[i] + w * ps[cand[j]]
            s = new
        out.append({**p, "scores": s.astype(np.float32)})
        prev = (info, b, s, lab)
    return out


def filter_pred_table(pr: pd.DataFrame, min_points: int = 1, col: str = "lidar_pts_e05") -> pd.DataFrame:
    """Same filter applied to an already-built prediction table (no LiDAR re-read)."""
    return pr[pr[col] >= min_points]


def fit_range_thresholds(gtm: pd.DataFrame, prm: pd.DataFrame, grid=None) -> dict:
    """Per-range-bucket score threshold maximising F1 inside that bucket."""
    grid = np.round(np.arange(0.05, 0.80, 0.025), 3) if grid is None else grid
    th = {}
    for rb in RANGE_LABELS:
        g = gtm[gtm.range_bin == rb]
        p = prm[prm.range_bin == rb]
        if len(g) == 0:
            th[rb] = float(grid[-1])
            continue
        best, best_t = -1, float(grid[0])
        for t in grid:
            tp = (p.tp & (p.score >= t)).sum()
            fp = (~p.tp & (p.score >= t)).sum()
            # recall counted on GT whose match is above t (handles cross-bucket matches)
            hits = (g.matched & (g.match_score >= t)).sum()
            prec = tp / max(tp + fp, 1)
            rec = hits / len(g)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            if f1 > best:
                best, best_t = f1, float(t)
        th[rb] = best_t
    return th


def apply_range_thresholds(prm: pd.DataFrame, th: dict) -> pd.Series:
    return prm.score >= prm.range_bin.astype(str).map(th).astype(float)


def operating_point(gtm: pd.DataFrame, prm: pd.DataFrame, keep: pd.Series) -> dict:
    """Precision / recall / F1 of the kept prediction set (matching already computed on all preds)."""
    kept = prm[keep]
    tp = int(kept.tp.sum())
    fp = int((~kept.tp).sum())
    kept_rows = set(kept[kept.tp].index)
    hit = gtm.pred_row.isin(kept_rows) & gtm.matched
    rec = hit.sum() / max(len(gtm), 1)
    prec = tp / max(tp + fp, 1)
    return {"tp": tp, "fp": fp, "fn": int(len(gtm) - hit.sum()), "precision": float(prec),
            "recall": float(rec), "f1": float(2 * prec * rec / max(prec + rec, 1e-12))}
