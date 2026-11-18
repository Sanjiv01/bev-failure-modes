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
