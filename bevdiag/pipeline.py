"""Glue: build tables, slice metrics, tag failures, evaluate fixes -> summary.json."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .diagnose import (LARGE_VEHICLES, LENGTH_LABELS, RANGE_LABELS, average_precision, best_f1_threshold,
                       gt_table, match, pred_table, slice_precision, slice_recall, tag_failures)
from .fixes import apply_range_thresholds, fit_range_thresholds, operating_point
from .official import official_eval

GROUPS = {"all": None, "car": ("car",), "large_vehicle": LARGE_VEHICLES,
          "vru": ("pedestrian", "bicycle", "motorcycle")}
VEHICLES = ("car",) + LARGE_VEHICLES


def _grp(df, group):
    cls = GROUPS[group]
    return df if cls is None else df[df.cls.isin(cls)]


def _records(df):
    return json.loads(df.to_json(orient="records"))


def pooled_ap(gtm, prm, group, rb=None):
    g, p = _grp(gtm, group), _grp(prm, group)
    if rb is not None:
        g, p = g[g.range_bin == rb], p[p.range_bin == rb]
    return average_precision(p.score.to_numpy(), p.tp.to_numpy(), len(g))


def range_profile(gtm, prm, thr):
    rows = []
    for group in GROUPS:
        g, p = _grp(gtm, group), _grp(prm, group)
        rec = slice_recall(g, "range_bin", thr).set_index("range_bin")
        prec = slice_precision(p, "range_bin", thr).set_index("range_bin")
        for rb in RANGE_LABELS:
            n = int(rec.n.get(rb, 0)) if rb in rec.index else 0
            rows.append(dict(group=group, range=rb, n_gt=n,
                             recall=float(rec.recall[rb]) if n else None,
                             n_pred=int(prec.n_pred[rb]) if rb in prec.index else 0,
                             precision=float(prec.precision[rb]) if rb in prec.index else None,
                             ap=None if n < 5 else pooled_ap(gtm, prm, group, rb)))
    return rows


def tp_error_profile(prm):
    tp = prm[prm.tp].copy()
    tp["gt_range_bin"] = pd.cut(tp.gt_dist, [0, 10, 20, 30, 40, 50, 60, np.inf], right=False,
                                labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+"])
    rows = []
    for group in ("car", "large_vehicle", "vru"):
        t = _grp(tp, group)
        for rb, s in t.groupby("gt_range_bin", observed=True):
            if len(s) < 3:
                continue
            rows.append(dict(group=group, range=str(rb), n=len(s),
                             trans_med=float(s.err_trans.median()), trans_p90=float(s.err_trans.quantile(.9)),
                             radial_bias=float(s.err_radial.median()),
                             len_err_med=float(s.err_len_rel.median()), yaw_med_deg=float(np.degrees(s.err_yaw.median()))))
    return rows


def worst_cases(gtm, prm, thr, k=10):
    """Rank frames by truck-relevant damage: missed large vehicles (x3), other misses, confident FPs."""
    fn = gtm[gtm.fail_tag.notna()]
    fp = prm[prm.fail_tag.notna()]
    big = fn[fn.cls.isin(LARGE_VEHICLES)].groupby("sample_idx").size()
    fn_n = fn[fn.in_bev_grid].groupby("sample_idx").size()
    fp_n = fp[fp.score >= max(thr, 0.4)].groupby("sample_idx").size()
    score = big.mul(3).add(fn_n, fill_value=0).add(fp_n * 2, fill_value=0).sort_values(ascending=False)
    picked, scenes_seen = [], {}
    for si in score.index:  # spread across scenes and avoid near-identical consecutive frames
        sc = gtm.loc[gtm.sample_idx == si, "scene"].iloc[0] if (gtm.sample_idx == si).any() else ""
        if any(abs(si - q) < 6 for q in picked) or scenes_seen.get(sc, 0) >= 2:
            continue
        picked.append(int(si))
        scenes_seen[sc] = scenes_seen.get(sc, 0) + 1
        if len(picked) >= k:
            break
    return picked


def _filter_preds(preds, prm, keep_mask):
    keep = prm[keep_mask]
    out = []
    for si, p in enumerate(preds):
        idx = np.sort(keep.loc[keep.sample_idx == si, "box_idx"].to_numpy().astype(int))
        out.append({**p, "boxes": p["boxes"][idx], "scores": p["scores"][idx], "labels": p["labels"][idx]})
    return out


def diagnose_and_fix(nusc, infos, preds, val_scenes, out_dir, base_metrics) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    gt = gt_table(nusc, infos)
    pr = pred_table(infos, preds)
    for df in (gt, pr):
        df["split"] = np.where(df.scene.isin(val_scenes), "heldout", "seen_in_training")
    sets = {"heldout": (gt[gt.split == "heldout"], pr[pr.split == "heldout"]), "all_mini": (gt, pr)}
    matched = {k: match(*v) for k, v in sets.items()}
    thr = best_f1_threshold(*matched["heldout"])
    summary = {"threshold": thr, "baseline_official": base_metrics, "sets": {}}
    for name, (gtm, prm) in matched.items():
        gtm, prm = tag_failures(gtm, prm, thr)
        matched[name] = (gtm, prm)
        veh = gtm[gtm.cls.isin(VEHICLES)]
        s = {
            "n_frames": int(gtm.sample_idx.nunique()), "n_gt": int(len(gtm)),
            "range_profile": range_profile(gtm, prm, thr),
            "class_range_recall": _records(
                gtm.assign(hit=gtm.matched & (gtm.match_score >= thr))
                .groupby(["cls", "range_bin"], observed=True).hit.agg(["size", "mean"]).reset_index()),
            "size_recall": _records(slice_recall(veh, "length_bin", thr)),
            "size_recall_by_range": _records(
                veh.assign(hit=veh.matched & (veh.match_score >= thr), far=np.where(veh.dist >= 40, ">=40 m", "<40 m"))
                .groupby(["length_bin", "far"], observed=True).hit.agg(["size", "mean"]).reset_index()),
            "visibility_recall": _records(slice_recall(gtm[gtm.in_bev_grid], "visibility", thr)),
            "lighting_range_recall": _records(
                gtm.assign(hit=gtm.matched & (gtm.match_score >= thr))
                .groupby(["lighting", "range_bin"], observed=True).hit.agg(["size", "mean"]).reset_index()),
            "tp_errors": tp_error_profile(prm),
            "fn_tags": _records(gtm[gtm.fail_tag.notna()].groupby(["range_bin", "fail_tag"], observed=True)
                                .size().rename("n").reset_index()),
            "fn_tags_large": _records(gtm[gtm.fail_tag.notna() & gtm.cls.isin(LARGE_VEHICLES)]
                                      .groupby("fail_tag").size().rename("n").reset_index()),
            "fp_tags": _records(prm[prm.fail_tag.notna()].groupby(["range_bin", "fail_tag"], observed=True)
                                .size().rename("n").reset_index()),
            "worst_cases": worst_cases(gtm, prm, thr),
            "operating_point": operating_point(gtm, prm, prm.score >= thr),
        }
        summary["sets"][name] = s
        gtm.to_pickle(os.path.join(out_dir, f"gt_{name}.pkl"))
        prm.to_pickle(os.path.join(out_dir, f"pred_{name}.pkl"))

    # ------------------------------------------------------------------ fix A: LiDAR sanity filter
    vi = [i for i, inf in enumerate(infos) if inf.scene_name in val_scenes]
    vinfos = [infos[i] for i in vi]
    fixA = {"rule": "drop box if < 1 LiDAR return inside box grown by 0.5 m per side", "sweep": []}
    for col, enl in (("lidar_pts", 0.0), ("lidar_pts_e05", 0.5)):
        for k in (1, 3, 5):
            keep = pr[col] >= k
            fp_preds = _filter_preds(preds, pr, keep)
            m = official_eval(nusc, vinfos, [fp_preds[i] for i in vi], "mini_val", out_dir, f"lidarfilter_k{k}_e{enl}")
            row = {"min_points": k, "enlarge_m": enl, "mAP": m["mAP"], "NDS": m["NDS"], "class_ap": m["class_ap"]}
            for name, (gtm, prm) in matched.items():
                kk = prm[col] >= k
                row[f"op_{name}"] = operating_point(gtm, prm, (prm.score >= thr) & kk)
                row[f"range_{name}"] = range_profile(gtm, prm[kk], thr)
                row[f"removed_tp_{name}"] = int((prm.tp & ~kk).sum())
                row[f"removed_fp_{name}"] = int((~prm.tp & ~kk & (prm.score >= thr)).sum())
            fixA["sweep"].append(row)
    fixA["chosen"] = next(r for r in fixA["sweep"] if r["min_points"] == 1 and r["enlarge_m"] == 0.5)
    summary["fix_lidar_filter"] = fixA

    # ------------------------------------------------------------------ fix B: range-aware thresholds (2-fold by scene)
    gtm, prm = matched["heldout"]
    scenes = sorted(gtm.scene.unique())
    tot = {"global": [], "range_aware": []}
    folds = []
    for test in scenes:
        tr = [s for s in scenes if s != test]
        gtr, ptr = gtm[gtm.scene.isin(tr)], prm[prm.scene.isin(tr)]
        gte, pte = gtm[gtm.scene == test], prm[prm.scene == test]
        t_glob = best_f1_threshold(gtr, ptr)
        th = fit_range_thresholds(gtr, ptr)
        a = operating_point(gte, pte, pte.score >= t_glob)
        b = operating_point(gte, pte, apply_range_thresholds(pte, th))
        folds.append({"test_scene": test, "global_thr": t_glob, "range_thr": th, "global": a, "range_aware": b})
        tot["global"].append(a)
        tot["range_aware"].append(b)

    def agg(lst):
        tp, fp, fn = (sum(x[k] for x in lst) for k in ("tp", "fp", "fn"))
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}

    summary["fix_range_threshold"] = {"folds": folds, "global": agg(tot["global"]), "range_aware": agg(tot["range_aware"])}

    # ------------------------------------------------------------------ headline numbers
    hp = {r["range"]: r for r in summary["sets"]["heldout"]["range_profile"] if r["group"] == "all"}
    ap_ = {r["range"]: r for r in summary["sets"]["all_mini"]["range_profile"] if r["group"] == "large_vehicle"}
    near = [hp[k]["recall"] for k in ("0-10", "10-20", "20-30") if hp[k]["recall"] is not None]
    far = [hp[k] for k in ("50-60", "60-80", "80+")]
    far_rec = sum((r["recall"] or 0) * r["n_gt"] for r in far) / max(sum(r["n_gt"] for r in far), 1)
    ch = fixA["chosen"]
    summary["headline"] = {
        "baseline_mAP_mini_val": base_metrics["mAP"], "baseline_NDS_mini_val": base_metrics["NDS"],
        "threshold": thr,
        "recall_0_30m_heldout": float(np.mean(near)),
        "recall_40_50m_heldout": hp["40-50"]["recall"],
        "recall_beyond_50m_heldout": float(far_rec),
        "large_vehicle_recall_beyond_50m_all": float(
            sum((ap_[k]["recall"] or 0) * ap_[k]["n_gt"] for k in ("50-60", "60-80", "80+")) /
            max(sum(ap_[k]["n_gt"] for k in ("50-60", "60-80", "80+")), 1)),
        "fixA_mAP": ch["mAP"], "fixA_NDS": ch["NDS"],
        "fixA_precision_heldout": [summary["sets"]["heldout"]["operating_point"]["precision"],
                                   ch["op_heldout"]["precision"]],
        "fixA_recall_heldout": [summary["sets"]["heldout"]["operating_point"]["recall"], ch["op_heldout"]["recall"]],
        "fixB_f1": [summary["fix_range_threshold"]["global"]["f1"], summary["fix_range_threshold"]["range_aware"]["f1"]],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
