"""Improvement study: bigger pretrained backbones x post-hoc fixes, measured with the official devkit.

Protocol (keeps the held-out scenes clean):
  * temporal-fusion parameters are grid-searched on the 8 `mini_train` scenes only;
  * the final configuration per model is chosen by `mini_train` NDS;
  * every rung of the ladder is then scored once on held-out `mini_val`.
"""
from __future__ import annotations

import itertools
import json
import os

from .fixes import lidar_sanity_filter, temporal_score_fusion
from .model import CONFIGS
from .official import official_eval

TEMPORAL_GRID = {"w": (0.2, 0.4, 0.6), "miss_penalty": (0.0, 0.15, 0.3)}
FIXES = ("none", "temporal", "lidar", "temporal+lidar")


def _split(infos, preds, scenes):
    idx = [i for i, inf in enumerate(infos) if inf.scene_name in scenes]
    return [infos[i] for i in idx], [preds[i] for i in idx]


def apply_fix(infos, preds, fix: str, tparams: dict | None) -> list[dict]:
    out = preds
    if "temporal" in fix:
        out = temporal_score_fusion(infos, out, **tparams)
    if "lidar" in fix:
        out = lidar_sanity_filter(infos, out, min_points=1, enlarge=0.5)
    return out


def tune_temporal(nusc, infos, preds, train_scenes, out_dir, tag) -> tuple[dict, list]:
    ti, tp = _split(infos, preds, train_scenes)
    rows = []
    for w, mp in itertools.product(TEMPORAL_GRID["w"], TEMPORAL_GRID["miss_penalty"]):
        params = {"w": w, "miss_penalty": mp}
        m = official_eval(nusc, ti, temporal_score_fusion(ti, tp, **params), "mini_train", out_dir,
                          f"{tag}_tune_w{w}_p{mp}")
        rows.append({**params, "mAP": m["mAP"], "NDS": m["NDS"]})
    best = max(rows, key=lambda r: r["NDS"])
    return {"w": best["w"], "miss_penalty": best["miss_penalty"]}, rows


def improvement_study(nusc, infos, preds_by_variant: dict, val_scenes, train_scenes, out_dir) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ladder, tuning, choice, final_preds = [], {}, {}, {}
    for variant, preds in preds_by_variant.items():
        tparams, grid = tune_temporal(nusc, infos, preds, train_scenes, out_dir, variant)
        tuning[variant] = {"best": tparams, "grid": grid}
        train_nds = {}
        for fix in FIXES:
            fixed = apply_fix(infos, preds, fix, tparams)
            ti, tp = _split(infos, fixed, train_scenes)
            train_nds[fix] = official_eval(nusc, ti, tp, "mini_train", out_dir, f"{variant}_{fix}_train")["NDS"]
            vinf, vp = _split(infos, fixed, val_scenes)
            m = official_eval(nusc, vinf, vp, "mini_val", out_dir, f"{variant}_{fix}".replace("+", "_"))
            ladder.append({"variant": variant, "model": CONFIGS[variant].name, "fix": fix,
                           "mAP": m["mAP"], "NDS": m["NDS"], "class_ap": m["class_ap"], "tp_errors": m["tp_errors"],
                           "train_NDS": train_nds[fix],
                           "published_NDS": CONFIGS[variant].published_nds,
                           "published_mAP": CONFIGS[variant].published_map})
            final_preds[(variant, fix)] = fixed
        choice[variant] = max(train_nds, key=train_nds.get)
    best_variant = max(preds_by_variant, key=lambda v: next(
        r["train_NDS"] for r in ladder if r["variant"] == v and r["fix"] == choice[v]))
    result = {"ladder": ladder, "temporal_tuning": tuning, "choice_by_train_nds": choice,
              "final": {"variant": best_variant, "fix": choice[best_variant]}}
    with open(os.path.join(out_dir, "improvement.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    return result, final_preds[(best_variant, choice[best_variant])]
