"""Thin wrapper around the official nuScenes detection evaluation (devkit)."""
from __future__ import annotations

import json
import os

from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

from .infer import to_nusc_results


def official_eval(nusc, infos, preds, eval_set: str, out_dir: str, tag: str = "baseline") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_{tag}_{eval_set}.json")
    with open(path, "w") as f:
        json.dump(to_nusc_results(infos, preds), f)
    ev = DetectionEval(nusc, config_factory("detection_cvpr_2019"), path, eval_set,
                       os.path.join(out_dir, f"eval_{tag}_{eval_set}"), verbose=False)
    metrics, _ = ev.evaluate()
    d = metrics.serialize()
    summary = {"mAP": d["mean_ap"], "NDS": d["nd_score"], "tp_errors": d["tp_errors"],
               "class_ap": {k: v for k, v in d["mean_dist_aps"].items()},
               "label_aps": d["label_aps"], "eval_set": eval_set, "tag": tag}
    with open(os.path.join(out_dir, f"metrics_{tag}_{eval_set}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
