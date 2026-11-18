"""One-command pipeline:  python -m bevdiag run  (see README for flags)."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys


def _load(args):
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import mini_train, mini_val

    from .data import build_infos

    nusc = NuScenes(args.version, args.dataroot, verbose=False)
    can = NuScenesCanBus(dataroot=args.dataroot) if os.path.isdir(os.path.join(args.dataroot, "can_bus")) else None
    if can is None:
        print("[warn] CAN bus expansion not found; ego-motion vector falls back to pose-only (degrades accuracy)")
    infos = build_infos(nusc, mini_val + mini_train, can)
    return nusc, infos, set(mini_val)


def _predictions(args, infos, variant, budget):
    from .model import CONFIGS

    out = os.path.join(args.out, variant)
    raw = os.path.join(out, "raw_preds.pkl")
    if args.force or not os.path.exists(raw):
        from .infer import run_inference

        ckpt = os.path.join(args.ckpt_dir, CONFIGS[variant].ckpt)
        if not os.path.exists(ckpt):
            print(f"[skip] {variant}: checkpoint {ckpt} not found")
            return None
        print(f"      inference: {CONFIGS[variant].name} on {len(infos)} keyframes")
        run_inference(infos, ckpt, out, workers=budget.loader_workers, variant=variant)
    with open(raw, "rb") as f:
        return pickle.load(f)


def cmd_run(args):
    from nuscenes.utils.splits import mini_train

    from .resources import apply_budget

    budget = apply_budget(args.budget)
    os.makedirs(args.out, exist_ok=True)
    nusc, infos, val_scenes = _load(args)

    print("[1/6] inference")
    preds_by = {v: p for v in args.models for p in [_predictions(args, infos, v, budget)] if p is not None}
    if "tiny" not in preds_by:
        raise SystemExit("the tiny baseline is required")
    preds = preds_by["tiny"]
    with open(os.path.join(args.out, args.models[0], "timing.json")) as f:
        timing = json.load(f)
    with open(os.path.join(args.out, "timing.json"), "w") as f:
        json.dump(timing, f)
    timings = {}
    for v in preds_by:
        with open(os.path.join(args.out, v, "timing.json")) as f:
            timings[v] = json.load(f)

    from .official import official_eval

    print("[2/6] official nuScenes eval of the baseline (held-out mini_val)")
    vi = [i for i, inf in enumerate(infos) if inf.scene_name in val_scenes]
    base = official_eval(nusc, [infos[i] for i in vi], [preds[i] for i in vi], "mini_val", args.out, "baseline")
    print(f"      mAP={base['mAP']:.4f} NDS={base['NDS']:.4f}")

    from .pipeline import analyze, build_tables, diagnose_and_fix

    print("[3/6] baseline slicing, failure tagging, LiDAR-filter sweep")
    summary = diagnose_and_fix(nusc, infos, preds, val_scenes, args.out, base)

    from .improve import improvement_study

    print(f"[4/6] improvement study over {list(preds_by)} (temporal fusion tuned on mini_train only)")
    imp, final_preds = improvement_study(nusc, infos, preds_by, val_scenes, set(mini_train),
                                         os.path.join(args.out, "improve"))
    summary["improvement"] = imp
    summary["timings"] = timings
    fin = imp["final"]
    print(f"      final system: {fin['variant']} + {fin['fix']}")

    print("[5/6] slicing the improved system")
    gt = __import__("pandas").read_pickle(os.path.join(args.out, "gt_all_mini.pkl"))
    gt = gt[[c for c in gt.columns if c not in ("matched", "match_score", "pred_row", "fail_tag")]]
    _, pr = build_tables(nusc, infos, final_preds, val_scenes, gt=gt.copy())
    res, _ = analyze(gt.copy(), pr, args.out, prefix="improved_")
    row = next(r for r in imp["ladder"] if r["variant"] == fin["variant"] and r["fix"] == fin["fix"])
    summary["improved"] = {**res, "official": {"mAP": row["mAP"], "NDS": row["NDS"], "class_ap": row["class_ap"]},
                           "variant": fin["variant"], "fix": fin["fix"]}
    with open(os.path.join(args.out, "improved_preds.pkl"), "wb") as f:
        pickle.dump(final_preds, f)

    from .pipeline import improved_headline

    improved_headline(summary)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    from .report import build_report

    print("[6/6] figures + report")
    build_report(nusc, infos, preds, summary, args.out, args.docs, improved_preds=final_preds)
    print(json.dumps(summary["headline"], indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bevdiag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="inference -> official eval -> slicing -> fixes -> improvement study -> report")
    r.add_argument("--dataroot", default="data/nuscenes")
    r.add_argument("--version", default="v1.0-mini")
    r.add_argument("--ckpt-dir", default="checkpoints")
    r.add_argument("--models", nargs="+", default=["tiny", "small", "base"], choices=["tiny", "small", "base"])
    r.add_argument("--out", default="outputs")
    r.add_argument("--docs", default="docs")
    r.add_argument("--budget", type=float, default=0.9, help="fraction of CPU/GPU to use (default 0.9)")
    r.add_argument("--force", action="store_true", help="re-run inference even if cached")
    r.set_defaults(func=cmd_run)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
