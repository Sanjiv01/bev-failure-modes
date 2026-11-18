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


def cmd_run(args):
    from .resources import apply_budget

    budget = apply_budget(args.budget)
    os.makedirs(args.out, exist_ok=True)
    nusc, infos, val_scenes = _load(args)
    raw = os.path.join(args.out, "raw_preds.pkl")
    if args.force or not os.path.exists(raw):
        from .infer import run_inference

        print(f"[1/5] inference on {len(infos)} keyframes")
        run_inference(infos, args.ckpt, args.out, workers=budget.loader_workers)
    with open(raw, "rb") as f:
        preds = pickle.load(f)

    from .official import official_eval

    print("[2/5] official nuScenes eval (held-out mini_val)")
    vi = [i for i, inf in enumerate(infos) if inf.scene_name in val_scenes]
    base = official_eval(nusc, [infos[i] for i in vi], [preds[i] for i in vi], "mini_val", args.out, "baseline")
    print(f"      mAP={base['mAP']:.4f} NDS={base['NDS']:.4f}")

    from .pipeline import diagnose_and_fix

    print("[3/5] slicing + failure tagging, [4/5] fix evaluation")
    summary = diagnose_and_fix(nusc, infos, preds, val_scenes, args.out, base)

    from .report import build_report

    print("[5/5] figures + report")
    build_report(nusc, infos, preds, summary, args.out, args.docs)
    print(json.dumps(summary["headline"], indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bevdiag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="inference -> official eval -> slicing -> fix -> report")
    r.add_argument("--dataroot", default="data/nuscenes")
    r.add_argument("--version", default="v1.0-mini")
    r.add_argument("--ckpt", default="checkpoints/bevformer_tiny_epoch_24.pth")
    r.add_argument("--out", default="outputs")
    r.add_argument("--docs", default="docs")
    r.add_argument("--budget", type=float, default=0.9, help="fraction of CPU/GPU to use (default 0.9)")
    r.add_argument("--force", action="store_true", help="re-run inference even if cached")
    r.set_defaults(func=cmd_run)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
