"""Before/after figures: baseline (BEVFormer-tiny as released) vs the improved system."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import viz
from .diagnose import RANGE_LABELS
from .viz import FN_LABEL, FP_LABEL, GROUP_COLOR, GROUP_LABEL, _range_x, _shade_beyond_grid

FIX_LABEL = {"none": "as released", "temporal": "+ temporal fusion", "lidar": "+ LiDAR veto",
             "temporal+lidar": "+ both fixes"}
FIX_COLOR = {"none": "neutral", "temporal": "s1", "lidar": "s3", "temporal+lidar": "s2"}
VARIANT_NAME = {"tiny": "BEVFormer-tiny", "small": "BEVFormer-small", "base": "BEVFormer-base"}


def system_name(summary):
    fin = summary["improved"]
    fix = {"none": "", "temporal": " + temporal fusion", "lidar": " + LiDAR veto",
           "temporal+lidar": " + temporal fusion + LiDAR veto"}[fin["fix"]]
    return VARIANT_NAME[fin["variant"]] + fix


def fig_improve_hero(summary):
    T = viz.T
    h = summary["headline"]
    b = summary["sets"]["heldout"]["range_profile"]
    a = summary["improved"]["sets"]["heldout"]["range_profile"]
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.35, 0.05, 1])
    ax = fig.add_subplot(gs[0, 0])
    x = _range_x()
    for g, col in (("all", T["s1"]), ("large_vehicle", T["s2"])):
        yb = np.array([r["recall"] if r["recall"] is not None else np.nan for r in b if r["group"] == g], float)
        ya = np.array([r["recall"] if r["recall"] is not None else np.nan for r in a if r["group"] == g], float)
        ax.plot(x, yb, color=col, ls=":", lw=1.8, marker="o", ms=4, alpha=0.8)
        ax.plot(x, ya, color=col, marker="o", ms=7, mec=T["surface"], mew=2,
                label=f"{GROUP_LABEL[g]}: improved (solid) vs baseline (dotted)")
        ax.fill_between(x[:6], yb[:6], ya[:6], color=col, alpha=0.13, lw=0)
    ax.set_xticks(x, [f"{r} m" for r in RANGE_LABELS])
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Recall at operating threshold")
    ax.set_xlabel("Distance from ego vehicle")
    _shade_beyond_grid(ax)
    ax.set_title("Same camera-only task, made better: more objects found at every distance inside the grid",
                 fontsize=13.5)
    ax.legend(loc="lower left", fontsize=9.5)
    side = fig.add_subplot(gs[0, 2])
    side.axis("off")
    tiles = [
        (f"{h['baseline_NDS_mini_val']:.3f} → {h['imp_NDS']:.3f}", "official NDS (held-out mini_val)"),
        (f"{h['baseline_mAP_mini_val']:.3f} → {h['imp_mAP']:.3f}", "official mAP (held-out mini_val)"),
        (f"{h['base_recall_30_50']:.0%} → {h['imp_recall_30_50']:.0%}", "recall at 30–50 m, all classes"),
        (f"{h['base_large_recall_30_50']:.0%} → {h['imp_large_recall_30_50']:.0%}", "truck / bus recall at 30–50 m"),
    ]
    for i, (big, small) in enumerate(tiles):
        y0 = 0.95 - i * 0.245
        side.text(0.0, y0, big, fontsize=23, fontweight="bold", color=T["text"], va="top", transform=side.transAxes)
        side.text(0.0, y0 - 0.115, small, fontsize=10.5, color=T["text2"], va="top", transform=side.transAxes)
    fig.text(0.01, -0.02, f"Baseline: BEVFormer-tiny as released. Improved: {system_name(summary)}. "
             "Fix settings chosen on training scenes only; scored on held-out nuScenes mini_val.",
             color=T["muted"], fontsize=9)
    return fig


def fig_ladder(summary):
    T = viz.T
    imp = summary["improvement"]
    rows = imp["ladder"]
    variants = [v for v in ("tiny", "small", "base") if any(r["variant"] == v for r in rows)]
    fin = imp["final"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    names = {"tiny": "tiny\nResNet-50 · 50×50 BEV", "small": "small\nResNet-101-DCN · 150×150",
             "base": "base\nResNet-101-DCN · 200×200"}
    for ax, key, title in ((axes[0], "NDS", "Official NDS, held-out mini_val"),
                           (axes[1], "mAP", "Official mAP, held-out mini_val")):
        width = 0.2
        for k, fix in enumerate(FIX_LABEL):
            vals = [next(r[key] for r in rows if r["variant"] == v and r["fix"] == fix) for v in variants]
            xs = np.arange(len(variants)) + (k - 1.5) * width
            ax.bar(xs, vals, width=width * 0.92, color=T[FIX_COLOR[fix]], label=FIX_LABEL[fix],
                   edgecolor=T["surface"], linewidth=1.5)
            for xb, v, var in zip(xs, vals, variants):
                star = var == fin["variant"] and fix == fin["fix"]
                ax.text(xb, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9 if star else 7.5,
                        color=T["text"] if star else T["text2"], fontweight="bold" if star else "normal",
                        rotation=0 if star else 90)
        base = next(r[key] for r in rows if r["variant"] == "tiny" and r["fix"] == "none")
        ax.axhline(base, color=T["text2"], ls="--", lw=1)
        ax.text(len(variants) - 0.5, base + 0.004, "baseline (tiny as released)", ha="right", va="bottom", fontsize=9, color=T["text2"])
        ax.set_xticks(range(len(variants)), [names[v] for v in variants], fontsize=9.5)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(r[key] for r in rows) * 1.25)
    axes[0].legend(fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def fig_before_after_range(summary):
    T = viz.T
    b = summary["sets"]["heldout"]["range_profile"]
    a = summary["improved"]["sets"]["heldout"]["range_profile"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    x = _range_x()[:6]
    for ax, g in zip(axes, ("car", "large_vehicle", "vru")):
        yb = [r["recall"] or 0 for r in b if r["group"] == g][:6]
        ya = [r["recall"] or 0 for r in a if r["group"] == g][:6]
        n = [r["n_gt"] for r in b if r["group"] == g][:6]
        ax.bar(x - 0.2, yb, width=0.38, color=T["neutral"], edgecolor=T["surface"], linewidth=1.5, label="baseline")
        ax.bar(x + 0.2, ya, width=0.38, color=T[GROUP_COLOR[g]], edgecolor=T["surface"], linewidth=1.5,
               label="improved")
        for i in range(6):
            if n[i] >= 5:
                d = ya[i] - yb[i]
                ax.text(x[i] + 0.2, ya[i] + 0.02, f"{d:+.0%}", ha="center", fontsize=8.5,
                        color=T["text"] if d > 0 else T["text2"])
            ax.text(x[i], -0.075, f"n={n[i]}", ha="center", fontsize=7.5, color=T["muted"])
        ax.set_xticks(x, RANGE_LABELS[:6], fontsize=9)
        ax.set_title(f"{GROUP_LABEL[g]}: recall", fontsize=12)
        ax.set_xlabel("metres from ego", labelpad=14)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_ylim(-0.1, 1.12)
    axes[0].legend(fontsize=9.5, loc="upper right")
    fig.tight_layout()
    return fig


def fig_class_ap_compare(summary):
    T = viz.T
    imp = summary["improvement"]
    base = next(r for r in imp["ladder"] if r["variant"] == "tiny" and r["fix"] == "none")["class_ap"]
    after = summary["improved"]["official"]["class_ap"]
    order = ["car", "truck", "bus", "pedestrian", "motorcycle", "bicycle", "traffic_cone"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    y = np.arange(len(order))
    for i, c in enumerate(order):
        ax.plot([base[c], after[c]], [i, i], color=T["grid"], lw=5, solid_capstyle="round", zorder=1)
    ax.scatter([base[c] for c in order], y, s=70, color=T["neutral"], zorder=2, label="baseline (tiny)",
               edgecolor=T["surface"], linewidth=1.5)
    ax.scatter([after[c] for c in order], y, s=95, color=T["s1"], zorder=3, label="improved",
               edgecolor=T["surface"], linewidth=1.5)
    for i, c in enumerate(order):
        ax.text(max(base[c], after[c]) + 0.02, i, f"{base[c]:.2f} → {after[c]:.2f}", va="center", fontsize=9.5,
                color=T["text2"])
    ax.set_yticks(y, [c.replace("_", " ") for c in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("official AP (mean over 0.5 / 1 / 2 / 4 m match thresholds), held-out mini_val")
    ax.set_title("Per-class AP, baseline vs improved (classes present in mini_val)", fontsize=12)
    ax.legend(fontsize=9.5, loc="lower right")
    fig.tight_layout()
    return fig


def fig_errors_compare(summary):
    T = viz.T
    fin = summary["improved"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.0))
    for ax, key, lab, title in ((axes[0], "fn_tags", FN_LABEL, "Misses by cause (held-out)"),
                                (axes[1], "fp_tags", FP_LABEL, "False positives by cause (held-out)")):
        tags = list(lab)
        b = [sum(r["n"] for r in summary["sets"]["heldout"][key] if r["fail_tag"] == t) for t in tags]
        a = [sum(r["n"] for r in fin["sets"]["heldout"][key] if r["fail_tag"] == t) for t in tags]
        y = np.arange(len(tags))
        ax.barh(y - 0.2, b, height=0.38, color=T["neutral"], label="baseline", edgecolor=T["surface"])
        ax.barh(y + 0.2, a, height=0.38, color=T["s1"], label="improved", edgecolor=T["surface"])
        for i in range(len(tags)):
            if b[i] or a[i]:
                ax.text(max(a[i], b[i]) + max(b + a) * 0.01, i, f"{b[i]} → {a[i]}", va="center", fontsize=9,
                        color=T["text2"])
        ax.set_yticks(y, [lab[t] for t in tags], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, max(b + a) * 1.25)
    axes[0].legend(fontsize=9.5, loc="lower right")
    fig.tight_layout()
    return fig
