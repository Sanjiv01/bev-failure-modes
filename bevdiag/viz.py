"""Figures for the failure report. Every chart renders in a light and a dark variant."""
from __future__ import annotations

import os
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from .diagnose import BEV_HALF_EXTENT, RANGE_LABELS  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984", grid="#e4e3de",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100", s8="#e34948", s7="#6250d6",
                  neutral="#c9c8c2", seq=["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                          "#184f95", "#0d366b"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#8f8e88", grid="#33332f",
                 s1="#3987e5", s2="#d95926", s3="#199e70", s4="#c98500", s8="#e66767", s7="#9085e9",
                 neutral="#4a4a46", seq=["#262624", "#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5",
                                         "#6da7ec", "#b7d3f6"]),
}
GROUP_COLOR = {"car": "s1", "large_vehicle": "s2", "vru": "s3", "all": "text2"}
GROUP_LABEL = {"car": "Cars", "large_vehicle": "Trucks · buses · trailers", "vru": "Pedestrians & cyclists",
               "all": "All classes"}
T = THEMES["light"]


@contextmanager
def theme(name):
    global T
    prev = T
    T = THEMES[name]
    rc = {"figure.facecolor": T["surface"], "axes.facecolor": T["surface"], "savefig.facecolor": T["surface"],
          "text.color": T["text"], "axes.labelcolor": T["text2"], "xtick.color": T["text2"],
          "ytick.color": T["text2"], "axes.edgecolor": T["grid"], "axes.grid": True, "grid.color": T["grid"],
          "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
          "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
          "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False,
          "axes.axisbelow": True, "lines.linewidth": 2.2, "lines.solid_capstyle": "round"}
    with plt.rc_context(rc):
        try:
            yield T
        finally:
            T = prev


def save_both(fn, path_stem, **kw):
    """Render `fn(**kw)` (which returns a figure) in both themes -> <stem>_light.png / <stem>_dark.png."""
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    for name in ("light", "dark"):
        with theme(name):
            fig = fn(**kw)
            fig.savefig(f"{path_stem}_{name}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)


def _range_x():
    return np.arange(len(RANGE_LABELS))


def _shade_beyond_grid(ax, ymax=1.0, label=True):
    # 51.2 m falls inside the "50-60" bin (index 5); shade from there on
    ax.axvspan(4.62, len(RANGE_LABELS) - 0.5, color=T["neutral"], alpha=0.35, lw=0, zorder=0)
    if label:
        ax.text(5.0, ymax * 0.96, "outside the model's\n±51.2 m BEV grid", color=T["text2"], fontsize=9.5,
                va="top", ha="left")


# --------------------------------------------------------------------------- hero
def fig_hero(summary):
    rp = summary["sets"]["heldout"]["range_profile"]
    h = summary["headline"]
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.35, 0.05, 1])
    ax = fig.add_subplot(gs[0, 0])
    x = _range_x()
    for g in ("car", "large_vehicle", "vru"):
        y = [r["recall"] if r["recall"] is not None else np.nan for r in rp if r["group"] == g]
        ax.plot(x, y, color=T[GROUP_COLOR[g]], marker="o", ms=7, mec=T["surface"], mew=2, label=GROUP_LABEL[g])
        last = max(i for i, v in enumerate(y) if not np.isnan(v))
    ax.set_xticks(x, [f"{r} m" for r in RANGE_LABELS])
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Recall at operating threshold")
    ax.set_xlabel("Distance from ego vehicle")
    _shade_beyond_grid(ax)
    ax.set_title("Camera-only BEV detection collapses past 40 m — exactly where highway trucks need to see",
                 fontsize=13.5)
    ax.legend(loc="upper right", bbox_to_anchor=(0.62, 1.0))
    side = fig.add_subplot(gs[0, 2])
    side.axis("off")
    tiles = [
        (f"{h['recall_0_30m_heldout']:.0%}", "recall within 30 m"),
        (f"{h['recall_40_50m_heldout']:.0%}", "recall at 40–50 m"),
        (f"{h['recall_beyond_50m_heldout']:.1%}", "recall beyond 50 m"),
        (f"{h['fixA_precision_heldout'][0]:.0%} → {h['fixA_precision_heldout'][1]:.0%}",
         "precision with LiDAR sanity filter"),
    ]
    for i, (big, small) in enumerate(tiles):
        y0 = 0.95 - i * 0.245
        side.text(0.0, y0, big, fontsize=26, fontweight="bold", color=T["text"], va="top",
                  transform=side.transAxes)
        side.text(0.0, y0 - 0.12, small, fontsize=10.5, color=T["text2"], va="top", transform=side.transAxes)
    fig.text(0.01, -0.02, "BEVFormer-tiny (official weights) on held-out nuScenes mini_val, 81 frames, "
             f"3,793 objects. Threshold {summary['threshold']:.2f} (best F1).", color=T["muted"], fontsize=9)
    return fig


def fig_range_budget(summary):
    """Ruler: where the detector works vs. how far a loaded truck travels before it can stop."""
    rp = {r["range"]: r for r in summary["sets"]["heldout"]["range_profile"] if r["group"] == "all"}
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    edges = [0, 10, 20, 30, 40, 50, 60, 80, 170]
    keys = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", T["seq"][1:])
    for (a, b), k in zip(zip(edges[:-1], edges[1:]), keys):
        v = rp[k]["recall"] or 0.0
        ax.add_patch(Rectangle((a, 0.35), b - a - 0.6, 0.4, color=cmap(v), lw=0))
        if b - a >= 10 and k != "80+":
            ink = ("#ffffff" if T is THEMES["light"] else "#0b0b0b") if v > 0.55 else T["text"]
            ax.text((a + b) / 2, 0.55, f"{v:.0%}", ha="center", va="center", fontsize=10, color=ink)
    ax.text(125, 0.55, "0 % recall", ha="center", va="center", fontsize=10, color=T["text2"])
    for x, lab, col, y in ((51.2, "BEV grid edge\n51.2 m", T["text"], 1.0),
                           (160, "≈160 m: loaded truck stopping\ndistance at 65 mph (≈525 ft)", T["s8"], 1.0)):
        ax.plot([x, x], [0.25, 0.85], color=col, lw=2)
        ax.text(x, y, lab, ha="center" if x < 150 else "right", va="bottom", fontsize=9.5, color=col)
    ax.text(0, 1.0, "Recall at the operating point, by distance (held-out)", fontsize=12, fontweight="semibold",
            va="bottom", color=T["text"])
    ax.set_xlim(-1, 171)
    ax.set_ylim(0.1, 1.35)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120, 140, 160], [f"{v} m" for v in range(0, 161, 20)])
    ax.tick_params(axis="x", length=0, labelsize=9)
    return fig


# --------------------------------------------------------------------------- range
def fig_range_metrics(summary, set_name="heldout"):
    rp = summary["sets"][set_name]["range_profile"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    x = _range_x()
    for ax, key, title in zip(axes, ("recall", "precision", "ap"),
                              ("Recall (GT bucketed by distance)", "Precision (predictions by distance)",
                               "AP, 2 m centre-distance match")):
        for g in ("car", "large_vehicle", "vru"):
            y = [r[key] if r[key] is not None else np.nan for r in rp if r["group"] == g]
            ax.plot(x, y, color=T[GROUP_COLOR[g]], marker="o", ms=6, mec=T["surface"], mew=1.5,
                    label=GROUP_LABEL[g])
        _shade_beyond_grid(ax, label=ax is axes[0])
        ax.set_xticks(x, RANGE_LABELS, rotation=0, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("metres from ego")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(loc="lower left", fontsize=9.5)
    n = [r["n_gt"] for r in rp if r["group"] == "all"]
    for i, v in enumerate(n):
        axes[0].text(i, -0.0, f"n={v}", ha="center", va="bottom", fontsize=7.5, color=T["muted"])
    fig.tight_layout()
    return fig


def fig_class_range_heatmap(summary):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", T["seq"])
    order = ["car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle",
             "traffic_cone", "barrier"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.9))
    for ax, set_name, title in zip(axes, ("heldout", "all_mini"),
                                   ("Held-out mini_val (2 scenes)", "All 10 mini scenes (8 seen in training)")):
        rows = summary["sets"][set_name]["class_range_recall"]
        M = np.full((len(order), len(RANGE_LABELS)), np.nan)
        N = np.zeros_like(M)
        for r in rows:
            if r["cls"] in order:
                i, j = order.index(r["cls"]), RANGE_LABELS.index(r["range_bin"])
                M[i, j], N[i, j] = r["mean"], r["size"]
        Mm = np.where(N >= 5, M, np.nan)
        ax.imshow(np.nan_to_num(Mm, nan=0), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(order)):
            for j in range(len(RANGE_LABELS)):
                if N[i, j] == 0:
                    continue
                if N[i, j] < 5:
                    ax.text(j, i, "·", ha="center", va="center", color=T["muted"], fontsize=9)
                    continue
                v = Mm[i, j]
                high = v > 0.55
                ink = ("#ffffff" if T is THEMES["light"] else "#0b0b0b") if high else T["text"]
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color=ink)
        ax.set_xticks(range(len(RANGE_LABELS)), RANGE_LABELS, fontsize=9)
        ax.set_yticks(range(len(order)), [o.replace("_", " ") for o in order], fontsize=9.5)
        ax.grid(False)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("metres from ego")
        ax.axvline(4.62, color=T["text2"], ls="--", lw=1)
    fig.suptitle("Recall by class × distance (cells with <5 objects shown as ·; dashed = BEV grid edge)",
                 x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    return fig


def fig_localization(summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.3))
    for g in ("car", "large_vehicle", "vru"):
        rows = [r for r in summary["sets"]["all_mini"]["tp_errors"] if r["group"] == g and r["n"] >= 8]
        xs = [r["range"] for r in rows]
        axes[0].plot(xs, [r["trans_med"] for r in rows], color=T[GROUP_COLOR[g]], marker="o", ms=6,
                     mec=T["surface"], mew=1.5, label=GROUP_LABEL[g])
        axes[0].plot(xs, [r["trans_p90"] for r in rows], color=T[GROUP_COLOR[g]], ls=":", lw=1.5)
        axes[1].plot(xs, [r["radial_bias"] for r in rows], color=T[GROUP_COLOR[g]], marker="o", ms=6,
                     mec=T["surface"], mew=1.5)
    axes[0].set_title("BEV centre error of true positives (solid = median, dotted = p90)", fontsize=12)
    axes[0].set_ylabel("metres")
    axes[0].legend(fontsize=9.5)
    axes[1].axhline(0, color=T["text2"], lw=1)
    axes[1].set_title("Radial bias (pred distance − true distance), median", fontsize=12)
    axes[1].set_ylabel("metres  (− = placed too close)")
    for ax in axes:
        ax.set_xlabel("true distance (m)")
    fig.text(0.01, -0.03, "Matched pairs from all 10 mini scenes. Error is capped by the 2 m match radius, "
             "so far-range p90 understates the true spread.", color=T["muted"], fontsize=9)
    fig.tight_layout()
    return fig


def fig_size(summary):
    from .diagnose import LENGTH_LABELS

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, set_name, title in zip(axes, ("heldout", "all_mini"), ("Held-out mini_val", "All 10 mini scenes")):
        rows = summary["sets"][set_name]["size_recall_by_range"]
        labs = [l for l in LENGTH_LABELS if any(r["length_bin"] == l for r in rows)]
        x = np.arange(len(labs))
        for k, (far, col) in enumerate((("<40 m", "s1"), (">=40 m", "s2"))):
            vals, ns = [], []
            for l in labs:
                r = next((r for r in rows if r["length_bin"] == l and r["far"] == far), None)
                vals.append(r["mean"] if r and r["size"] >= 5 else np.nan)
                ns.append(r["size"] if r else 0)
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=T[col], label=f"objects {far.replace('>=', '≥')}",
                          edgecolor=T["surface"], linewidth=2)
            for b, v, n in zip(bars, vals, ns):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=8.5,
                            color=T["text2"])
                    ax.text(b.get_x() + b.get_width() / 2, -0.06, f"n={n}", ha="center", fontsize=7.5,
                            color=T["muted"])
        ax.set_xticks(x, labs)
        ax.set_title(f"{title}: vehicle recall by true length", fontsize=12)
        ax.set_xlabel("ground-truth box length")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_ylim(-0.09, 1.05)
    axes[0].legend(fontsize=9.5, loc="upper left")
    fig.tight_layout()
    return fig


FN_LABEL = {"low_confidence": "found, but below threshold", "beyond_bev_grid": "outside ±51.2 m grid",
            "depth_error": "found 2–4 m off (depth)", "missed": "missed outright",
            "heavily_occluded": "≤40% visible", "class_confusion": "wrong class"}
FP_LABEL = {"depth_error": "real object, 2–4 m off", "duplicate": "duplicate box",
            "ghost_no_lidar": "ghost, 0 LiDAR pts", "ghost_with_lidar": "ghost, has LiDAR pts",
            "class_confusion": "wrong class"}


def fig_failures(summary, set_name="heldout"):
    fn_order = ["low_confidence", "depth_error", "missed", "heavily_occluded", "class_confusion", "beyond_bev_grid"]
    fn_cols = ["s4", "s2", "s8", "s7", "s3", "neutral"]
    fp_order = ["depth_error", "duplicate", "ghost_no_lidar", "ghost_with_lidar", "class_confusion"]
    fp_cols = ["s2", "s7", "s8", "s4", "s3"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for ax, key, order, cols, title in ((axes[0], "fn_tags", fn_order, fn_cols, "Why objects were missed (FN)"),
                                        (axes[1], "fp_tags", fp_order, fp_cols, "Why boxes were wrong (FP)")):
        rows = summary["sets"][set_name][key]
        x = _range_x()
        bottom = np.zeros(len(RANGE_LABELS))
        for tag, col in zip(order, cols):
            v = np.array([sum(r["n"] for r in rows if r["fail_tag"] == tag and r["range_bin"] == rb)
                          for rb in RANGE_LABELS], dtype=float)
            if v.sum() == 0:
                continue
            lab = (FN_LABEL if key == "fn_tags" else FP_LABEL)[tag]
            ax.bar(x, v, bottom=bottom, color=T[col], width=0.72, edgecolor=T["surface"], linewidth=1.5,
                   label=f"{lab} ({int(v.sum())})")
            bottom += v
        ax.set_xticks(x, RANGE_LABELS, fontsize=9)
        ax.set_xlabel("metres from ego (GT distance for FN, predicted distance for FP)")
        ax.set_ylabel("count")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def fig_lighting(summary):
    rows = summary["sets"]["all_mini"]["lighting_range_recall"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for light, col in (("day", "s4"), ("night", "s7")):
        y = []
        for rb in RANGE_LABELS:
            r = next((r for r in rows if r["lighting"] == light and r["range_bin"] == rb), None)
            y.append(r["mean"] if r and r["size"] >= 10 else np.nan)
        ax.plot(_range_x(), y, color=T[col], marker="o", ms=6, mec=T["surface"], mew=1.5,
                label=f"{light} ({sum(r['size'] for r in rows if r['lighting'] == light):,} objects)")
    _shade_beyond_grid(ax)
    ax.set_xticks(_range_x(), RANGE_LABELS, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Day vs night recall (all mini scenes; night scenes were in training)", fontsize=12)
    ax.set_xlabel("metres from ego")
    ax.legend(fontsize=9.5, loc="lower left")
    fig.tight_layout()
    return fig


def fig_fix(summary):
    fa = summary["fix_lidar_filter"]
    base = summary["sets"]["heldout"]["range_profile"]
    ch = fa["chosen"]["range_heldout"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    x = _range_x()[:6]
    for ax, key, title in ((axes[0], "precision", "Precision by predicted distance"),
                           (axes[1], "recall", "Recall by true distance")):
        b = [r[key] if r[key] is not None else np.nan for r in base if r["group"] == "all"][:6]
        a = [r[key] if r[key] is not None else np.nan for r in ch if r["group"] == "all"][:6]
        ax.bar(x - 0.2, b, width=0.38, color=T["neutral"], edgecolor=T["surface"], linewidth=2, label="baseline")
        ax.bar(x + 0.2, a, width=0.38, color=T["s1"], edgecolor=T["surface"], linewidth=2,
               label="+ LiDAR sanity filter")
        for i in range(len(x)):
            if not (np.isnan(a[i]) or np.isnan(b[i])):
                d = a[i] - b[i]
                ax.text(x[i] + 0.2, a[i] + 0.02, f"{d:+.0%}".replace("+0%", "±0%"), ha="center", fontsize=8.5,
                        color=T["text2"])
        ax.set_xticks(x, RANGE_LABELS[:6], fontsize=9)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("metres from ego")
    axes[0].legend(fontsize=9.5, loc="lower left")
    ax = axes[2]
    base_map = summary["baseline_official"]["mAP"]
    for enl, col, mk in ((0.0, "s2", "s"), (0.5, "s1", "o")):
        rows = [r for r in fa["sweep"] if r["enlarge_m"] == enl]
        ax.plot([r["min_points"] for r in rows], [r["mAP"] for r in rows], color=T[col], marker=mk, ms=7,
                mec=T["surface"], mew=1.5, label=f"box grown {enl:g} m")
    ax.axhline(base_map, color=T["text2"], ls="--", lw=1.2)
    ax.text(5, base_map + 0.002, f"baseline {base_map:.3f}", ha="right", va="bottom", fontsize=9, color=T["text2"])
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("min LiDAR points required inside box")
    ax.set_title("Official mAP (mini_val) vs filter setting", fontsize=12)
    ax.legend(fontsize=9.5, loc="lower left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- BEV + camera
def box_corners_bev(x, y, w, l, yaw_nusc):
    c, s = np.cos(yaw_nusc), np.sin(yaw_nusc)
    pts = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])


def box_corners_3d(x, y, z, w, l, h, yaw_nusc):
    c, s = np.cos(yaw_nusc), np.sin(yaw_nusc)
    dx = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * l / 2
    dy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    dz = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * h / 2
    return np.stack([x + c * dx - s * dy, y + s * dx + c * dy, z + dz], 1)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_bev(ax, gtm, prm, thr, points=None, lim=80, title=None, highlight=None):
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.6, lim)
    ax.grid(False)
    if points is not None:
        sel = (np.abs(points[:, 0]) < lim) & (np.abs(points[:, 1]) < lim)
        ax.scatter(points[sel, 0], points[sel, 1], s=0.15, color=T["muted"], alpha=0.35, lw=0, zorder=1)
    for r in (25, 50, 75):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=T["grid"], lw=1, zorder=0))
        ax.text(r * 0.707 + 1, r * 0.707, f"{r} m", color=T["muted"], fontsize=8)
    ax.add_patch(Rectangle((-BEV_HALF_EXTENT, -BEV_HALF_EXTENT), 2 * BEV_HALF_EXTENT, 2 * BEV_HALF_EXTENT,
                           fill=False, ls="--", lw=1.2, ec=T["text2"], zorder=2))
    ax.add_patch(Rectangle((-1, -2.3), 2, 4.6, color=T["text"], zorder=5))  # ego
    for _, g in gtm.iterrows():
        hit = g.matched and g.match_score >= thr
        c = box_corners_bev(g.x, g.y, g.w, g.l, g.yaw)
        ax.add_patch(Polygon(c, closed=True, fill=False, ec=T["text2"] if hit else T["s7"],
                             lw=1.0 if hit else 2.0, ls="-" if hit else (0, (2, 1.2)), zorder=3))
    for _, p in prm[prm.score >= thr].iterrows():
        c = box_corners_bev(p.x, p.y, p.w, p.l, p.yaw)
        col = T["s1"] if p.tp else T["s2"]
        ax.add_patch(Polygon(c, closed=True, fc=col, ec=col, alpha=0.55 if p.tp else 0.8, lw=1, zorder=4))
    if highlight is not None:
        hx, hy = highlight
        ax.add_patch(plt.Circle((hx, hy), 6, fill=False, ec=T["s4"], lw=2.2, zorder=6))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)


def bev_legend(ax):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    h = [Patch(fc=T["s1"], ec=T["s1"], alpha=0.6, label="true positive"),
         Patch(fc=T["s2"], ec=T["s2"], label="false positive"),
         Line2D([], [], color=T["s7"], lw=2.0, ls=(0, (2, 1.2)), label="missed GT"),
         Line2D([], [], color=T["text2"], lw=1, label="detected GT"),
         Line2D([], [], color=T["text2"], lw=1.2, ls="--", label="BEV grid ±51.2 m")]
    ax.legend(handles=h, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0),
              handlelength=1.6, columnspacing=1.0)


def project_boxes(lidar2img, corners_list, img_wh):
    """Project 3D corner sets with a 4x4 lidar2img; returns list of (8,2) or None if behind camera."""
    out = []
    for c in corners_list:
        h = np.c_[c, np.ones(8)] @ lidar2img.T
        if (h[:, 2] <= 0.5).sum() > 0:
            out.append(None)
            continue
        uv = h[:, :2] / h[:, 2:3]
        if (uv[:, 0].max() < 0) or (uv[:, 0].min() > img_wh[0]) or (uv[:, 1].max() < 0) or (uv[:, 1].min() > img_wh[1]):
            out.append(None)
            continue
        out.append(uv)
    return out


def draw_camera(ax, img, lidar2img, gtm, prm, thr, title=None):
    ax.imshow(img)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    items = []
    for _, g in gtm.iterrows():
        hit = g.matched and g.match_score >= thr
        if not hit:
            items.append((box_corners_3d(g.x, g.y, g.z, g.w, g.l, g.h, g.yaw), "#9085e9", 2.2, "--"))
    for _, p in prm[prm.score >= thr].iterrows():
        items.append((box_corners_3d(p.x, p.y, p.z, p.w, p.l, p.h, p.yaw),
                      THEMES["light"]["s1"] if p.tp else THEMES["light"]["s2"], 1.6, "-"))
    proj = project_boxes(lidar2img, [i[0] for i in items], (img.shape[1], img.shape[0]))
    n = 0
    for uv, (_, col, lw, ls) in zip(proj, items):
        if uv is None:
            continue
        n += 1
        for a, b in EDGES:
            ax.plot(uv[[a, b], 0], uv[[a, b], 1], color=col, lw=lw, ls=ls)
    if title:
        ax.set_title(title, fontsize=11)
    return n
