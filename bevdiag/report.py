"""Render all figures, the worst-case gallery, the hero GIF and the interactive HTML report."""
from __future__ import annotations

import io
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import viz
from .data import CAM_ORDER, load_lidar_points
from .diagnose import LARGE_VEHICLES

NICE = {"truck": "truck", "bus": "bus", "trailer": "trailer", "construction_vehicle": "construction vehicle",
        "car": "car", "pedestrian": "pedestrian", "motorcycle": "motorcycle", "bicycle": "bicycle",
        "traffic_cone": "cone", "barrier": "barrier"}


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else ("es" if word.endswith(("s", "ch")) else "s"))


def diagnose_frame(g: pd.DataFrame, p: pd.DataFrame, thr: float) -> tuple[str, list[str], tuple | None]:
    """One-line diagnosis + tag list + BEV focus point for a single frame."""
    fn = g[g.fail_tag.notna()]
    fp = p[p.fail_tag.notna()]
    parts, tags, focus = [], [], None
    big_out = fn[(fn.fail_tag == "beyond_bev_grid") & fn.cls.isin(LARGE_VEHICLES)]
    if len(big_out):
        if len(big_out) == 1:
            r = big_out.iloc[0]
            parts.append(f"a {r.l:.0f} m {NICE[r.cls]} at {r.dist:.0f} m sits outside the ±51.2 m BEV grid "
                         f"and can never be detected")
        else:
            parts.append(f"{len(big_out)} large vehicles at {big_out.dist.min():.0f}–{big_out.dist.max():.0f} m "
                         f"sit outside the ±51.2 m BEV grid and can never be detected")
        tags.append("beyond_bev_grid")
        focus = focus or (big_out.iloc[0].x, big_out.iloc[0].y)
    big_in = fn[fn.cls.isin(LARGE_VEHICLES) & (fn.fail_tag != "beyond_bev_grid")]
    if len(big_in):
        r = big_in.sort_values("dist").iloc[0]
        why = {"low_confidence": f"was seen but scored below {thr:.2f}", "missed": "got no box at all",
               "depth_error": "got a box 2–4 m off in depth", "heavily_occluded": "is ≤40 % visible and was missed",
               "class_confusion": "was detected as the wrong class"}[r.fail_tag]
        parts.append(f"a {r.l:.0f} m {NICE[r.cls]} at {r.dist:.0f} m {why}")
        tags.append(r.fail_tag)
        focus = focus or (r.x, r.y)
    for tag, txt in (("depth_error", "{n} above-threshold box{es} placed 2–4 m off the true depth"),
                     ("ghost_no_lidar", "{n} ghost box{es} with zero LiDAR returns"),
                     ("duplicate", "{n} duplicate box{es} on already-detected objects")):
        k = int((fp.fail_tag == tag).sum())
        if k >= 2:
            parts.append(txt.format(n=k, es="" if k == 1 else "es"))
            tags.append(tag)
            if focus is None:
                q = fp[fp.fail_tag == tag].sort_values("score", ascending=False).iloc[0]
                focus = (q.x, q.y)
    lc = int((fn.fail_tag == "low_confidence").sum())
    if lc >= 3 and len(parts) < 3:
        parts.append(f"{lc} more objects were found but scored under the threshold")
        tags.append("low_confidence")
    if not parts:
        parts.append(f"{len(fn)} misses, {len(fp)} false positives")
    tags = list(dict.fromkeys(tags))
    line = parts[0][0].upper() + parts[0][1:] + ("; " + "; ".join(parts[1:3]) if len(parts) > 1 else "") + "."
    return line, tags, focus


def _pick_camera(info, g, p, thr):
    best, best_n = 0, -1
    fn = g[g.fail_tag.notna() & g.in_bev_grid]
    fp = p[p.fail_tag.notna()]
    for ci in range(6):
        corners = [viz.box_corners_3d(r.x, r.y, r.z, r.w, r.l, r.h, r.yaw) for _, r in fn.iterrows()] + \
                  [viz.box_corners_3d(r.x, r.y, r.z, r.w, r.l, r.h, r.yaw) for _, r in fp.iterrows()]
        n = sum(u is not None for u in viz.project_boxes(info.lidar2img[ci], corners, (1600, 900)))
        n += 0.5 if CAM_ORDER[ci] == "CAM_FRONT" else 0
        if n > best_n:
            best, best_n = ci, n
    return best


def _load_cam(info, ci, scale=0.5):
    img = cv2.cvtColor(cv2.imread(info.cam_paths[ci]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    s = np.eye(4)
    s[0, 0] = s[1, 1] = scale
    return img, s @ info.lidar2img[ci]


def fig_case(info, g, p, thr, line, focus, title):
    ci = _pick_camera(info, g, p, thr)
    img, l2i = _load_cam(info, ci)
    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.02)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    viz.draw_camera(a1, img, l2i, g, p, thr, title=CAM_ORDER[ci].replace("_", " "))
    viz.draw_bev(a2, g, p, thr, points=load_lidar_points(info.lidar_path), lim=66, title="Bird's-eye view (forward = up)",
                 highlight=focus)
    viz.bev_legend(a2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold", y=1.02)
    fig.text(0.01, 0.955, line, ha="left", fontsize=11, color=viz.T["text2"], wrap=True)
    return fig, CAM_ORDER[ci]


def render_gallery(infos, gtm, prm, thr, sample_ids, out_dir, prefix):
    cases = []
    for rank, si in enumerate(sample_ids):
        info = infos[si]
        g, p = gtm[gtm.sample_idx == si], prm[prm.sample_idx == si]
        line, tags, focus = diagnose_frame(g, p, thr)
        frame = sum(1 for j in range(si) if infos[j].scene_name == info.scene_name)
        title = f"{info.scene_name} · frame {frame} · {info.tags['lighting']}"
        stem = os.path.join(out_dir, f"{prefix}_{rank:02d}")
        cam = None
        for th in ("light", "dark"):
            with viz.theme(th):
                fig, cam = fig_case(info, g, p, thr, line, focus, title)
                fig.savefig(f"{stem}_{th}.jpg", dpi=110, bbox_inches="tight", pil_kwargs={"quality": 84})
                plt.close(fig)
        n_fn = int(g.fail_tag.notna().sum())
        n_fp = int(p.fail_tag.notna().sum())
        cases.append({"rank": rank, "sample_token": info.token, "scene": info.scene_name, "frame": frame,
                      "lighting": info.tags["lighting"], "camera": cam, "diagnosis": line, "tags": tags,
                      "n_fn": n_fn, "n_fp": n_fp,
                      "n_fn_large": int((g.fail_tag.notna() & g.cls.isin(LARGE_VEHICLES)).sum()),
                      "image_light": os.path.basename(stem) + "_light.jpg",
                      "image_dark": os.path.basename(stem) + "_dark.jpg"})
    return cases


def render_gif(infos, gtm, prm, thr, scene, path, max_frames=40):
    frames = []
    idx = [i for i, inf in enumerate(infos) if inf.scene_name == scene][:max_frames]
    with viz.theme("dark"):
        for k, si in enumerate(idx):
            info = infos[si]
            g, p = gtm[gtm.sample_idx == si], prm[prm.sample_idx == si]
            img, l2i = _load_cam(info, 0, scale=0.4)
            fig = plt.figure(figsize=(12.8, 4.3))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.02)
            a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
            viz.draw_camera(a1, img, l2i, g, p, thr)
            viz.draw_bev(a2, g, p, thr, points=load_lidar_points(info.lidar_path)[::2], lim=70)
            hit = int((g.matched & (g.match_score >= thr)).sum())
            fig.text(0.01, 0.985, f"{scene} (held-out) · frame {k + 1}/{len(idx)} · BEVFormer-tiny, camera only",
                     fontsize=11, color=viz.T["text"], fontweight="semibold", va="top")
            fig.text(0.01, 0.925, f"detected {hit}/{len(g)} objects · {int((~p.tp & (p.score >= thr)).sum())} false "
                     f"positives  ·  blue = TP, orange = FP, violet dashed = missed, dashed square = ±51.2 m grid",
                     fontsize=9.5, color=viz.T["text2"], va="top")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=72)
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGB"))
    pal = frames[len(frames) // 2].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=350, loop=0, optimize=True)


def build_report(nusc, infos, preds, summary, out_dir, docs_dir, improved_preds=None):
    assets = os.path.join(docs_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    thr = summary["threshold"]
    print("  charts")
    viz.save_both(viz.fig_hero, os.path.join(assets, "hero"), summary=summary)
    viz.save_both(viz.fig_range_budget, os.path.join(assets, "range_budget"), summary=summary)
    viz.save_both(viz.fig_range_metrics, os.path.join(assets, "range_metrics"), summary=summary)
    viz.save_both(viz.fig_class_range_heatmap, os.path.join(assets, "class_range_heatmap"), summary=summary)
    viz.save_both(viz.fig_localization, os.path.join(assets, "localization"), summary=summary)
    viz.save_both(viz.fig_size, os.path.join(assets, "size"), summary=summary)
    viz.save_both(viz.fig_failures, os.path.join(assets, "failures"), summary=summary)
    viz.save_both(viz.fig_lighting, os.path.join(assets, "lighting"), summary=summary)
    viz.save_both(viz.fig_fix, os.path.join(assets, "fix"), summary=summary)

    print("  gallery")
    gal = os.path.join(assets, "gallery")
    os.makedirs(gal, exist_ok=True)
    gh = pd.read_pickle(os.path.join(out_dir, "gt_heldout.pkl"))
    ph = pd.read_pickle(os.path.join(out_dir, "pred_heldout.pkl"))
    ga = pd.read_pickle(os.path.join(out_dir, "gt_all_mini.pkl"))
    pa = pd.read_pickle(os.path.join(out_dir, "pred_all_mini.pkl"))
    held = summary["sets"]["heldout"]["worst_cases"][:4]
    seen = [i for i in summary["sets"]["all_mini"]["worst_cases"] if i not in held][:4]
    cases = render_gallery(infos, gh, ph, thr, held, gal, "heldout") + \
        render_gallery(infos, ga, pa, thr, seen, gal, "trainscene")
    for c in cases:
        c["split"] = "held-out" if c["image_light"].startswith("heldout") else "seen in training"
    summary["gallery"] = cases

    print("  gif")
    big = gh[gh.cls.isin(LARGE_VEHICLES)].groupby("scene").size()
    scene = big.idxmax() if len(big) else gh.scene.iloc[0]
    render_gif(infos, gh, ph, thr, scene, os.path.join(assets, "scene.gif"))
    summary["gif_scene"] = scene

    if "improved" in summary:
        render_improvement(infos, summary, out_dir, assets, gh, ph, scene)

    tpath = os.path.join(out_dir, "timing.json")
    if os.path.exists(tpath):
        with open(tpath) as f:
            summary["timing"] = json.load(f)
    with open(os.path.join(docs_dir, "report_data.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    from .html import write_html

    write_html(summary, os.path.join(docs_dir, "index.html"))


# --------------------------------------------------------------------------- before / after
def _counts(g, p, thr):
    hit = int((g.matched & (g.match_score >= thr)).sum())
    fp = int((~p.tp & (p.score >= thr)).sum())
    return hit, len(g), fp


def fig_compare_case(info, gb, pb, tb, ga, pa, ta, title, system):
    img, l2i = _load_cam(info, _pick_camera(info, gb, pb, tb))
    pts = load_lidar_points(info.lidar_path)
    fig = plt.figure(figsize=(18, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.03)
    a0, a1, a2 = (fig.add_subplot(gs[i]) for i in range(3))
    viz.draw_camera(a0, img, l2i, ga, pa, ta, title="camera, improved system")
    hb, nb, fb = _counts(gb, pb, tb)
    ha, na, fa = _counts(ga, pa, ta)
    viz.draw_bev(a1, gb, pb, tb, points=pts, lim=60, title=f"BEFORE: tiny as released  ·  {hb}/{nb} found · {fb} FP")
    viz.draw_bev(a2, ga, pa, ta, points=pts, lim=60, title=f"AFTER: {system}  ·  {ha}/{na} found · {fa} FP")
    viz.bev_legend(a2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold", y=0.9)
    return fig, (hb, fb, ha, fa)


def render_improvement(infos, summary, out_dir, assets, gh, ph, scene):
    from . import viz_compare as vc

    print("  before/after charts")
    for fn, name in ((vc.fig_improve_hero, "improve_hero"), (vc.fig_ladder, "improve_ladder"),
                     (vc.fig_before_after_range, "improve_range"), (vc.fig_class_ap_compare, "improve_class_ap"),
                     (vc.fig_errors_compare, "improve_errors")):
        viz.save_both(fn, os.path.join(assets, name), summary=summary)
    ga = pd.read_pickle(os.path.join(out_dir, "gt_improved_heldout.pkl"))
    pa = pd.read_pickle(os.path.join(out_dir, "pred_improved_heldout.pkl"))
    tb, ta = summary["threshold"], summary["improved"]["threshold"]
    system = vc.system_name(summary).replace("BEVFormer-", "")
    print("  before/after gallery")
    comp = os.path.join(assets, "compare")
    os.makedirs(comp, exist_ok=True)
    cases = []
    for k, si in enumerate(summary["sets"]["heldout"]["worst_cases"][:4]):
        info = infos[si]
        frame = sum(1 for j in range(si) if infos[j].scene_name == info.scene_name)
        title = f"{info.scene_name} · frame {frame} (held-out): the same frame before and after"
        for th in ("light", "dark"):
            with viz.theme(th):
                fig, cnt = fig_compare_case(info, gh[gh.sample_idx == si], ph[ph.sample_idx == si], tb,
                                            ga[ga.sample_idx == si], pa[pa.sample_idx == si], ta, title, system)
                fig.savefig(os.path.join(comp, f"compare_{k:02d}_{th}.jpg"), dpi=100, bbox_inches="tight",
                            pil_kwargs={"quality": 84})
                plt.close(fig)
        cases.append({"scene": info.scene_name, "frame": frame, "found_before": cnt[0], "fp_before": cnt[1],
                      "found_after": cnt[2], "fp_after": cnt[3], "n_gt": int((gh.sample_idx == si).sum()),
                      "image": f"compare_{k:02d}"})
    summary["compare_gallery"] = cases
    print("  before/after gif")
    render_compare_gif(infos, gh, ph, tb, ga, pa, ta, scene, os.path.join(assets, "compare.gif"), system)


def render_compare_gif(infos, gb, pb, tb, ga, pa, ta, scene, path, system, max_frames=40):
    frames = []
    idx = [i for i, inf in enumerate(infos) if inf.scene_name == scene][:max_frames]
    with viz.theme("dark"):
        for k, si in enumerate(idx):
            info = infos[si]
            g0, p0 = gb[gb.sample_idx == si], pb[pb.sample_idx == si]
            g1, p1 = ga[ga.sample_idx == si], pa[pa.sample_idx == si]
            img, l2i = _load_cam(info, 0, scale=0.4)
            pts = load_lidar_points(info.lidar_path)[::2]
            fig = plt.figure(figsize=(15, 4.3))
            gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.02)
            a0, a1, a2 = (fig.add_subplot(gs[i]) for i in range(3))
            viz.draw_camera(a0, img, l2i, g1, p1, ta)
            h0, n0, f0 = _counts(g0, p0, tb)
            h1, n1, f1 = _counts(g1, p1, ta)
            viz.draw_bev(a1, g0, p0, tb, points=pts, lim=60, title=f"before (tiny): {h0}/{n0} found, {f0} FP")
            viz.draw_bev(a2, g1, p1, ta, points=pts, lim=60, title=f"after ({system}): {h1}/{n1} found, {f1} FP")
            fig.text(0.01, 0.99, f"{scene} (held-out) · frame {k + 1}/{len(idx)} · front camera shows the improved system",
                     fontsize=11, color=viz.T["text"], fontweight="semibold", va="top")
            fig.text(0.01, 0.93, "blue = true positive · orange = false positive · violet dashed = missed · "
                     "dashed square = ±51.2 m BEV grid", fontsize=9.5, color=viz.T["text2"], va="top")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=68)
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGB"))
    pal = frames[len(frames) // 2].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=380, loop=0, optimize=True)
