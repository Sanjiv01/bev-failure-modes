"""Run BEVFormer-tiny over nuScenes keyframes; save raw LiDAR-frame predictions + nuScenes JSON."""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import torch
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from .data import CLASS_NAMES, CameraDataset, SampleInfo
from .model import CONFIGS, TemporalRunner, decode, load_bevformer

DEFAULT_ATTR = {"car": "vehicle.parked", "pedestrian": "pedestrian.moving", "trailer": "vehicle.parked",
                "truck": "vehicle.parked", "bus": "vehicle.moving", "motorcycle": "cycle.without_rider",
                "construction_vehicle": "vehicle.parked", "bicycle": "cycle.without_rider",
                "barrier": "", "traffic_cone": ""}


def attribute(name: str, vx: float, vy: float) -> str:
    """Same attribute heuristic as mmdet3d's nuScenes formatter."""
    if np.hypot(vx, vy) > 0.2:
        if name in ("car", "construction_vehicle", "bus", "truck", "trailer"):
            return "vehicle.moving"
        if name in ("bicycle", "motorcycle"):
            return "cycle.with_rider"
        return DEFAULT_ATTR[name]
    if name == "pedestrian":
        return "pedestrian.standing"
    if name == "bus":
        return "vehicle.stopped"
    return DEFAULT_ATTR[name]


def lidar_boxes_to_global(boxes: np.ndarray, info: SampleInfo) -> list[Box]:
    """boxes: (N, 9) x, y, z_center, w, l, h, yaw_mmdet3d, vx, vy in the LiDAR frame."""
    out = []
    l2e, e2g = info.lidar2ego, info.ego2global
    q_l2e, q_e2g = Quaternion(matrix=l2e[:3, :3]), Quaternion(matrix=e2g[:3, :3])
    for b in boxes:
        yaw = -b[6] - np.pi / 2  # mmdet3d<1.0 LiDAR yaw -> nuScenes yaw
        box = Box(b[:3].tolist(), b[3:6].tolist(), Quaternion(axis=[0, 0, 1], radians=yaw),
                  velocity=(b[7], b[8], 0.0))
        box.rotate(q_l2e)
        box.translate(l2e[:3, 3])
        box.rotate(q_e2g)
        box.translate(e2g[:3, 3])
        out.append(box)
    return out


def to_nusc_results(infos, preds) -> dict:
    results = {}
    for info, p in zip(infos, preds):
        rows = []
        for box, s, l in zip(lidar_boxes_to_global(p["boxes"], info), p["scores"], p["labels"]):
            name = CLASS_NAMES[int(l)]
            rows.append({"sample_token": info.token, "translation": box.center.tolist(),
                         "size": box.wlh.tolist(), "rotation": box.orientation.elements.tolist(),
                         "velocity": box.velocity[:2].tolist(), "detection_name": name,
                         "detection_score": float(s), "attribute_name": attribute(name, *box.velocity[:2])})
        results[info.token] = rows
    return {"meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                     "use_map": False, "use_external": True}, "results": results}


@torch.no_grad()
def run_inference(infos, ckpt: str, out_dir: str, workers: int = 4, video_test_mode: bool = True,
                  device: str = "cuda", variant: str = "tiny") -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    cfg = CONFIGS[variant]
    model = load_bevformer(ckpt, device, variant)
    runner = TemporalRunner(model, video_test_mode=video_test_mode)
    loader = torch.utils.data.DataLoader(CameraDataset(infos, cfg.img_scale, cfg.img_norm), batch_size=None, shuffle=False,
                                         num_workers=workers, pin_memory=True,
                                         persistent_workers=False, prefetch_factor=2 if workers else None)
    preds, times = [], []
    for batch in loader:
        info = infos[batch["index"]]
        img = batch["img"][None].to(device, non_blocking=True)
        t0 = time.perf_counter()
        cls, box = runner.step(img, batch["lidar2img"][None], info.can_bus, info.scene_token,
                               tuple(int(x) for x in batch["img_shape"]))
        boxes, scores, labels = decode(cls.float(), box.float())
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        preds.append({"token": info.token, "boxes": boxes.cpu().numpy().astype(np.float64),
                      "scores": scores.cpu().numpy(), "labels": labels.cpu().numpy()})
        if len(preds) % 50 == 0:
            print(f"  [{len(preds)}/{len(infos)}] {np.median(times) * 1000:.0f} ms/frame")
    with open(os.path.join(out_dir, "raw_preds.pkl"), "wb") as f:
        pickle.dump(preds, f)
    with open(os.path.join(out_dir, "results_nusc.json"), "w") as f:
        json.dump(to_nusc_results(infos, preds), f)
    stats = {"model": cfg.name, "frames": len(preds), "median_ms": float(np.median(times[1:] if len(times) > 1 else times) * 1000),
             "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
             "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else 0.0}
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[infer] {stats}")
    return preds
