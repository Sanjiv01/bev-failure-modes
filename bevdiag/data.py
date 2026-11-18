"""nuScenes loading for BEVFormer inference + GT for diagnostics.

Reproduces the info / meta construction of the BEVFormer reference code
(tools/data_converter/nuscenes_converter.py + CustomNuScenesDataset) without mmdet3d:
camera order, lidar2img, CAN-bus vector (including the reference implementation's
quirk of reading accel/rotation_rate/vel from the *next* pose message), 0.5x image
scale and pad-to-32.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion

CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CLASS_NAMES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
               "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # RGB, torchvision-style (tiny)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
CAFFE_MEAN = np.array([103.530, 116.280, 123.675], dtype=np.float32)  # BGR, std 1 (small / base)
IMG_SCALE = 0.5
PAD_DIVISOR = 32


def transform_matrix(translation, rotation) -> np.ndarray:
    """4x4 homogeneous transform from a nuScenes translation + quaternion."""
    m = np.eye(4)
    m[:3, :3] = Quaternion(rotation).rotation_matrix
    m[:3, 3] = np.asarray(translation)
    return m


def quaternion_yaw(q: Quaternion) -> float:
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    return float(np.arctan2(v[1], v[0]))


@dataclass
class SampleInfo:
    token: str
    scene_token: str
    scene_name: str
    scene_desc: str
    timestamp: int
    lidar_token: str
    lidar_path: str
    lidar2ego: np.ndarray
    ego2global: np.ndarray
    cam_paths: list
    cam_tokens: list
    lidar2img: list  # full-resolution 4x4 per camera
    cam_intrinsics: list
    lidar2cam: list
    can_bus: np.ndarray
    tags: dict = field(default_factory=dict)

    @property
    def lidar2global(self) -> np.ndarray:
        return self.ego2global @ self.lidar2ego


def scene_tags(desc: str) -> dict:
    d = desc.lower()
    return {"lighting": "night" if "night" in d else "day",
            "weather": "rain" if "rain" in d else "dry"}


def can_bus_vector(nusc_can, scene_name: str, sample_timestamp: int) -> np.ndarray:
    """18-d CAN bus vector exactly as the BEVFormer converter builds it."""
    if nusc_can is None:
        return np.zeros(18)
    try:
        pose_list = nusc_can.get_messages(scene_name, "pose")
    except Exception:
        return np.zeros(18)
    if not pose_list:
        return np.zeros(18)
    last_pose = pose_list[0]
    pose = last_pose
    for pose in pose_list:
        if pose["utime"] > sample_timestamp:
            break
        last_pose = pose
    can = []
    keys = [k for k in last_pose.keys() if k not in ("utime", "pos", "orientation")]
    can.extend(last_pose["pos"])
    can.extend(last_pose["orientation"])
    for k in keys:  # reference code reads these from `pose`, the first message after the sample
        can.extend(pose[k])
    can.extend([0.0, 0.0])
    return np.asarray(can, dtype=np.float64)


def build_infos(nusc, scene_names, nusc_can=None) -> list[SampleInfo]:
    """One SampleInfo per keyframe, ordered by scene then time (required for temporal BEV)."""
    infos = []
    scenes = [s for s in nusc.scene if s["name"] in set(scene_names)]
    scenes.sort(key=lambda s: nusc.get("sample", s["first_sample_token"])["timestamp"])
    for scene in scenes:
        tok = scene["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            infos.append(_sample_info(nusc, nusc_can, sample, scene))
            tok = sample["next"]
    return infos


def _sample_info(nusc, nusc_can, sample, scene) -> SampleInfo:
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    l_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    l_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    lidar2ego = transform_matrix(l_cs["translation"], l_cs["rotation"])
    ego2global = transform_matrix(l_pose["translation"], l_pose["rotation"])
    global2lidar = np.linalg.inv(ego2global @ lidar2ego)

    cam_paths, cam_tokens, lidar2img, intrinsics, lidar2cam = [], [], [], [], []
    for cam in CAM_ORDER:
        sd = nusc.get("sample_data", sample["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        cam2global = transform_matrix(pose["translation"], pose["rotation"]) @ \
            transform_matrix(cs["translation"], cs["rotation"])
        l2c = np.linalg.inv(global2lidar @ cam2global)  # lidar -> camera
        viewpad = np.eye(4)
        viewpad[:3, :3] = np.asarray(cs["camera_intrinsic"])
        cam_paths.append(os.path.join(nusc.dataroot, sd["filename"]))
        cam_tokens.append(sd["token"])
        lidar2img.append(viewpad @ l2c)
        intrinsics.append(viewpad)
        lidar2cam.append(l2c)

    can = can_bus_vector(nusc_can, scene["name"], sample["timestamp"])
    rot = Quaternion(l_pose["rotation"])
    can[:3] = l_pose["translation"]
    can[3:7] = l_pose["rotation"]
    patch_angle = quaternion_yaw(rot) / np.pi * 180
    if patch_angle < 0:
        patch_angle += 360
    can[-2] = patch_angle / 180 * np.pi
    can[-1] = patch_angle

    return SampleInfo(
        token=sample["token"], scene_token=scene["token"], scene_name=scene["name"],
        scene_desc=scene["description"], timestamp=sample["timestamp"],
        lidar_token=lidar_sd["token"], lidar_path=os.path.join(nusc.dataroot, lidar_sd["filename"]),
        lidar2ego=lidar2ego, ego2global=ego2global, cam_paths=cam_paths, cam_tokens=cam_tokens,
        lidar2img=lidar2img, cam_intrinsics=intrinsics, lidar2cam=lidar2cam, can_bus=can,
        tags=scene_tags(scene["description"]))


def load_image_tensor(paths, scale: float = IMG_SCALE, norm: str = "torchvision") -> tuple[torch.Tensor, tuple[int, int]]:
    """mmcv-equivalent test pipeline: normalize, resize by `scale` (bilinear), pad to /32.

    norm="torchvision": BGR->RGB, (x-mean)/std (BEVFormer-tiny); norm="caffe": stay BGR, x-mean (small/base).
    """
    out = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if norm == "caffe":
            img = img - CAFFE_MEAN
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img - IMG_MEAN) / IMG_STD
        h, w = img.shape[:2]
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        h2, w2 = img.shape[:2]
        ph = int(np.ceil(h2 / PAD_DIVISOR)) * PAD_DIVISOR
        pw = int(np.ceil(w2 / PAD_DIVISOR)) * PAD_DIVISOR
        pad = np.zeros((ph, pw, 3), dtype=np.float32)
        pad[:h2, :w2] = img
        out.append(pad)
    arr = np.stack(out).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(arr)), (arr.shape[2], arr.shape[3])


def scaled_lidar2img(info: SampleInfo, scale: float = IMG_SCALE) -> np.ndarray:
    s = np.eye(4)
    s[0, 0] = s[1, 1] = scale
    return np.stack([s @ m for m in info.lidar2img])


class CameraDataset(torch.utils.data.Dataset):
    """Yields preprocessed 6-camera tensors in temporal order (keep shuffle=False)."""

    def __init__(self, infos, scale: float = IMG_SCALE, norm: str = "torchvision"):
        self.infos = infos
        self.scale, self.norm = scale, norm

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, i):
        info = self.infos[i]
        img, shape = load_image_tensor(info.cam_paths, self.scale, self.norm)
        return {"index": i, "img": img, "img_shape": shape,
                "lidar2img": torch.from_numpy(scaled_lidar2img(info, self.scale))}


def load_lidar_points(path: str) -> np.ndarray:
    """(N, 3) LIDAR_TOP points in the lidar frame."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :3]
