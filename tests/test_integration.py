"""Integration tests that need the checkpoint / nuScenes mini (skipped when absent)."""
import json
import os
import pickle

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "bevformer_tiny_epoch_24.pth")
DATA = os.path.join(ROOT, "data", "nuscenes")
OUT = os.path.join(ROOT, "outputs")

needs_ckpt = pytest.mark.skipif(not os.path.exists(CKPT), reason="checkpoint not downloaded")
needs_data = pytest.mark.skipif(not os.path.isdir(os.path.join(DATA, "v1.0-mini")), reason="nuScenes mini missing")


@needs_ckpt
def test_checkpoint_loads_strict():
    from bevdiag.model import load_bevformer_tiny

    m = load_bevformer_tiny(CKPT, "cpu")  # raises on any missing / unexpected key
    n = sum(p.numel() for p in m.parameters())
    assert 30e6 < n < 40e6


@needs_ckpt
@needs_data
def test_first_frame_detects_cars():
    from nuscenes.nuscenes import NuScenes

    from bevdiag.data import CameraDataset, build_infos
    from bevdiag.model import TemporalRunner, decode, load_bevformer_tiny

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nusc = NuScenes("v1.0-mini", DATA, verbose=False)
    info = build_infos(nusc, ["scene-0103"])[:1]
    s = CameraDataset(info)[0]
    r = TemporalRunner(load_bevformer_tiny(CKPT, dev))
    cls, box = r.step(s["img"][None].to(dev), s["lidar2img"][None], info[0].can_bus, info[0].scene_token,
                      s["img_shape"])
    b, sc, lab = decode(cls, box)
    assert (sc > 0.5).sum() >= 3
    assert b.shape[1] == 9


@needs_data
@pytest.mark.skipif(not os.path.exists(os.path.join(OUT, "raw_preds.pkl")), reason="run `python -m bevdiag run` first")
def test_official_metrics_reproduce():
    """Re-scores cached predictions with the devkit; guards against regressions in box conversion."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import mini_val

    from bevdiag.data import build_infos
    from bevdiag.official import official_eval

    nusc = NuScenes("v1.0-mini", DATA, verbose=False)
    infos = build_infos(nusc, mini_val)
    with open(os.path.join(OUT, "raw_preds.pkl"), "rb") as f:
        preds = {p["token"]: p for p in pickle.load(f)}
    m = official_eval(nusc, infos, [preds[i.token] for i in infos], "mini_val", os.path.join(OUT, "test_tmp"), "t")
    with open(os.path.join(OUT, "metrics_baseline_mini_val.json")) as f:
        ref = json.load(f)
    assert m["mAP"] == pytest.approx(ref["mAP"], abs=1e-6)
    assert 0.22 < m["mAP"] < 0.32  # published BEVFormer-tiny: 0.252 on full val
