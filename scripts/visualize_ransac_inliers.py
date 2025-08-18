from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml
import sys


ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party" / "Depth2HHA-python"
if str(TP) not in sys.path:
    sys.path.insert(0, str(TP))
from utils.rgbd_util import processDepthImage  # type: ignore


def read_depth_txt(depth_txt: Path) -> np.ndarray:
    H = None
    W = None
    lines = depth_txt.read_text(encoding="utf-8").splitlines()
    for line in lines[:10]:
        ls = line.lower()
        if ls.startswith("width:"):
            W = int(line.split(":", 1)[1])
        if ls.startswith("height:"):
            H = int(line.split(":", 1)[1])
    if H is None or W is None:
        max_r = 0
        max_c = 0
        for s in lines:
            s = s.strip()
            if not s or s[0].isalpha():
                continue
            r, c, _ = s.split(",")
            max_r = max(max_r, int(r))
            max_c = max(max_c, int(c))
        H = max_r + 1
        W = max_c + 1
    Zmm = np.zeros((H, W), np.float32)
    for s in lines:
        s = s.strip()
        if not s or s[0].isalpha():
            continue
        r, c, v = s.split(",")
        Zmm[int(r), int(c)] = float(v)
    return Zmm


def build_roi_mask(H: int, W: int, bottom_band: float, side_band: float, center_exclude_width: float) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    roi_start = int((1.0 - bottom_band) * H)
    mask[roi_start:, :] = True
    if side_band > 0.0:
        sw = max(1, int(side_band * W))
        mask[:, :sw] = True
        mask[:, W - sw:] = True
    if center_exclude_width > 0.0:
        hw = int(0.5 * center_exclude_width * W)
        cx = W // 2
        mask[roi_start:, max(0, cx - hw): min(W, cx + hw)] = False
    return mask


def fit_plane_ransac(P: np.ndarray, max_iters: int, thresh: float, min_inlier_ratio: float, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, float] | None:
    if rng is None:
        rng = np.random.default_rng()
    n_pts = P.shape[0]
    if n_pts < 3:
        return None
    best_inliers = None
    best_model = None
    for _ in range(max_iters):
        idx = rng.choice(n_pts, size=3, replace=False)
        p1, p2, p3 = P[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            continue
        n = n / norm_n
        d = -float(np.dot(n, p1))
        dist = np.abs(P @ n + d)
        inliers = dist <= thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_model = (n, d)
    if best_inliers is None or best_inliers.sum() < max(int(min_inlier_ratio * n_pts), 50):
        return None
    Pin = P[best_inliers]
    Pc = Pin.mean(axis=0)
    _, _, Vt = np.linalg.svd(Pin - Pc, full_matrices=False)
    n_ls = Vt[-1, :]
    n_ls = n_ls / (np.linalg.norm(n_ls) + 1e-12)
    d_ls = -float(np.dot(n_ls, Pc))
    return n_ls, d_ls


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize RANSAC inliers for one frame")
    ap.add_argument("--config", default=str(ROOT / "configs" / "config_example.yaml"))
    ap.add_argument("--frame-id", default="1_20250804_170810_371")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["raw_dir"]).resolve()
    frame_id = args.frame_id

    rgb_path = raw_dir / "rgb" / f"rgb_frame_{frame_id}_png.rf.9edbadbaaecd1b4150bd902cee3b35d3.jpg"
    depth_txt = raw_dir / "depth" / f"depth_data_frame_{frame_id}.txt"

    Zmm = read_depth_txt(depth_txt)

    C = np.array([
        [cfg["cameras"]["depth_camera_matrix"]["fx"], 0.0, cfg["cameras"]["depth_camera_matrix"]["cx"]],
        [0.0, cfg["cameras"]["depth_camera_matrix"]["fy"], cfg["cameras"]["depth_camera_matrix"]["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Convert to centimetres for processDepthImage
    Zcm = Zmm / 10.0
    missing = (Zmm == 0).astype(np.uint8)

    roi = cfg.get("hha", {})
    pc, N, yDir, h, pcRot, NRot = processDepthImage(
        Zcm,
        missing,
        C,
        roi=roi,
        excludeMask=None,
        ransac_thresh_cm=float(roi.get("ransac_thresh_cm", 2.0)),
        min_inlier_ratio=float(roi.get("min_inlier_ratio", 0.1)),
    )

    H, W = pcRot.shape[:2]
    valid = (missing == 0) & np.isfinite(pcRot[:, :, 2])
    roi_mask = build_roi_mask(
        H,
        W,
        float(roi.get("bottom_band_frac", 0.35)),
        float(roi.get("side_band_frac", 0.0)),
        float(roi.get("center_exclude_width_frac", 0.0)),
    )
    valid &= roi_mask

    ys, xs = np.where(valid)
    if ys.size == 0:
        raise SystemExit("No valid points in ROI to fit a plane")
    P = np.stack([pcRot[ys, xs, 0], pcRot[ys, xs, 1], pcRot[ys, xs, 2]], axis=1)

    model = fit_plane_ransac(
        P,
        max_iters=800,
        thresh=float(roi.get("ransac_thresh_cm", 2.0)),
        min_inlier_ratio=float(roi.get("min_inlier_ratio", 0.1)),
    )
    if model is None:
        raise SystemExit("RANSAC failed to find a plane")
    n, d = model
    if float(np.dot(n, np.array([0.0, 1.0, 0.0]))) < 0:
        n = -n
        d = -d

    dist = np.abs(pcRot[:, :, 0] * n[0] + pcRot[:, :, 1] * n[1] + pcRot[:, :, 2] * n[2] + d)
    inlier_mask = (dist <= float(roi.get("ransac_thresh_cm", 2.0))) & valid

    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise FileNotFoundError(f"RGB not found: {rgb_path}")
    rgb = cv2.resize(rgb, (W, H))
    overlay = rgb.copy()
    overlay[inlier_mask] = (0.2 * rgb[inlier_mask] + 0.8 * np.array([0, 255, 0])).astype(np.uint8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "ransac_inliers_overlay.png"), overlay)
    cv2.imwrite(str(out_dir / "ransac_inliers_mask.png"), (inlier_mask.astype(np.uint8) * 255))
    print("saved overlays to", out_dir)


if __name__ == "__main__":
    main()


