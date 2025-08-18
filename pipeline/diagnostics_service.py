from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class DiagnosticsService:
    """Compute plane diagnostics for depth -> HHA conversion.

    Uses third_party Depth2HHA-python to obtain rotated point cloud, then fits
    a plane inside the configured ROI to derive inliers and debugging data.
    """

    def __init__(self) -> None:
        # Ensure third_party on sys.path
        root = Path(__file__).resolve().parents[1]
        tp_path = root / "third_party" / "Depth2HHA-python"
        if str(tp_path) not in sys.path:
            sys.path.insert(0, str(tp_path))
        # Import after path setup
        from utils import rgbd_util as _rgbd_util  # type: ignore

        self._rgbd_util = _rgbd_util

    @staticmethod
    def _build_roi_mask(h: int, w: int, *, bottom_band_frac: float, side_band_frac: float, center_exclude_width_frac: float) -> np.ndarray:
        mask = np.zeros((h, w), dtype=bool)
        roi_start = int((1.0 - bottom_band_frac) * h)
        mask[roi_start:, :] = True
        if side_band_frac > 0.0:
            sw = max(1, int(side_band_frac * w))
            mask[:, :sw] = True
            mask[:, w - sw:] = True
        if center_exclude_width_frac > 0.0:
            hw = int(0.5 * center_exclude_width_frac * w)
            cx = w // 2
            mask[roi_start:, max(0, cx - hw): min(w, cx + hw)] = False
        return mask

    @staticmethod
    def _fit_plane_ransac(points: np.ndarray, *, max_iters: int = 800, thresh_cm: float = 2.0, min_inlier_ratio: float = 0.1, rng: Optional[np.random.Generator] = None) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
        if rng is None:
            rng = np.random.default_rng()
        n_pts = points.shape[0]
        if n_pts < 3:
            return None
        best_inliers = None
        best_model = None
        for _ in range(max_iters):
            idx = rng.choice(n_pts, size=3, replace=False)
            p1, p2, p3 = points[idx]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-6:
                continue
            n = n / norm_n
            d = -float(np.dot(n, p1))
            dist = np.abs(points @ n + d)
            inliers = dist <= thresh_cm
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
                best_model = (n, d)
        if best_inliers is None or best_inliers.sum() < max(int(min_inlier_ratio * n_pts), 50):
            return None
        P = points[best_inliers]
        Pc = P.mean(axis=0)
        _, _, Vt = np.linalg.svd(P - Pc, full_matrices=False)
        n_ls = Vt[-1, :]
        n_ls = n_ls / (np.linalg.norm(n_ls) + 1e-12)
        d_ls = -float(np.dot(n_ls, Pc))
        dist_full = np.abs(points @ n_ls + d_ls)
        inliers_refined = dist_full <= thresh_cm
        return n_ls, d_ls, inliers_refined

    def compute(self, depth_m: np.ndarray, camera_matrix: np.ndarray, *, roi: Dict[str, float], exclude_mask: Optional[np.ndarray], ransac_thresh_cm: float, min_inlier_ratio: float, fixed_height_cm_max: float = 20.0) -> Dict[str, object]:
        # Prepare inputs for third_party (expects centimetres)
        Zcm = (depth_m * 100.0).astype(np.float32)
        miss = (Zcm <= 0) | (np.isnan(Zcm))
        if exclude_mask is not None:
            miss = miss | exclude_mask.astype(bool)

        pc, N, yDir, h_cm, pcRot, NRot = self._rgbd_util.processDepthImage(
            Zcm,
            miss.astype(np.uint8),
            camera_matrix.astype(np.float32),
            roi=roi,
            excludeMask=exclude_mask.astype(bool) if exclude_mask is not None else None,
            ransac_thresh_cm=float(ransac_thresh_cm),
            min_inlier_ratio=float(min_inlier_ratio),
        )

        H, W = pcRot.shape[:2]
        roi_mask = self._build_roi_mask(
            H,
            W,
            bottom_band_frac=float(roi.get("bottom_band_frac", 0.35)),
            side_band_frac=float(roi.get("side_band_frac", 0.0)),
            center_exclude_width_frac=float(roi.get("center_exclude_width_frac", 0.0)),
        )
        valid = (~miss.astype(bool)) & np.isfinite(pcRot[:, :, 2]) & roi_mask
        ys, xs = np.where(valid)
        if ys.size > 0:
            P = np.stack([pcRot[ys, xs, 0], pcRot[ys, xs, 1], pcRot[ys, xs, 2]], axis=1)
        else:
            P = np.empty((0, 3), dtype=np.float64)

        n_d_inliers = self._fit_plane_ransac(
            P,
            max_iters=800,
            thresh_cm=float(ransac_thresh_cm),
            min_inlier_ratio=float(min_inlier_ratio),
        )

        plane_info: Dict[str, object]
        inlier_mask_full = np.zeros((H, W), dtype=bool)
        if n_d_inliers is not None:
            n, d, inliers = n_d_inliers
            # Orient normal upward (+Y)
            if float(np.dot(n, np.array([0.0, 1.0, 0.0]))) < 0:
                n = -n
                d = -d
            # Fill inlier mask at selected pixels
            inlier_mask_full[ys, xs] = inliers
            angle_to_up_deg = float(np.degrees(np.arccos(np.clip(np.dot(n, np.array([0.0, 1.0, 0.0])) / (np.linalg.norm(n) + 1e-12), -1.0, 1.0))))
            plane_info = {
                "normal": n.tolist(),
                "d": float(d),
                "angle_to_up_deg": angle_to_up_deg,
                "inliers": int(inliers.sum()),
                "candidates": int(P.shape[0]),
                "inlier_ratio": float(inliers.sum() / max(1, P.shape[0])),
                "thresh_cm": float(ransac_thresh_cm),
            }
        else:
            plane_info = {
                "normal": None,
                "d": None,
                "angle_to_up_deg": None,
                "inliers": 0,
                "candidates": int(P.shape[0]),
                "inlier_ratio": 0.0,
                "thresh_cm": float(ransac_thresh_cm),
            }

        # Fixed-range height visualization (0..fixed_height_cm_max)
        h_fixed = np.clip(h_cm, 0.0, float(fixed_height_cm_max))
        h_fixed_u8 = (h_fixed * (255.0 / max(1e-6, float(fixed_height_cm_max)))).astype(np.uint8)

        return {
            "roi_mask": roi_mask.astype(np.uint8) * 255,
            "inlier_mask": inlier_mask_full.astype(np.uint8) * 255,
            "plane": plane_info,
            "height_fixed_u8": h_fixed_u8,
        }


