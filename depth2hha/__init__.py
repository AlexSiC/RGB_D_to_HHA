from __future__ import annotations

"""Local adapter to third_party Depth2HHA-python.

Exposes a simple `convert(depth_map_m: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray`
API expected by `pipeline.hha_service.HHAService`.
"""

from pathlib import Path
import sys
from typing import Any

import numpy as np


def _import_backend() -> Any:
    root = Path(__file__).resolve().parents[1]
    tp_path = root / "third_party" / "Depth2HHA-python"
    if str(tp_path) not in sys.path:
        sys.path.insert(0, str(tp_path))
    # Import getHHA from third_party package
    import getHHA  # type: ignore

    return getHHA


def convert(
    depth_map_m: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    roi: dict | None = None,
    exclude_mask: np.ndarray | None = None,
    ransac_thresh_cm: float = 2.0,
    min_inlier_ratio: float = 0.1,
) -> np.ndarray:
    backend = _import_backend()
    # RD (raw depth) can be same as D when not available
    D = depth_map_m.astype(np.float32)
    RD = D
    C = camera_matrix.astype(np.float32)
    # If backend exposes processDepthImage args via getHHA only, we patch through by setting globals
    try:
        from utils import rgbd_util  # type: ignore
        # Monkey-patch optional parameters through module attributes for this call scope
        # but since getHHA calls processDepthImage directly, we temporarily wrap it.
        original_pdi = rgbd_util.processDepthImage

        def _pdi_wrapper(z, missingMask, C_):
            return original_pdi(
                z,
                missingMask,
                C_,
                roi=roi,
                excludeMask=exclude_mask,
                ransac_thresh_cm=ransac_thresh_cm,
                min_inlier_ratio=min_inlier_ratio,
            )

        # Set deterministic RANSAC seed if provided via config/env
        import os
        # RANSAC seed
        if 'HHA_RANSAC_SEED' not in os.environ and roi is not None:
            try:
                os.environ['HHA_RANSAC_SEED'] = str(int(roi.get('ransac_seed')))  # type: ignore[arg-type]
            except Exception:
                pass
        # Gravity init
        if 'HHA_GRAVITY_INIT' not in os.environ and roi is not None:
            try:
                os.environ['HHA_GRAVITY_INIT'] = str(roi.get('gravity_init'))  # type: ignore[arg-type]
            except Exception:
                pass

        rgbd_util.processDepthImage = _pdi_wrapper  # type: ignore[attr-defined]
        try:
            hha_bgr_u8 = backend.getHHA(C, D, RD)
        finally:
            rgbd_util.processDepthImage = original_pdi  # type: ignore[attr-defined]
    except Exception:
        # Fallback: call without custom ROI if wrapper injection fails
        hha_bgr_u8 = backend.getHHA(C, D, RD)
    # The backend returns HHA as 3-channel uint8 BGR suitable for saving/displaying.
    # Our pipeline expects a float32 array; keep uint8 here and let caller scale if needed.
    return hha_bgr_u8


