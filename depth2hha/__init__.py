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


def convert(depth_map_m: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    backend = _import_backend()
    # RD (raw depth) can be same as D when not available
    D = depth_map_m.astype(np.float32)
    RD = D
    C = camera_matrix.astype(np.float32)
    hha_bgr_u8 = backend.getHHA(C, D, RD)
    # The backend returns HHA as 3-channel uint8 BGR suitable for saving/displaying.
    # Our pipeline expects a float32 array; keep uint8 here and let caller scale if needed.
    return hha_bgr_u8


