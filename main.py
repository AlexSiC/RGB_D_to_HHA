from __future__ import annotations

import argparse

from pipeline.config_service import ConfigService
from pipeline.file_service import FileService
from pipeline.inpainting_service import InpaintingService
from pipeline.annotation_service import AnnotationService
from pipeline.augmentation_service import AugmentationService
from pipeline.hha_service import HHAService
from pipeline.pipeline_orchestrator import PipelineOrchestrator
from typing import Any
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB/Depth -> HHA data preparation pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    # Generic overrides: repeatable --set path=value (e.g., --set augmentation.seed=41)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config field using dotted path, e.g. augmentation.seed=41; can be repeated",
    )
    # Convenience flags
    parser.add_argument("--augmentation-seed", type=int, help="Override augmentation.seed")
    parser.add_argument(
        "--augmentation-enabled",
        type=str,
        choices=["true", "false"],
        help="Override augmentation.enabled",
    )
    parser.add_argument("--inpainting-method", type=str, help="Override inpainting.method")
    # New toggles for exports
    parser.add_argument("--train-export", type=str, choices=["true", "false"], help="Override outputs.enable_train_export")
    parser.add_argument("--processed-export", type=str, choices=["true", "false"], help="Override outputs.enable_processed_export")
    return parser.parse_args()


def _coerce_value(value_str: str) -> Any:
    # Try json for true/false/null/numbers/arrays, fall back to raw string
    try:
        return json.loads(value_str)
    except Exception:
        return value_str


def _apply_override_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for key in parts[:-1]:
        cur = getattr(cur, key)
    setattr(cur, parts[-1], value)


def main() -> None:
    args = parse_args()

    cfg_service = ConfigService()
    config = cfg_service.load_config(args.config)

    # Apply convenience flags
    if args.augmentation_seed is not None:
        config.augmentation.seed = int(args.augmentation_seed)
    if args.augmentation_enabled is not None:
        config.augmentation.enabled = args.augmentation_enabled.lower() == "true"
    if args.inpainting_method is not None:
        config.inpainting.method = str(args.inpainting_method)
    if args.train_export is not None:
        config.outputs.enable_train_export = args.train_export.lower() == "true"
    if args.processed_export is not None:
        config.outputs.enable_processed_export = args.processed_export.lower() == "true"

    # Apply generic overrides
    for ov in args.overrides or []:
        if "=" not in ov:
            raise SystemExit(f"Invalid --set override '{ov}', expected path=value")
        path, value_str = ov.split("=", 1)
        value = _coerce_value(value_str)
        _apply_override_path(config, path.strip(), value)

    orchestrator = PipelineOrchestrator(
        config=config,
        file_service=FileService(),
        inpainting_service=InpaintingService(),
        annotation_service=AnnotationService(),
        augmentation_service=AugmentationService(),
        hha_service=HHAService(),
    )
    orchestrator.run_full_pipeline()


if __name__ == "__main__":
    main()



