from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config_service import ConfigService
from pipeline.file_service import FileService
from pipeline.inpainting_service import InpaintingService
from pipeline.annotation_service import AnnotationService
from pipeline.augmentation_service import AugmentationService
from pipeline.hha_service import HHAService
from pipeline.pipeline_orchestrator import PipelineOrchestrator


def main() -> None:
    ap = argparse.ArgumentParser(description="Run pipeline in a specified order")
    ap.add_argument("--config", required=True)
    ap.add_argument("--put-last", dest="put_last", help="Frame ID to process last")
    ap.add_argument("--only", dest="only", help="Process only this frame ID")
    args = ap.parse_args()

    cfg = ConfigService().load_config(args.config)
    orchestrator = PipelineOrchestrator(
        config=cfg,
        file_service=FileService(),
        inpainting_service=InpaintingService(),
        annotation_service=AnnotationService(),
        augmentation_service=AugmentationService(),
        hha_service=HHAService(),
    )

    frames = orchestrator.file_service.discover_frames(cfg.paths.raw_dir)
    if args.only:
        frames = [f for f in frames if f.base_name == args.only]
    elif args.put_last:
        last = [f for f in frames if f.base_name == args.put_last]
        rest = [f for f in frames if f.base_name != args.put_last]
        frames = rest + last

    for f in frames:
        orchestrator.process_single_frame(f)


if __name__ == "__main__":
    main()


