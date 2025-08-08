from __future__ import annotations

import argparse

from pipeline.config_service import ConfigService
from pipeline.file_service import FileService
from pipeline.inpainting_service import InpaintingService
from pipeline.annotation_service import AnnotationService
from pipeline.augmentation_service import AugmentationService
from pipeline.hha_service import HHAService
from pipeline.pipeline_orchestrator import PipelineOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB/Depth -> HHA data preparation pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_service = ConfigService()
    config = cfg_service.load_config(args.config)

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



