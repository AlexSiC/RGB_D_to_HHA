from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from .data_models import PipelineConfig


class ConfigService:
    """Service responsible for loading and providing pipeline configuration.

    Loads YAML configuration and validates it against `PipelineConfig`.
    """

    def __init__(self) -> None:
        self._config: Optional[PipelineConfig] = None

    def load_config(self, path: str) -> PipelineConfig:
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Validated `PipelineConfig` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValidationError: If YAML content doesn't match `PipelineConfig` schema.
            yaml.YAMLError: If YAML is malformed.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        try:
            config = PipelineConfig(**(data or {}))
        except ValidationError:
            # Re-raise to keep original error details for the caller
            raise

        self._config = config
        return config

    def get_config(self) -> PipelineConfig:
        """Return previously loaded configuration or raise if not loaded."""
        if self._config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_config() first.")
        return self._config



