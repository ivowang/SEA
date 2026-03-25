"""OmegaConf-based configuration loading and merging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config file and apply CLI overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: List of "key=value" strings for CLI overrides.

    Returns:
        Merged DictConfig.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    logger.info("Loaded config from %s", config_path)
    return cfg


def merge_configs(*configs: DictConfig | dict) -> DictConfig:
    """Merge multiple configs (later ones override earlier ones)."""
    result = OmegaConf.create({})
    for cfg in configs:
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        result = OmegaConf.merge(result, cfg)
    return result


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to plain dict."""
    return OmegaConf.to_container(cfg, resolve=True)
