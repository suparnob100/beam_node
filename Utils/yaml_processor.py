"""Thin wrappers for YAML configuration file I/O."""
from __future__ import annotations

from typing import Any

import yaml


def load_config(config_file: str) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_config(config_file: str, config: dict[str, Any]) -> None:
    """Write *config* to a YAML file."""
    with open(config_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
