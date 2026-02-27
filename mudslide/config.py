# -*- coding: utf-8 -*-
"""User configuration file support for mudslide.

Loads settings from ``~/.config/mudslide/config.yaml`` (or
``$XDG_CONFIG_HOME/mudslide/config.yaml`` when that variable is set).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


def _config_path() -> Path:
    """Return the path to the user config file."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "mudslide" / "config.yaml"


_config_cache: Optional[dict] = None


def load_config() -> dict:
    """Load and cache the user config.  Returns ``{}`` if file is missing."""
    global _config_cache
    if _config_cache is None:
        path = _config_path()
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                _config_cache = yaml.safe_load(f) or {}
        else:
            _config_cache = {}
    return _config_cache


def get_config(key: str, default: Any = None) -> Any:
    """Dotted-key lookup into the config dict.

    Example::

        get_config("turbomole.command_prefix")
    """
    obj: Any = load_config()
    for part in key.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return default
    return obj if obj is not None else default
