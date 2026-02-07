# -*- coding: utf-8 -*-
"""YAML formatting utilities for compact output."""

from __future__ import annotations

from typing import Any

import yaml


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_tree(value: object) -> bool:
    """Check if value is a numeric scalar, or a (nested) list of numerics."""
    if _is_numeric(value):
        return True
    if isinstance(value, list):
        return all(_is_numeric_tree(item) for item in value)
    return False


def _is_scalar(value: object) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


SHORT_LIST_THRESHOLD = 5


def _compact_represent_list(
        orig_represent_list: Any) -> Any:  # type: ignore[no-untyped-def]
    """Create a compact list representer wrapping the given base representer."""

    def representer(dumper, data):  # type: ignore[no-untyped-def]
        node = orig_represent_list(dumper, data)
        if _is_numeric_tree(data):
            node.flow_style = True
        elif len(data) <= SHORT_LIST_THRESHOLD and all(
                _is_scalar(v) for v in data):
            node.flow_style = True
        return node

    return representer


class CompactDumper(yaml.Dumper):
    """YAML dumper that uses flow style for short lists and numeric lists/trees."""


CompactDumper.add_representer(
    list, _compact_represent_list(yaml.Dumper.represent_list))


class CompactSafeDumper(yaml.SafeDumper):
    """YAML safe dumper that uses flow style for short lists and numeric lists/trees."""


CompactSafeDumper.add_representer(
    list, _compact_represent_list(yaml.SafeDumper.represent_list))
