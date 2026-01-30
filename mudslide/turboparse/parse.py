#!/usr/bin/env python

import argparse
import json

import yaml

from .parse_turbo import parse_turbo
from ..version import get_version_info


def _is_numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_tree(value):
    """Check if value is a numeric scalar, or a (nested) list of numerics."""
    if _is_numeric(value):
        return True
    if isinstance(value, list):
        return all(_is_numeric_tree(item) for item in value)
    return False


class _CompactDumper(yaml.Dumper):
    """YAML dumper that uses flow style for short lists and numeric lists/trees."""
    SHORT_LIST_THRESHOLD = 5


_orig_represent_list = yaml.Dumper.represent_list


def _is_scalar(value):
    return isinstance(value, (int, float, str, bool)) or value is None


def _represent_list(dumper, data):
    node = _orig_represent_list(dumper, data)
    if _is_numeric_tree(data):
        node.flow_style = True
    elif len(data) <= _CompactDumper.SHORT_LIST_THRESHOLD and all(_is_scalar(v) for v in data):
        node.flow_style = True
    return node


_CompactDumper.add_representer(list, _represent_list)


def parse():
    ap = argparse.ArgumentParser(
        description="Collects excited state information from an egrad run and prepares as JSON",
        epilog=get_version_info(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-v', '--version', action='version', version=get_version_info())

    ap.add_argument("file", help="Turbomole output file", type=str)
    ap.add_argument("--format", "-f", choices=["json", "yaml"], default="yaml", help="print format")

    args = ap.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = parse_turbo(f)

    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    elif args.format == "yaml":
        print(yaml.dump(data, Dumper=_CompactDumper, default_flow_style=False))
