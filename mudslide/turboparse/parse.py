#!/usr/bin/env python

from __future__ import print_function

from .parse_turbo import parse_turbo
from ..version import get_version_info

import argparse
import json
import yaml


def parse():
    ap = argparse.ArgumentParser(
        description="Collects excited state information from an egrad run and prepares as JSON",
        epilog=get_version_info(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-v', '--version', action='version', version=get_version_info())

    ap.add_argument("file", help="Turbomole output file", type=str)
    ap.add_argument("--format", "-f", choices=["json", "yaml"], default="yaml", help="print format")

    args = ap.parse_args()

    with open(args.file, "r") as f:
        data = parse_turbo(f)

    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    elif args.format == "yaml":
        print(yaml.dump(data))
