#!/usr/bin/env python

from __future__ import print_function

from .parse_turbo import parse_turbo

import json
import yaml


def parse():
    import argparse

    ap = argparse.ArgumentParser(
        description="Collects excited state information from an egrad run and prepares as JSON")

    ap.add_argument("file", help="Turbomole output file", type=str)
    ap.add_argument("--format", "-f", choices=["json", "yaml"], default="yaml", help="print format")

    args = ap.parse_args()

    with open(args.file, "r") as f:
        data = parse_turbo(f)

    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    elif args.format == "yaml":
        print(yaml.dump(data))
