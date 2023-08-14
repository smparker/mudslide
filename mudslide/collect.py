# -*- coding: utf-8 -*-
"""CLI for collecting data from a trajectory"""

import sys
import argparse

import numpy as np

import mudslide

from typing import Any, List
from .typing import ArrayLike

legend = { "t": "time", "k": "kinetic", "p": "potential", "e": "energy", "a": "active" }
legend_format = { "t": "12.8f", "k": "12.8f", "p": "12.8f", "e": "12.8f", "a": "12d" }

def add_collect_parser(subparsers) -> None:
    parser = subparsers.add_parser('collect', help="Collect data from a trajectory")
    parser.add_argument('logname', help="name of the trajectory to collect")
    parser.add_argument('-k', '--keys', default="tkpea", help="keys to collect (default: %(default)s)")

    parser.set_defaults(func=collect_wrapper)

def collect(logname: str, keys: str="tkpea") -> None:
    log = mudslide.YAMLTrace(load_main_log=logname)

    with open(logname + ".dat", "w") as f:
        print("#", " ".join([f"{legend[k]:>12s}" for k in keys]), file=f)
        for snap in log:
            print(" " + " ".join([f"{snap[legend[k]]:{legend_format[k]}}" for k in keys]), file=f)

def collect_wrapper(args) -> None:
    collect(args.logname, args.keys)
