#-*- coding: utf-8 -*-
"""Mudslide CLI

This setup is temporary until I fully remove the __main__ and replace it with this.
"""

from __future__ import annotations

import sys
import argparse
from typing import Any

import mudslide
from .version import get_version_info


def mud_main(argv: list[str] | None = None,
             file: Any = sys.stdout) -> None:
    """Mudslide CLI
    """
    parser = argparse.ArgumentParser(
        prog="mudslide",
        description="Mudslide CLI",
        epilog=get_version_info(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=get_version_info())
    parser.add_argument('-d',
                        '--debug',
                        action='store_true',
                        help="enable debug mode")

    subparsers = parser.add_subparsers(title="commands",
                                       description='valid commands',
                                       dest="subcommand",
                                       required=True)

    # subparser for the "collect" command
    mudslide.collect.add_collect_parser(subparsers)

    # subparser for the "surface" command
    mudslide.surface.add_surface_parser(subparsers)

    # subparser for the "make-harmonic" command
    mudslide.turbo_make_harmonic.add_make_harmonic_parser(subparsers)

    args = parser.parse_args(argv)
    args.func(args)
