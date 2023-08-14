#-*- coding: utf-8 -*-
"""Mudslide CLI

This setup is temporary until I fully remove the __main__ and replace it with this.
"""

import sys
import argparse

import mudslide

def mud_main(argv=None, file=sys.stdout) -> None:
    parser = argparse.ArgumentParser(prog="mudslide", description="Mudslide CLI")
    parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {mudslide.__version__}")
    parser.add_argument('-d', '--debug', action='store_true', help="enable debug mode")

    subparsers = parser.add_subparsers(title="commands",
                                       description='valid commands',
                                       dest="subcommand", required=True)

    # subparser for the "collect" command
    mudslide.collect.add_collect_parser(subparsers)

    # subparser for the "surface" command
    mudslide.surface.add_surface_parser(subparsers)

    # subparser for the "make-harmonic" command
    mudslide.turbo_make_harmonic.add_make_harmonic_parser(subparsers)

    args = parser.parse_args(argv)
    args.func(args)

