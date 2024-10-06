# -*- coding: utf-8 -*-
"""CLI for collecting data from a trajectory"""

from .tracer import YAMLTrace

legend = { "t": "time", "k": "kinetic", "p": "potential", "e": "energy", "a": "active" }
legend_format = { "t": "12.8f", "k": "12.8f", "p": "12.8f", "e": "12.8f", "a": "12d" }

def add_collect_parser(subparsers) -> None:
    """
    Add a parser for the collect command to the subparsers

    :param subparsers: subparsers to add the parser to
    """
    parser = subparsers.add_parser('collect', help="Collect data from a trajectory")
    parser.add_argument('logname', help="name of the trajectory to collect")
    parser.add_argument('-k', '--keys', default="tkpea", help="keys to collect (default: %(default)s)")

    parser.set_defaults(func=collect_wrapper)

def collect(logname: str, keys: str="tkpea") -> None:
    """
    Collect data from a trajectory

    :param logname: name of the trajectory to collect
    :param keys: keys to collect
    """
    log = YAMLTrace(load_main_log=logname)

    with open(logname + ".dat", "w", encoding="utf-8") as f:
        print("#", " ".join([f"{legend[k]:>12s}" for k in keys]), file=f)
        for snap in log:
            print(" " + " ".join([f"{snap[legend[k]]:{legend_format[k]}}" for k in keys]), file=f)

def collect_wrapper(args) -> None:
    """
    Wrapper for collect
    """
    collect(args.logname, args.keys)
