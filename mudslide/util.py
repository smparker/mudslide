# -*- coding: utf-8 -*-
"""Util functions"""

import os
import sys


def find_unique_name(name: str, location="", always_enumerate: bool = False, ending: str = "") -> str:
    """
    Given an input basename, checks whether a file with the given name already exists.
    If a file already exists, a suffix is added to make the file unique.

    :param name: initial basename

    :returns: unique basename
    """
    name_yaml = "{}{}".format(name, ending)
    if not always_enumerate and not os.path.exists(os.path.join(location, name_yaml)):
        return name
    for i in range(sys.maxsize):
        out = "{}-{:d}".format(name, i)
        out_yaml = "{}{}".format(out, ending)
        if not os.path.exists(os.path.join(location, out_yaml)):
            return out
    raise Exception("No unique name could be made from base {}.".format(name))
    return ""
