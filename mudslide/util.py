# -*- coding: utf-8 -*-
"""Util functions"""

import os
import sys

def find_unique_name(name, always_enumerate = False, ending = ""):
    """
    Given an input basename, checks whether a file with the given name already exists.
    If a file already exists, a suffix is added to make the file unique.

    :param name: initial basename

    :returns: unique basename
    """
    if not always_enumerate and not os.path.exists("{}{}".format(name, ending)):
        return name
    for i in range(sys.maxsize):
        out = "{}-{:d}".format(name, i)
        if not os.path.exists("{}{}".format(out, ending)):
            return out
    raise Exception("No unique name could be made from base {}.".format(name))
    return ""

