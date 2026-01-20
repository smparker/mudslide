#!/usr/bin/env python

from __future__ import print_function

import re
import json


class LineParser(object):
    """Base class to parse a single line and return results. Implementations require a process function"""

    def __init__(self, reg):
        self.reg = re.compile(reg)

    def parse(self, liter, out):
        """
        Parse line found at liter.top()

        return: result, advanced
        """
        result = self.reg.search(liter.top())
        if (result):
            self.process(result, out)
        return bool(result), False


class SimpleLineParser(LineParser):
    """Parse a single line and return a list of all matched groups"""

    def __init__(self, reg, names, type=None, types=None, title="", multi=False, first_only=False):
        self.reg = re.compile(reg)
        self.names = names
        if type is None and types is None:
            self.types = [str] * len(names)
        elif types is not None:
            self.types = types
        elif type is not None:
            self.types = [type] * len(names)
        self.title = title
        self.multi = multi
        self.first_only = first_only

        if self.multi and self.title == "":
            raise Exception("SimpleLineParser in multi mode requires title")

    def process(self, m, out):
        data = {n: self.types[i](m.group(i + 1)) for i, n in enumerate(self.names)}
        if not self.multi:
            if self.title != "":
                if not (self.first_only and self.title in out):
                    out[self.title] = data
            else:
                out.update(data)
        else:
            if self.title not in out:
                out[self.title] = []
            out[self.title].append(data)
