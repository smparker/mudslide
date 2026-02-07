#!/usr/bin/env python

import re


class LineParser:
    """Base class to parse a single line and return results. Implementations require a process function"""

    def __init__(self, reg):
        self.reg = re.compile(reg)

    def parse(self, liter, out):
        """
        Parse line found at liter.top()

        return: result, advanced
        """
        result = self.reg.search(liter.top())
        if result:
            self.process(result, out)
        return bool(result), False

    def process(self, m, out):
        raise NotImplementedError("Subclasses must implement process()")


class SimpleLineParser(LineParser):
    """Parse a single line and return a list of all matched groups"""

    def __init__(self,
                 reg,
                 names,
                 converter=None,
                 types=None,
                 title="",
                 multi=False,
                 first_only=False):
        super().__init__(reg)
        self.names = names
        if converter is None and types is None:
            self.types = [str] * len(names)
        elif types is not None:
            self.types = types
        elif converter is not None:
            self.types = [converter] * len(names)
        self.title = title
        self.multi = multi
        self.first_only = first_only

        if self.multi and self.title == "":
            raise ValueError("SimpleLineParser in multi mode requires title")

    def process(self, m, out):
        data = {
            n: self.types[i](m.group(i + 1)) for i, n in enumerate(self.names)
        }
        if not self.multi:
            if self.title != "":
                if not (self.first_only and self.title in out):
                    out[self.title] = data
            else:
                # For first_only without title, check if any of the keys already exist
                if self.first_only and any(n in out for n in self.names):
                    return
                out.update(data)
        else:
            if self.title not in out:
                out[self.title] = []
            out[self.title].append(data)


class BooleanLineParser(LineParser):
    """Parse a line and store a boolean based on success/failure regex matches."""

    def __init__(self, success_reg, failure_reg, key, first_only=False):
        self.success_reg = re.compile(success_reg)
        self.failure_reg = re.compile(failure_reg)
        self.key = key
        self.first_only = first_only

    def parse(self, liter, out):
        """
        Parse line found at liter.top()

        return: result, advanced
        """
        line = liter.top()
        success = self.success_reg.search(line)
        if success:
            self.process(True, out)
            return True, False

        failure = self.failure_reg.search(line)
        if failure:
            self.process(False, out)
            return True, False

        return False, False

    def process(self, value, out):
        if not (self.first_only and self.key in out):
            out[self.key] = value
