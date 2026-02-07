#!/usr/bin/env python
"""Line-level parsers that match individual lines against regex patterns.

LineParser is the base class; SimpleLineParser handles the common case
of extracting named groups with type conversion, and BooleanLineParser
stores True/False based on which of two patterns matches.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from .stack_iterator import StackIterator


class LineParser:
    """Base class to parse a single line and return results. Implementations require a process function"""

    def __init__(self, reg: str) -> None:
        self.reg: re.Pattern[str] = re.compile(reg)

    def parse(self, liter: StackIterator, out: dict[str, Any]) -> tuple[bool, bool]:
        """
        Parse line found at liter.top()

        return: result, advanced
        """
        result = self.reg.search(liter.top())
        if result:
            self.process(result, out)
        return bool(result), False

    def process(self, m: re.Match[str], out: dict[str, Any]) -> None:
        """Process a regex match and store results in out. Must be overridden."""
        raise NotImplementedError("Subclasses must implement process()")


class SimpleLineParser(LineParser):
    """Line parser that extracts regex groups into named fields.

    Matches a line against a regex and stores the captured groups as named
    entries in the output dict, with optional type conversion.

    Args:
        reg: Regex pattern with capturing groups.
        names: List of keys corresponding to each captured group.
        converter: A single callable applied to all groups (e.g., float).
            Mutually exclusive with types.
        types: List of callables, one per group, for individual type
            conversion. If neither converter nor types is given, all values
            are stored as str.
        title: If non-empty, results are stored as a sub-dict under this key
            rather than merged directly into the output dict.
        multi: If True, each match appends to a list under title (requires
            title to be set).
        first_only: If True, only the first match is stored; subsequent
            matches for the same key are ignored.
    """

    def __init__(self,
                 reg: str,
                 names: list[str],
                 converter: Callable[..., Any] | None = None,
                 types: list[Callable[..., Any]] | None = None,
                 title: str = "",
                 multi: bool = False,
                 first_only: bool = False) -> None:
        super().__init__(reg)
        self.names = names
        if converter is None and types is None:
            self.types: list[Callable[..., Any]] = [str] * len(names)
        elif types is not None:
            self.types = types
        elif converter is not None:
            self.types = [converter] * len(names)
        self.title = title
        self.multi = multi
        self.first_only = first_only

        if self.multi and self.title == "":
            raise ValueError("SimpleLineParser in multi mode requires title")

    def process(self, m: re.Match[str], out: dict[str, Any]) -> None:
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
    """Line parser that stores a boolean based on matching one of two patterns.

    Tests the current line against a success regex and a failure regex.
    If the success regex matches, stores True; if the failure regex matches,
    stores False. If neither matches, reports no match.

    Args:
        success_reg: Regex pattern indicating a True result.
        failure_reg: Regex pattern indicating a False result.
        key: Dict key under which the boolean is stored.
        first_only: If True, only the first match is stored.
    """

    def __init__(self, success_reg: str, failure_reg: str, key: str,
                 first_only: bool = False) -> None:
        self.success_reg: re.Pattern[str] = re.compile(success_reg)
        self.failure_reg: re.Pattern[str] = re.compile(failure_reg)
        self.key = key
        self.first_only = first_only

    def parse(self, liter: StackIterator, out: dict[str, Any]) -> tuple[bool, bool]:
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

    def process(self, value: bool, out: dict[str, Any]) -> None:  # type: ignore[override]
        if not (self.first_only and self.key in out):
            out[self.key] = value
