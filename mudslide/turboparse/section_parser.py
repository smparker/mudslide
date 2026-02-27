#!/usr/bin/env python
"""Section-based parser framework for structured text output.

ParseSection matches a region of text delimited by head and tail regex
patterns, then delegates line-by-line parsing to child parsers.
"""
from __future__ import annotations

import re
from typing import Any, Protocol

from .stack_iterator import StackIterator


class ParserProtocol(Protocol):
    """Protocol for objects that can parse lines from a StackIterator."""

    def parse(self, liter: StackIterator, out: dict[str, Any]) -> tuple[bool, bool]: ...


class ParseSection:
    """Parser for a delimited section of output.

    A section is defined by a head regex (start marker) and a tail regex
    (end marker). When parse() detects the head pattern on the current line,
    it enters parse_driver() which iterates through lines, calling each child
    parser on every line until the tail pattern is matched.

    Child parsers (stored in self.parsers) can be LineParser instances for
    single-line matches or nested ParseSection instances for subsections.

    Attributes:
        name: Key under which parsed results are stored in the output dict.
            Empty string means results merge into the parent dict.
        multi: If True, each match appends a new dict to a list under
            self.name, allowing multiple instances of the same section.
        parsers: List of child parsers to apply within this section.
    """
    DEBUG: bool = False
    name: str = ""
    parsers: list[ParserProtocol] = []

    def __init__(self, head: str, tail: str, multi: bool = False) -> None:
        self.head: re.Pattern[str] = re.compile(head)
        self.tail: re.Pattern[str] = re.compile(tail)
        self.lastsearch: re.Match[str] | None = None
        self.multi = multi

    def test(self, reg: re.Pattern[str], line: str) -> re.Match[str] | None:
        """Test line against a regex and store the match result."""
        self.lastsearch = reg.search(line)
        return self.lastsearch

    def test_head(self, line: str) -> re.Match[str] | None:
        """Test if line matches the head (section start) pattern."""
        self.lastsearch = self.head.search(line)
        return self.lastsearch

    def test_tail(self, line: str) -> re.Match[str] | None:
        """Test if line matches the tail (section end) pattern."""
        self.lastsearch = self.tail.search(line)
        return self.lastsearch

    def parse_driver(self, liter: StackIterator, out: dict[str, Any]) -> bool:
        """
        Driver to parse a section

        return: advanced (whether next has been called on liter)
        """
        done = False
        first = True
        advanced = False
        while not done:
            found = False
            advanced_by_parse = False
            for i in self.parsers:
                fnd, adv = i.parse(liter, out)
                found = found or fnd
                advanced_by_parse = advanced_by_parse or adv

            if not first and self.test_tail(liter.top()):
                done = True
            else:
                first = False
                if not advanced_by_parse:  # only advance if not already done by parsing action
                    next(liter)
                advanced = True  # either already advanced or about to be advanced

        return advanced

    def parse(self, liter: StackIterator, out: dict[str, Any]) -> tuple[bool, bool]:
        """
        Parse line found at liter.top()

        return: result, advanced
        """
        found = False
        advanced = False
        if self.test_head(liter.top()):
            if self.DEBUG:
                print(f"{self.name} ({type(self)}) HEAD tested true at:")
                print(liter.top())
            dest = self.prepare(out)
            advanced = self.parse_driver(liter, dest)
            found = True
            self.clean(liter, dest)
        elif self.DEBUG:
            print(f"No match for {self.name} at {liter.top().strip()}")
        return found, advanced

    def prepare(self, out: dict[str, Any]) -> dict[str, Any]:
        """Set up the output dict entry for this section and return the destination.

        If name is empty, returns the parent dict directly (results merge in).
        If multi is True, appends a new dict to a list under self.name.
        Otherwise, creates a single dict under self.name.
        """
        if self.name == "":
            return out
        if self.multi:
            if self.name not in out:
                out[self.name] = []
            out[self.name].append({})
            return out[self.name][-1]
        out[self.name] = {}
        return out[self.name]

    def clean(self, liter: StackIterator, out: dict[str, Any]) -> None:
        """Post-processing hook called after a section is fully parsed.

        Override in subclasses to transform or validate parsed data.
        Default implementation does nothing.
        """
        return
