#!/usr/bin/env python
"""Iterator wrapper that maintains a lookback stack of recent lines.

Used by the parsing framework to allow parsers to inspect the current
line (via top()) without consuming it, since multiple parsers may need
to test the same line.
"""
from __future__ import annotations

from typing import Iterable, Iterator


class StackIterator:
    """Iterator with a fixed-size lookback stack.

    Wraps any iterable and maintains a FIFO stack of the most recently
    yielded items. The current item is always accessible via top() without
    advancing the iterator.

    Args:
        iterable: The underlying iterable to wrap.
        stacksize: Maximum number of items to retain in the lookback stack.
        current: Initial line counter value (defaults to -1 so first next()
            sets it to 0).
    """

    def __init__(self, iterable: Iterable[str], stacksize: int = 1, current: int = -1) -> None:
        self.stacksize = stacksize
        self.stack: list[str] = []
        self.current = current
        self.iterable = iterable
        self.it: Iterator[str] = self.iterable.__iter__()

    def __iter__(self) -> StackIterator:
        return self

    def __next__(self) -> str:
        nx = next(self.it)
        self.add_to_stack(nx)
        self.current += 1
        return nx

    def add_to_stack(self, item: str) -> None:
        """Add item to the stack, evicting the oldest if at capacity."""
        self.stack.append(item)
        if len(self.stack) > self.stacksize:
            self.stack.pop(0)

    def top(self) -> str:
        """Return the most recently yielded item without advancing."""
        return self.stack[-1]
