#!/usr/bin/env python

from __future__ import print_function


class StackIterator(object):
    """FIFO stack used to iterate over file while holding onto most recent lines"""

    def __init__(self, iterable, stacksize=1, current=-1):
        self.stacksize = stacksize
        self.stack = []
        self.current = current
        self.iterable = iterable
        self.it = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        nx = next(self.it)
        self.add_to_stack(nx)
        self.current += 1
        return nx

    next = __next__  # for python2 compatibility

    def add_to_stack(self, item):
        self.stack.append(item)
        if (len(self.stack) > self.stacksize):
            self.stack.pop(0)

    def top(self):
        return self.stack[-1]
