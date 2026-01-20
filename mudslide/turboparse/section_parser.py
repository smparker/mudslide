#!/usr/bin/env python

from __future__ import print_function

import re


class ParseSection(object):
    DEBUG = False
    """Parse a section by looping over attached parsers until tail search is true"""

    def __init__(self, head, tail, multi=False):
        self.head = re.compile(head)
        self.tail = re.compile(tail)
        self.lastsearch = None
        self.multi = multi

    def test(self, reg, line):
        self.lastsearch = reg.search(line)
        return self.lastsearch

    def test_head(self, line):
        self.lastsearch = self.head.search(line)
        return self.lastsearch

    def test_tail(self, line):
        self.lastsearch = self.tail.search(line)
        return self.lastsearch

    def parse_driver(self, liter, out):
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

            if (not first and self.test_tail(liter.top())):
                done = True
            else:
                first = False
                if not advanced_by_parse:  # only advance if not already done by parsing action
                    next(liter)
                advanced = True  # either already advanced or about to be advanced

        return advanced

    def parse(self, liter, out):
        """
        Parse line found at liter.top()

        return: result, advanced
        """
        found = False
        advanced = False
        if (self.test_head(liter.top())):
            if self.DEBUG:
                print("%s (%s) HEAD tested true at:" % (self.name, type(self)))
                print(liter.top())
            dest = self.prepare(out)
            advanced = self.parse_driver(liter, dest)
            found = True
            self.clean(liter, dest)
        elif self.DEBUG:
            print("No match for %s at %s" % (self.name, liter.top().strip()))
        return found, advanced

    def prepare(self, out):
        if self.name == "":
            return out
        else:
            if self.multi:
                if self.name not in out:
                    out[self.name] = []
                out[self.name].append({})
                return out[self.name][-1]
            else:
                out[self.name] = {}
                return out[self.name]

    def clean(self, liter, dest):
        return
