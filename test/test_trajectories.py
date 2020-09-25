#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import sys
import os
from contextlib import contextmanager
from io import StringIO

import mudslide
import mudslide.__main__

testdir = os.path.dirname(__file__)

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def print_problem(problem, file=sys.stdout):
    what = problem["what"]
    if what == "incorrect data":
        where = problem["where"]
        line1 = problem["a"]
        line2 = problem["b"]
        print("files differ at column: %s" % (", ".join([str(x) for x in where])), file=file)
        print("< %s" % (line1.rstrip()), file=file)
        print("> %s" % (line2.rstrip()), file=file)
    else:
        print(what, file=file)

def compare_line_by_line(f1, f2, typespec, tol=1e-3):
    """Compare two files line by line

    :param f1: file like object to iterate over lines for file 1
    :param f2: file like object to iterate over lines for file 2
    :param types: list of f (float), d (integer), s (string)
    :param tol: floating point tolerance

    :returns: [ problems ]
    """

    def compare(x, y, typekey):
        if typekey == "f":
            return abs(x-y) < tol
        elif typekey == "d":
            return x == y
        elif typekey == "s":
            return x == y
        else:
            raise Exception("only float, integer, and string comparisons allowed right now")

    types = { "f" : float, "d" : int, "s" : str }
    typelist = [ types[x] for x in typespec ]

    failed = False

    problems = []

    for l1, l2 in zip(f1, f2):
        if l1[0] == '#' and l2[0] == '#': continue
        ldata = [ typ(x) for x, typ in zip(l1.split(), typelist) ]
        rdata = [ typ(x) for x, typ in zip(l2.split(), typelist) ]

        lineproblems = []
        for i in range(len(ldata)):
            if not compare(ldata[i], rdata[i], typespec[i]):
                lineproblems.append(i)

        if lineproblems:
            problems.append( { "what" : "incorrect data", "where": lineproblems, "a": l1, "b": l2 } )

    try:
        next(f1) # this should throw
        problems.append( { "what" : "file1 is longer than file2" } )
    except StopIteration:
        pass

    try:
        next(f2) # this should throw
        problems.append( { "what" : "file2 is longer than file1" } )
    except StopIteration:
        pass

    return problems

class TrajectoryTest(object):
    def capture_traj_problems(self, k, tol):
        options = "-s {0:d} -m {1:s} -k {2:f} {2:f} -x {3:f} --dt {4:f} -n {5:d} -z {6:d} -o {7:s} -j {8:d}".format(self.samples, self.model, k, self.x, self.dt, self.n, self.z, self.o, self.j).split()

        outfile = os.path.join(testdir, "checks", "{:s}".format(self.model), "k{:d}.out".format(k))
        with open(outfile, "w") as f:
            mudslide.__main__.main(options, f)

        form = "f" * (6 + 2*self.nstate) + "df"
        reffile = os.path.join(testdir, "ref", "{:s}".format(self.model), "k{:d}.ref".format(k))
        with open(reffile) as ref, open(outfile) as out:
            problems = compare_line_by_line(ref, out, form, tol)
            #for p in problems:
            #    print_problem(p)

        return problems

class TestTSAC(unittest.TestCase, TrajectoryTest):
    """Test Suite for tully simple avoided crossing"""
    model = "simple"
    nstate = 2
    samples = 1
    x = -10
    dt = 5
    n = 1
    z = 200
    o = "single"
    j = 1

    def test_tsac_k8(self):
        probs = self.capture_traj_problems(8, 1e-3)
        self.assertEqual(len(probs), 0)

    def test_tsac_k14(self):
        probs = self.capture_traj_problems(14, 1e-3)
        self.assertEqual(len(probs), 0)

    def test_tsac_k20(self):
        probs = self.capture_traj_problems(20, 1e-3)
        self.assertEqual(len(probs), 0)

if __name__ == '__main__':
    unittest.main()
