#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import sys
import os

import mudslide
import mudslide.__main__
import mudslide.surface

testdir = os.path.dirname(__file__)

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
    samples = 1
    method = "fssh"
    x = -10
    dt = 5
    n = 1
    seed = 200
    o = "single"
    j = 1
    electronic = "exp"

    def capture_traj_problems(self, k, tol, extra_options = []):
        options = "-s {0:d} -m {1:s} -k {2:f} {2:f} -x {3:f} --dt {4:f} -n {5:d} -z {6:d} -o {7:s} -j {8:d} -a {9:s} --electronic {10:s}".format(self.samples, self.model, k, self.x, self.dt, self.n, self.seed, self.o, self.j, self.method, self.electronic).split()
        options += extra_options

        checkdir = os.path.join(testdir, "checks", self.method)
        os.makedirs(checkdir, exist_ok=True)
        outfile = os.path.join(checkdir, "{:s}_k{:d}.out".format(self.model, k))
        with open(outfile, "w") as f:
            mudslide.__main__.main(options, f)

        if self.o == "single":
            form = "f" * (6 + 2*self.nstate) + "df"
        elif self.o == "averaged":
            form = "ffff"
        reffile = os.path.join(testdir, "ref", self.method, "{:s}_k{:d}.ref".format(self.model, k))
        with open(reffile) as ref, open(outfile) as out:
            problems = compare_line_by_line(ref, out, form, tol)
            for p in problems:
                print_problem(p)

        return problems

class TestTSAC(unittest.TestCase, TrajectoryTest):
    """Test Suite for tully simple avoided crossing"""
    model = "simple"
    nstate = 2

    def test_tsac(self):
        for k in [8, 14, 20]:
            with self.subTest(k=k):
                probs = self.capture_traj_problems(k, 1e-3)
                self.assertEqual(len(probs), 0)

class TestDual(unittest.TestCase, TrajectoryTest):
    """Test Suite for tully dual avoided crossing"""
    model = "dual"
    nstate = 2

    def test_dual(self):
        for k in [20, 50, 100]:
            with self.subTest(k=k):
                probs = self.capture_traj_problems(k, 1e-3)
                self.assertEqual(len(probs), 0)

class TestExtended(unittest.TestCase, TrajectoryTest):
    """Test Suite for tully dual avoided crossing"""
    model = "extended"
    nstate = 2

    def test_extended(self):
        for k in [10, 15, 20]:
            with self.subTest(k=k):
                probs = self.capture_traj_problems(k, 1e-3)
                self.assertEqual(len(probs), 0)

class TestTSACc(unittest.TestCase, TrajectoryTest):
    """Test Suite for tully simple avoided crossing with cumulative hopping"""
    model = "simple"
    nstate = 2
    seed = 756396545
    method = "cumulative-sh"
    electronic = "linear-rk4"

    def test_tsac_c(self):
        for k in [10, 20]:
            with self.subTest(k=k):
                probs = self.capture_traj_problems(k, 1e-3)
                self.assertEqual(len(probs), 0)

class TestEhrenfest(unittest.TestCase, TrajectoryTest):
    """Test suite for ehrenfest trajectory"""
    model = "simple"
    nstate = 2
    method = "ehrenfest"

    def test_ehrenfest(self):
        k = 15
        probs = self.capture_traj_problems(k, 1e-3)
        self.assertEqual(len(probs), 0)

class TestES(unittest.TestCase, TrajectoryTest):
    """Test Suite for tully simple avoided crossing with cumulative hopping"""
    model = "simple"
    nstate = 2
    dt = 20
    seed = 84329
    method = "even-sampling"
    o = "averaged"

    def test_es_tsac(self):
        for k in [10, 20]:
            with self.subTest(k=k):
                probs = self.capture_traj_problems(k, 1e-3, extra_options=["--sample-stack", "5"])
                self.assertEqual(len(probs), 0)

class TestSurface(unittest.TestCase):
    """Test Suite for surface writer"""

    def test_surface(self):
        tol = 1e-3
        for m in [ "simple", "extended", "dual", "super", "shin-metiu", "modelx", "models" ]:
            with self.subTest(m=m):
                options = "-m {:s} -r -11 11 -n 200".format(m).split()
                checkdir = os.path.join(testdir, "checks", "surface")
                os.makedirs(checkdir, exist_ok=True)
                outfile = os.path.join(checkdir, "{:s}.out".format(m))
                with open(outfile, "w") as f:
                    mudslide.surface.main(options, f)

                form = "f" * (8 if m in ["simple", "extended", "dual"] else 13)
                reffile = os.path.join(testdir, "ref", "surface", "{:s}.ref".format(m))
                with open(reffile) as ref, open(outfile) as out:
                    problems = compare_line_by_line(ref, out, form, tol)
                    for p in problems:
                        print_problem(p)
                self.assertEqual(len(problems), 0)

if __name__ == '__main__':
    unittest.main()
