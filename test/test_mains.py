#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os, shutil

import pytest

import mudslide
import mudslide.__main__
import mudslide.surface

testdir = os.path.dirname(__file__)


def clean_directory(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)


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
            return abs(x - y) < tol
        elif typekey == "d":
            return x == y
        elif typekey == "s":
            return x == y
        else:
            raise Exception("only float, integer, and string comparisons allowed right now")

    types = {"f": float, "d": int, "s": str}
    typelist = [types[x] for x in typespec]

    problems = []

    for l1, l2 in zip(f1, f2):
        if l1[0] == '#' and l2[0] == '#':
            continue
        ldata = [typ(x) for x, typ in zip(l1.split(), typelist)]
        rdata = [typ(x) for x, typ in zip(l2.split(), typelist)]

        lineproblems = []
        for i in range(len(ldata)):
            if not compare(ldata[i], rdata[i], typespec[i]):
                lineproblems.append(i)

        if lineproblems:
            problems.append({"what": "incorrect data", "where": lineproblems, "a": l1, "b": l2})

    try:
        next(f1)  # this should throw
        problems.append({"what": "file1 is longer than file2"})
    except StopIteration:
        pass

    try:
        next(f2)  # this should throw
        problems.append({"what": "file2 is longer than file1"})
    except StopIteration:
        pass

    return problems


def capture_traj_problems(model, nstate, k, tol, method="fssh", x=-10, dt=5,
                          n=1, seed=200, o="single", j=1, electronic="exp",
                          log="memory", extra_options=None):
    if extra_options is None:
        extra_options = []
    options = ("-s {0:d} -m {1:s} -k {2:f} {2:f} -x {3:f} --dt {4:f} -n {5:d} "
               "-z {6:d} -o {7:s} -j {8:d} -a {9:s} --electronic {10:s} "
               "--log {11:s}").format(
        1, model, k, x, dt, n, seed, o, j, method, electronic, log).split()
    options += extra_options

    checkdir = os.path.join(testdir, "checks", method)
    options += "--logdir {}".format(checkdir).split()

    os.makedirs(checkdir, exist_ok=True)
    outfile = os.path.join(checkdir, f"{model:s}_k{k:d}.out")
    with open(outfile, "w") as f:
        mudslide.__main__.main(options, f)

    if o == "single":
        form = "f" * (6 + 2 * nstate) + "df"
    elif o == "averaged":
        form = "ffff"
    reffile = os.path.join(testdir, "ref", method, "{:s}_k{:d}.ref".format(model, k))
    with open(reffile) as ref, open(outfile) as out:
        problems = compare_line_by_line(ref, out, form, tol)
        for p in problems:
            print_problem(p)

    return problems


# -- Tully Simple Avoided Crossing (FSSH) --

@pytest.mark.parametrize("k", [8, 14, 20])
def test_tsac(k):
    """Tully Simple Avoided Crossing"""
    probs = capture_traj_problems("simple", 2, k, 1e-3)
    assert len(probs) == 0


# -- Tully Dual Avoided Crossing (FSSH) --

@pytest.mark.parametrize("k", [20, 50, 100])
def test_dual(k):
    """Tully Dual Avoided Crossing"""
    probs = capture_traj_problems("dual", 2, k, 1e-3)
    assert len(probs) == 0


# -- Tully Extended Coupling (FSSH) --

@pytest.mark.parametrize("k", [10, 15, 20])
def test_extended(k):
    """Tully Extended Coupling"""
    probs = capture_traj_problems("extended", 2, k, 1e-3)
    assert len(probs) == 0


# -- Tully Simple Avoided Crossing (cumulative-sh with linear-rk4) --

@pytest.mark.parametrize("k", [10, 20])
def test_tsac_cumulative(k):
    """Tully Simple Avoided Crossing (FSSH-c)"""
    probs = capture_traj_problems("simple", 2, k, 1e-3,
                                  method="cumulative-sh", seed=756396545,
                                  electronic="linear-rk4")
    assert len(probs) == 0


# -- Ehrenfest --

def test_ehrenfest():
    """Tully Simple Avoided Crossing (Ehrenfest)"""
    probs = capture_traj_problems("simple", 2, 15, 1e-3, method="ehrenfest")
    assert len(probs) == 0


# -- A-FSSH --

def test_afssh():
    """Tully Dual Avoided Crossing (A-FSSH)"""
    probs = capture_traj_problems("dual", 2, 14, 1e-3, method="afssh", seed=78341)
    assert len(probs) == 0


# -- Even Sampling --

@pytest.mark.parametrize("k", [10, 20])
def test_even_sampling(k):
    """Even Sampling"""
    probs = capture_traj_problems("simple", 2, k, 1e-3,
                                  method="even-sampling", dt=20, seed=84329,
                                  o="averaged", log="yaml",
                                  extra_options=["--sample-stack", "5"])
    assert len(probs) == 0


# -- Surface Writer --

@pytest.mark.parametrize("m", ["simple", "extended", "dual", "super",
                                "shin-metiu", "modelx", "models", "vibronic"])
def test_surface(m):
    """Surface Writer"""
    tol = 1e-3
    if m in ["vibronic"]:
        options = "-m {:s} --x0 0 0 0 0 0 -s 2 -r -5 5".format(m).split()
    else:
        options = "-m {:s} -r -11 11 -n 200".format(m).split()
    checkdir = os.path.join(testdir, "checks", "surface")
    os.makedirs(checkdir, exist_ok=True)
    outfile = os.path.join(checkdir, f"{m:s}.out")
    options.append(f"--output={outfile}")
    with open(outfile, "w") as f:
        mudslide.surface.main(options)

    form = "f" * (8 if m in ["simple", "extended", "dual"] else 13)
    if m in ["vibronic"]:
        form = "f" * 20
    reffile = os.path.join(testdir, "ref", "surface", "{:s}.ref".format(m))
    with open(reffile) as ref, open(outfile) as out:
        problems = compare_line_by_line(ref, out, form, tol)
        for p in problems:
            print_problem(p)
    assert len(problems) == 0
