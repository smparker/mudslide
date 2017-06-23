#!/usr/bin/env python

from __future__ import print_function

def print_diff(where, line1, line2):
    print("files differ at column: %s" % (", ".join([str(x) for x in where])))
    print("< %s" % (line1.rstrip()))
    print("> %s" % (line2.rstrip()))

if __name__ == "__main__":
    import argparse as ap
    import math as m

    parser = ap.ArgumentParser(description="Compare the traces from two single runs of fssh.py")
    parser.add_argument('file1', help="file 1")
    parser.add_argument('file2', help="file 2")
    parser.add_argument('types', help="string detailing expected types. For example: fffd")
    parser.add_argument('-t', '--tol', default=1.0e-3, help="tolerance for two results to be considered equal")

    args = parser.parse_args()

    try:
        f1 = open(args.file1, "r")
        f2 = open(args.file2, "r")
    except IOError:
        print("One or both files could not be opened!")
        raise

    tol = args.tol

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
    typelist = [ types[x] for x in args.types ]

    failed = False

    for l1, l2 in zip(f1, f2):
        if l1[0] == '#' and l2[0] == '#': continue
        ldata = [ typ(x) for x, typ in zip(l1.split(), typelist) ]
        rdata = [ typ(x) for x, typ in zip(l2.split(), typelist) ]

        problems = []
        for i in range(len(ldata)):
            if not compare(ldata[i], rdata[i], args.types[i]):
                problems.append(i)

        if (len(problems) > 0):
            print_diff(problems, l1, l2)
            failed = True

    try:
        next(f1) # this should throw
        print("file1 is longer than file2")
        failed = True
    except StopIteration:
        pass

    try:
        next(f2) # this should throw
        print("file2 is longer than file1")
        failed = True
    except StopIteration:
        pass

    if not failed:
        print("pass")
