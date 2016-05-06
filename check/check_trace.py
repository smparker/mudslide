#!/usr/bin/env python

def print_diff(where, line1, line2):
    print "%s differ at:" % (", ".join(where))
    print "< %s" % (line1.rstrip())
    print "> %s" % (line2.rstrip())

if __name__ == "__main__":
    import argparse as ap
    import math as m

    parser = ap.ArgumentParser(description="Compare the traces from two single runs of fssh.py")
    parser.add_argument('file1', help="file 1")
    parser.add_argument('file2', help="file 2")
    parser.add_argument('-t', '--tol', default=1.0e-3, help="tolerance for two results to be considered equal")

    args = parser.parse_args()

    try:
        f1 = open(args.file1)
        f2 = open(args.file2)
    except IOError:
        print "One or both files could not be opened!"
        raise

    tol = args.tol

    for l1, l2 in zip(f1, f2):
        split1 = l1.split()
        split2 = l2.split()

        t1 = float(split1[0])
        x1 = float(split1[1])
        p1 = float(split1[2])
        s1 = int(split1[3])

        t2 = float(split2[0])
        x2 = float(split2[1])
        p2 = float(split2[2])
        s2 = int(split2[3])

        problems = []

        # time
        if (abs(t1-t2) > tol):
            problems.append("times")
        # position
        if (abs(x1-x2) > tol):
            problems.append("positions")
        # momentum
        if (abs(p1-p2) > tol):
            problems.append("momenta")
        # active state
        if (s1 != s2):
            problems.append("active states")

        if (len(problems) > 0):
            print_diff(problems, l1, l2)

    try:
        f1.next()
        print "file1 is longer than file2"
    except Exception:
        pass

    try:
        f2.next()
        print "file2 is longer than file1"
    except Exception:
        pass
