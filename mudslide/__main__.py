# -*- coding: utf-8 -*-
"""Code for the mudslide runtime"""

from __future__ import print_function, division

import numpy as np

import pickle
import sys

from .trajectory_sh import TrajectorySH
from .cumulative_sh import TrajectoryCum
from .even_sampling import EvenSamplingTrajectory
from .ehrenfest import Ehrenfest
from .batch import TrajGenConst, TrajGenNormal, BatchedTraj
from .models import models

import argparse as ap

from typing import Any

# Add a method into this dictionary to register it with argparse
methods = {
        "fssh": TrajectorySH,
        "cumulative-sh": TrajectoryCum,
        "ehrenfest": Ehrenfest,
        "even-sampling" : EvenSamplingTrajectory
        }

def main(argv = None, file=sys.stdout) -> None:
    parser = ap.ArgumentParser(description="Mudslide test driver")

    parser.add_argument('-a', '--method', default="fssh", choices=methods.keys(), type=str.lower, help="Variant of SH")
    parser.add_argument('-m', '--model', default='simple', choices=models.keys(), type=str, help="Tully model to plot (%(default)s)")
    parser.add_argument('--electronic', default='exp', choices=("exp", "linear-rk4"), type=str, help="Electronic Propagation method (%(default)s)")
    parser.add_argument('-k', '--krange', default=(0.1,30.0), nargs=2, type=float, help="range of momenta to consider (%(default)s)")
    parser.add_argument('-n', '--nk', default=20, type=int, help="number of momenta to compute (%(default)d)")
    parser.add_argument('-l', '--kspacing', default="linear", type=str, choices=('linear', 'log'), help="linear or log spacing for momenta (%(default)s)")
    parser.add_argument('-K', '--ksampling', default="none", type=str, choices=('none', 'normal'), help="how to sample momenta for a set of simulations (%(default)s)")
    parser.add_argument('-f', '--normal', default=20, type=float, help="standard deviation as a proportion of inverse momentum for normal samping (%(default)s)")
    parser.add_argument('-s', '--samples', default=200, type=int, help="number of samples (%(default)d)")
    parser.add_argument('--sample-stack', default=[10], nargs='*', type=int, help="number of samples at each sampling depth for even sampling algorithm (%(default)s)")
    parser.add_argument('-j', '--nprocs', default=1, type=int, help="number of processors (%(default)d)")
    parser.add_argument('-M', '--mass', default=2000.0, type=float, help="particle mass (%(default)s)")
    parser.add_argument('-t', '--dt', default=20.0, type=float, help="time step in a.u.(%(default)s)")
    parser.add_argument('-y', '--scale_dt', dest="scale_dt", action="store_true", help="use dt=[dt]/k (%(default)s)")
    parser.add_argument('-T', '--nt', default=50000, type=int, help="max number of steps (%(default)s)")
    parser.add_argument('-e', '--every', default=1, type=int, help="store a snapshot every nth step (%(default)s)")
    parser.add_argument('-x', '--position', default=-10.0, type=float, help="starting position (%(default)s)")
    parser.add_argument('-b', '--bounds', default=5.0, type=float, help="bounding box to end simulation (%(default)s)")
    parser.add_argument('-p', '--probability', choices=["tully", "poisson"], default="tully", type=str, help="how to determine hopping probabilities from gk->n * dt (%(default)s)")
    parser.add_argument('-o', '--output', default="averaged", type=str, choices=('averaged', 'single', 'pickle', 'swarm', 'hack'), help="what to produce as output (%(default)s)")
    parser.add_argument('-O', '--outfile', default="sh.pickle", type=str, help="name of pickled file to produce (%(default)s)")
    parser.add_argument('-z', '--seed', default=None, type=int, help="random seed (None)")
    parser.add_argument('--published', dest="published", action="store_true", help="override ranges to use those found in relevant papers (%(default)s)")

    args = parser.parse_args(argv)

    model = models[args.model](mass=args.mass)

    nk = args.nk
    min_k, max_k = args.krange

    if (args.published): # hack spacing to resemble Tully's
        if (args.model == "simple"):
            min_k, max_k = 1.0, 35.0
        elif (args.model == "dual"):
            min_k, max_k = np.log10(np.sqrt(2.0 * args.mass * np.exp(-4.0))), np.log10(np.sqrt(2.0 * args.mass * np.exp(1.0)))
        elif (args.model == "extended"):
            min_k, max_k = 1.0, 35.0
        elif (args.model == "super"):
            min_k, max_k = 0.5, 20.0
        else:
            print("Warning! Published option chosen but no available bounds! Using inputs.", file=sys.stderr)

    kpoints = []
    if args.kspacing == "linear":
        kpoints = np.linspace(min_k, max_k, nk)
    elif args.kspacing == "log":
        kpoints = np.logspace(min_k, max_k, nk)
    else:
        raise Exception("Unrecognized type of spacing")

    trajectory_type = methods[args.method]

    all_results = []

    if (args.output == "averaged" or args.output == "pickle"):
        print("# momentum ", end='', file=file)
        for ist in range(model.nstates()):
            for d in [ "reflected", "transmitted"]:
                print("%d_%s" % (ist, d), end=' ', file=file)
        print(file=file)

    for k in kpoints:
        traj_gen: Any = None
        if args.ksampling == "none":
            traj_gen = TrajGenConst(args.position, k, "ground", seed=args.seed)
        elif args.ksampling == "normal":
            traj_gen = TrajGenNormal(args.position, k, "ground", sigma=args.normal/k, seed = args.seed)

        dt = (args.dt / k) if args.scale_dt else args.dt

        fssh = BatchedTraj(model, traj_gen,
                           trajectory_type = trajectory_type,
                           momentum = k,
                           position = args.position,
                           samples = args.samples,
                           nprocs = args.nprocs,
                           dt = dt,
                           bounds = [ -abs(args.bounds), abs(args.bounds) ],
                           trace_every = args.every,
                           spawn_stack = args.sample_stack,
                           electronic_integration=args.electronic,
                           hopping_probability = args.probability
                   )
        results = fssh.compute()
        outcomes = results.outcomes

        if (args.output == "single"):
            results.traces[0].print(file=file)
        elif (args.output == "swarm"):
            maxsteps = max([ len(t) for t in results.traces ])
            outfiles = [ "state_%d.trace" % i for i in range(model.nstates()) ]
            fils = [ open(o, "w") for o in outfiles ]
            for i in range(maxsteps):
                nswarm = [ 0 for x in fils ]
                for t in results.traces:
                    if i < len(t):
                        iact = t[i]["active"]
                        nswarm[iact] += 1
                        print("{position:12.6f}".format(**t[i]), file=fils[iact])

                for ist in range(model.nstates()):
                    if nswarm[ist] == 0:
                        print("%12.6f" % -9999999, file=fils[ist])
                    print(file=fils[ist])
                    print(file=fils[ist])
            for f in fils:
                f.close()
        elif (args.output == "averaged" or args.output == "pickle"):
            print("%12.6f %s" % (k, " ".join(["%12.6f" % x for x in np.nditer(outcomes)])), file=file)
            if (args.output == "pickle"): # save results for later processing
                all_results.append((k, results))
        elif (args.output == "hack"):
            print("Hack something here, if you like.", file=file)
        else:
            print("Not printing results. This is probably not what you wanted!", file=file)

    if (len(all_results) > 0):
        pickle.dump(all_results, open(args.outfile, "wb"))

if __name__ == "__main__":
    main()
