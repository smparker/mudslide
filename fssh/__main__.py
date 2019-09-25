#!/usr/bin/env python
## @package fssh
#  Module responsible for propagating surface hopping trajectories

# fssh: program to run surface hopping simulations for model problems
# Copyright (C) 2018, Shane Parker <smparker@uci.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import print_function, division

import numpy as np

import math as m
import multiprocessing as mp
import collections
import pickle

import sys

from .fssh import TrajectorySH, TrajectoryCum, Ehrenfest
from .trajectory import TrajGenConst, TrajGenNormal, BatchedTraj
from .models import models

import argparse as ap

# Add a method into this dictionary to register it with argparse
methods = {
        "fssh": TrajectorySH,
        "fssh-cumulative": TrajectoryCum,
        "ehrenfest": Ehrenfest
        }

def main():

    parser = ap.ArgumentParser(description="FSSH driver")

    parser.add_argument('-a', '--method', default="fssh", choices=methods.keys(), type=str.lower, help="Variant of SH")
    parser.add_argument('-m', '--model', default='simple', choices=models.keys(), type=str, help="Tully model to plot (%(default)s)")
    parser.add_argument('-k', '--krange', default=(0.1,30.0), nargs=2, type=float, help="range of momenta to consider (%(default)s)")
    parser.add_argument('-n', '--nk', default=20, type=int, help="number of momenta to compute (%(default)d)")
    parser.add_argument('-l', '--kspacing', default="linear", type=str, choices=('linear', 'log'), help="linear or log spacing for momenta (%(default)s)")
    parser.add_argument('-K', '--ksampling', default="none", type=str, choices=('none', 'normal'), help="how to sample momenta for a set of simulations (%(default)s)")
    parser.add_argument('-f', '--normal', default=20, type=float, help="standard deviation as a proportion of inverse momentum for normal samping (%(default)s)")
    parser.add_argument('-s', '--samples', default=200, type=int, help="number of samples (%(default)d)")
    parser.add_argument('-j', '--nprocs', default=2, type=int, help="number of processors (%(default)d)")
    parser.add_argument('-M', '--mass', default=2000.0, type=float, help="particle mass (%(default)s)")
    parser.add_argument('-t', '--dt', default=20.0, type=float, help="time step in a.u.(%(default)s)")
    parser.add_argument('-y', '--scale_dt', dest="scale_dt", action="store_true", help="scale (hack-like) time step using momentum (%(default)s)")
    parser.add_argument('-T', '--nt', default=50000, type=int, help="max number of steps (%(default)s)")
    parser.add_argument('-x', '--position', default=-10.0, type=float, help="starting position (%(default)s)")
    parser.add_argument('-b', '--bounds', default=5.0, type=float, help="bounding box to end simulation (%(default)s)")
    parser.add_argument('-o', '--output', default="averaged", type=str, choices=('averaged', 'single', 'pickle', 'swarm', 'hack'), help="what to produce as output (%(default)s)")
    parser.add_argument('-O', '--outfile', default="sh.pickle", type=str, help="name of pickled file to produce (%(default)s)")
    parser.add_argument('-z', '--seed', default=None, type=int, help="random seed (current date)")
    parser.add_argument('--published', dest="published", action="store_true", help="override ranges to use those found in relevant papers (%(default)s)")

    args = parser.parse_args()

    model = models[args.model]()

    if (args.seed is not None):
        np.random.seed(args.seed)

    nk = args.nk
    min_k, max_k = args.krange

    if (args.published): # hack spacing to resemble Tully's
        if (args.model == "simple"):
            min_k, max_k = 1.0, 35.0
        elif (args.model == "dual"):
            min_k, max_k = m.log10(m.sqrt(2.0 * args.mass * m.exp(-4.0))), m.log10(m.sqrt(2.0 * args.mass * m.exp(1.0)))
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
        print("# momentum ", end='')
        for ist in range(model.nstates()):
            for d in [ "reflected", "transmitted"]:
                print("%d_%s" % (ist, d), end=' ')
        print()

    for k in kpoints:
        if args.ksampling == "none":
            traj_gen = TrajGenConst(args.position, k, "ground")
        elif args.ksampling == "normal":
            traj_gen = TrajGenNormal(args.position, k, "ground", sigma = args.normal/k)

        # hack-y scale of time step so that the input amount roughly makes sense for 10.0 a.u.
        dt = args.dt * (10.0 / k) if args.scale_dt else args.dt

        fssh = BatchedTraj(model, traj_gen,
                           trajectory_type = trajectory_type,
                           momentum = k,
                           position = args.position,
                           mass = args.mass,
                           samples = args.samples,
                           nprocs = args.nprocs,
                           dt = dt,
                           seed = args.seed
                   )
        results = fssh.compute()
        outcomes = results.outcomes

        if (args.output == "single"):
            nst = results.traces[0][0].rho.shape[0]
            headerlist = [ "%12s" % x for x in [ "time", "x", "p", "V", "T", "E" ] ]
            headerlist += [ "%12s" % x for x in [ "rho_{%d,%d}" % (i,i) for i in range(nst) ] ]
            headerlist += [ "%12s" % x for x in [ "H_{%d,%d}" % (i,i) for i in range(nst) ] ]
            headerlist += [ "%12s" % "active" ]
            headerlist += [ "%12s" % "hopping" ]
            print("#" + " ".join(headerlist))
            for i in results.traces[0]:
                line = " %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f " % (i.time, i.position, i.momentum, i.potential, i.kinetic, i.energy)
                line += " ".join(["%12.6f" % x for x in np.real(np.diag(i.rho))])
                line += " " + " ".join(["%12.6f" % x for x in np.real(np.diag(i.electronics.hamiltonian))])
                line += " %12d" % i.activestate
                line += " %12e" % i.hopping
                print(line)
        elif (args.output == "swarm"):
            maxsteps = max([ len(t) for t in results.traces ])
            outfiles = [ "state_%d.trace" % i for i in range(model.nstates()) ]
            fils = [ open(o, "w") for o in outfiles ]
            for i in range(maxsteps):
                nswarm = [ 0 for x in fils ]
                for t in results.traces:
                    if i < len(t):
                        iact = t[i].activestate
                        nswarm[iact] += 1
                        print("%12.6f" % t[i].position, file=fils[iact])

                for ist in range(model.nstates()):
                    if nswarm[ist] == 0:
                        print("%12.6f" % -9999999, file=fils[ist])
                    print(file=fils[ist])
                    print(file=fils[ist])
            for f in fils:
                f.close()
        elif (args.output == "averaged" or args.output == "pickle"):
            print("%12.6f %s" % (k, " ".join(["%12.6f" % x for x in np.nditer(outcomes)])))
            if (args.output == "pickle"): # save results for later processing
                all_results.append((k, results))
        elif (args.output == "hack"):
            print("Hack something here, if you like.")
        else:
            print("Not printing results. This is probably not what you wanted!")

    if (len(all_results) > 0):
        pickle.dump(all_results, open(args.outfile, "wb"))

if __name__ == "__main__":
    main()
