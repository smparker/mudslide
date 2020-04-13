#!/usr/bin/env python
## @package tracer
#  Module responsible for propagating surface hopping trajectories

# fssh: program to run surface hopping simulations for model problems
# Copyright (C) 2018-2020, Shane Parker <shane.parker@case.edu>
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

from .version import __version__

import numpy as np
import sys

## Collect results from a single trajectory
class Trace(object):
    def __init__(self, weight=1.0):
        self.data = []
        self.hops = []
        self.weight = weight

    ## collect and optionally process data
    def collect(self, trajectory_snapshot):
        self.data.append(trajectory_snapshot)

    def hop(self, time, hop_from, hop_to, zeta, prob):
        self.hops.append({
            "time" : time,
            "from" : hop_from,
            "to"   : hop_to,
            "zeta" : zeta,
            "prob" : prob
            })

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def print(self, f=sys.stdout):
        nst = self.data[0]["density_matrix"].shape[0]
        headerlist =  [ "%12s" % x for x in [ "time", "x", "p", "V", "T", "E" ] ]
        headerlist += [ "%12s" % x for x in [ "rho_{%d,%d}" % (i,i) for i in range(nst) ] ]
        headerlist += [ "%12s" % x for x in [ "H_{%d,%d}" % (i,i) for i in range(nst) ] ]
        headerlist += [ "%12s" % "active" ]
        headerlist += [ "%12s" % "hopping" ]
        print("#" + " ".join(headerlist), file=f)
        for i in self.data:
            line = " {time:12.6f} {position[0]:12.6f} {momentum[0]:12.6f} {potential:12.6f} {kinetic:12.6f} {energy:12.6f} ".format(**i)
            line += " ".join(["%12.6f" % x for x in np.real(np.diag(i["density_matrix"]))])
            line += " " + " ".join(["%12.6f" % x for x in np.real(np.diag(i["electronics"].hamiltonian))])
            line += " {active:12d} {hopping:12e}".format(**i)
            print(line, file=f)

    ## Classifies end of simulation:
    #
    #  2*state + [0 for left, 1 for right]
    def outcome(self):
        last_snapshot = self.data[-1]
        nst = last_snapshot["density_matrix"].shape[0]
        position = last_snapshot["position"]
        active = last_snapshot["active"]

        out = np.zeros([nst, 2], dtype=np.float64)

        lr = 0 if position < 0.0 else 1
        # first bit is left (0) or right (1), second bit is electronic state
        out[active, lr] = 1.0

        return out

    def as_dict(self):
        return {
                "hops" : self.hops,
                "data" : self.data,
                "weight" : self.weight
                }


## Class to manage the collection of observables from a set of trajectories
class TraceManager(object):
    def __init__(self):
        self.traces = []

    ## returns a Tracer object that will collect all of the observables for a given
    #  trajectory
    def spawn_tracer(self):
        return Trace()

    ## accepts a Tracer object and adds it to list of traces
    def merge_tracer(self, tracer):
        self.traces.append(tracer)

    ## merge other manager into self
    def add_batch(self, traces):
        self.traces.extend(traces)

    def __iter__(self):
        return self.traces.__iter__()

    def __getitem__(self, i):
        return self.traces[i]

    def outcome(self):
        weight_norm = np.sum( (t.weight for t in self.traces) )
        outcome = np.sum((t.weight * t.outcome() for t in self.traces))/weight_norm
        return outcome

    def summarize(self, f=sys.stdout, verbose=False):
        norm = sum((t.weight for t in self.traces))
        print("Running the FSSH package ({})".format(__version__), file=f)
        print("------------------------------------", file=f)
        print("# of trajectories: {}".format(len(self.traces)), file=f)

        nhops = [ len(t.hops) for t in self.traces ]
        hop_stats = [ np.sum( (t.weight for t in self.traces if len(t.hops)==i) )/norm for i in range(max(nhops)+1) ]
        print("{:5s} {:16s}".format("nhops", "percentage"), file=f)
        for i, w in enumerate(hop_stats):
            print("{:5d} {:16.12f}".format(i, w), file=f)

        if verbose:
            print(file=f)
            print("{:>6s} {:>16s} {:>6s} {:>12s}".format("trace", "runtime", "nhops", "weight"), file=f)
            for i, t in enumerate(self.traces):
                print("{:6d} {:16.4f} {:6d} {:12.6f}".format(i, t.data[-1]["time"], len(t.hops), t.weight/norm), file=f)

