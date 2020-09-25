# -*- coding: utf-8 -*-
"""Collect results from single trajectories"""

from __future__ import print_function, division

from .version import __version__

import numpy as np
import sys

from typing import List, Any, Dict, Iterator
from .typing import ArrayLike

class Trace(object):
    """Collect results from a single trajectory"""
    def __init__(self, weight: float = 1.0):
        self.data: List = []
        self.hops: List = []
        self.weight: float = weight

    def collect(self, trajectory_snapshot: Any) -> None:
        """collect and optionally process data"""
        self.data.append(trajectory_snapshot)

    def hop(self, time: float, hop_from: int, hop_to: int, zeta: float, prob: float) -> None:
        self.hops.append({
            "time" : time,
            "from" : hop_from,
            "to"   : hop_to,
            "zeta" : zeta,
            "prob" : prob
            })

    def __iter__(self) -> Iterator:
        return self.data.__iter__()

    def __getitem__(self, i: int) -> Any:
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def print(self, file: Any = sys.stdout) -> None:
        nst = self.data[0]["density_matrix"].shape[0]
        headerlist =  [ "%12s" % x for x in [ "time", "x", "p", "V", "T", "E" ] ]
        headerlist += [ "%12s" % x for x in [ "rho_{%d,%d}" % (i,i) for i in range(nst) ] ]
        headerlist += [ "%12s" % x for x in [ "H_{%d,%d}" % (i,i) for i in range(nst) ] ]
        headerlist += [ "%12s" % "active" ]
        headerlist += [ "%12s" % "hopping" ]
        print("#" + " ".join(headerlist), file=file)
        for i in self.data:
            line = " {time:12.6f} {position[0]:12.6f} {momentum[0]:12.6f} {potential:12.6f} {kinetic:12.6f} {energy:12.6f} ".format(**i)
            line += " ".join(["%12.6f" % x for x in np.real(np.diag(i["density_matrix"]))])
            line += " " + " ".join(["%12.6f" % x for x in np.real(np.diag(i["electronics"].hamiltonian))])
            line += " {active:12d} {hopping:12e}".format(**i)
            print(line, file=file)

    def outcome(self) -> ArrayLike:
        """Classifies end of simulation: 2*state + [0 for left, 1 for right]"""
        last_snapshot = self.data[-1]
        nst = last_snapshot["density_matrix"].shape[0]
        position = last_snapshot["position"]
        active = last_snapshot["active"]

        out = np.zeros([nst, 2], dtype=np.float64)

        lr = 0 if position < 0.0 else 1
        # first bit is left (0) or right (1), second bit is electronic state
        out[active, lr] = 1.0

        return out

    def as_dict(self) -> Dict:
        return {
                "hops" : self.hops,
                "data" : self.data,
                "weight" : self.weight
                }


class TraceManager(object):
    """Manage the collection of observables from a set of trajectories"""
    def __init__(self) -> None:
        self.traces: List = []
        self.outcomes: ArrayLike

    def spawn_tracer(self) -> Trace:
        """returns a Tracer object that will collect all of the observables for a given trajectory"""
        return Trace()

    def merge_tracer(self, tracer: Trace) -> None:
        """accepts a Tracer object and adds it to list of traces"""
        self.traces.append(tracer)

    def add_batch(self, traces: List[Trace]) -> None:
        """merge other manager into self"""
        self.traces.extend(traces)

    def __iter__(self) -> Iterator:
        return self.traces.__iter__()

    def __getitem__(self, i: int) -> Trace:
        return self.traces[i]

    def outcome(self) -> ArrayLike:
        weight_norm = sum( (t.weight for t in self.traces) )
        outcome = sum((t.weight * t.outcome() for t in self.traces))/weight_norm
        return outcome

    def counts(self) -> ArrayLike:
        out = np.sum((t.outcome() for t in self.traces))
        return out

    def summarize(self, verbose: bool = False, file: Any = sys.stdout) -> None:
        norm = sum((t.weight for t in self.traces))
        print("Using mudslide (v{})".format(__version__), file=file)
        print("------------------------------------", file=file)
        print("# of trajectories: {}".format(len(self.traces)), file=file)

        nhops = [ len(t.hops) for t in self.traces ]
        hop_stats = [ np.sum( (t.weight for t in self.traces if len(t.hops)==i) )/norm for i in range(max(nhops)+1) ]
        print("{:5s} {:16s}".format("nhops", "percentage"), file=file)
        for i, w in enumerate(hop_stats):
            print("{:5d} {:16.12f}".format(i, w), file=file)

        if verbose:
            print(file=file)
            print("{:>6s} {:>16s} {:>6s} {:>12s}".format("trace", "runtime", "nhops", "weight"), file=file)
            for i, t in enumerate(self.traces):
                print("{:6d} {:16.4f} {:6d} {:12.6f}".format(i, t.data[-1]["time"], len(t.hops), t.weight/norm), file=file)

