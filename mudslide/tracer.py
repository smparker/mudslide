# -*- coding: utf-8 -*-
"""Collect results from single trajectories"""

from __future__ import print_function, division

from .version import __version__

import numpy as np
import sys, os
import copy as cp
import shutil

from typing import List, Any, Dict, Iterator
from .typing import ArrayLike

from .util import find_unique_name

import yaml

class Trace_(object):
    def __init__(self, weight: float = 1.0):
        self.weight: float = weight

    def collect(self, snapshot: Any) -> None:
        """add a single snapshot to the trace"""
        return

    def record_event(self, event_dict: Dict) -> None:
        """add a single event (e.g., hop or collapse) to the log"""
        return

    def __iter__(self) -> Iterator:
        """option to iterate through every snapshot"""
        pass

    def __getitem__(self, i: int) -> Any:
        """option to get a particular snapshot"""
        pass

    def __len__(self) -> int:
        return 0

    def form_data(self, snap_dict: Dict) -> Dict:
        out = {}
        for k, v in snap_dict.items():
            if isinstance(v, list):
                o = np.array(v)
                if k in [ "density_matrix" ]:
                    o = o.view(dtype=np.complex128)
                out[k] = o
            elif isinstance(v, dict):
                out[k] = self.form_data(v)
            else:
                out[k] = v
        return out

    def clone(self) -> 'Trace_':
        return cp.deepcopy(self)

    def print(self, file: Any = sys.stdout) -> None:
        nst = len(self[0]["density_matrix"])
        headerlist =  [ "%12s" % x for x in [ "time", "x", "p", "V", "T", "E" ] ]
        headerlist += [ "%12s" % x for x in [ "rho_{%d,%d}" % (i,i) for i in range(nst) ] ]
        headerlist += [ "%12s" % x for x in [ "H_{%d,%d}" % (i,i) for i in range(nst) ] ]
        headerlist += [ "%12s" % "active" ]
        headerlist += [ "%12s" % "hopping" ]
        print("#" + " ".join(headerlist), file=file)
        for i in self:
            line = " {time:12.6f} {position[0]:12.6f} {momentum[0]:12.6f} {potential:12.6f} {kinetic:12.6f} {energy:12.6f} ".format(**i)
            line += " ".join(["%12.6f" % x for x in np.real(np.diag(i["density_matrix"]))])
            line += " " + " ".join(["%12.6f" % x for x in np.real(np.diag(i["electronics"]["hamiltonian"]))])
            line += " {active:12d} {hopping:12e}".format(**i)
            print(line, file=file)

    def outcome(self) -> ArrayLike:
        """Classifies end of simulation: 2*state + [0 for left, 1 for right]"""
        last_snapshot = self[-1]
        ndim = len(last_snapshot["position"])
        nst = len(last_snapshot["density_matrix"])
        position = last_snapshot["position"][0]
        active = last_snapshot["active"]

        out = np.zeros([nst, 2], dtype=np.float64)

        if ndim != 1:
            return out

        lr = 0 if position < 0.0 else 1
        # first bit is left (0) or right (1), second bit is electronic state
        out[active, lr] = 1.0

        return out

class InMemoryTrace(Trace_):
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

    def record_event(self, event_dict):
        self.hops.append(event_dict)

    def __iter__(self) -> Iterator:
        for snap in self.data:
            yield self.form_data(snap)

    def __getitem__(self, i: int) -> Any:
        return self.form_data(self.data[i])

    def __len__(self) -> int:
        return len(self.data)

    def as_dict(self) -> Dict:
        return {
                "hops" : self.hops,
                "data" : self.data,
                "weight" : self.weight
                }

class YAMLTrace(Trace_):
    """Collect results from a single trajectory and write to yaml files"""
    def __init__(self, base_name: str = "traj", weight: float = 1.0,
            log_pitch = 512, location="", load_main_log=None):

        self.weight: float = weight
        self.log_pitch = log_pitch
        self.base_name = base_name
        self.location = location

        if not os.path.isdir(self.location) and self.location != "":
            os.makedirs(self.location, exist_ok=True)

        if load_main_log is None: # initialize
            self.logsize = 0
            self.nlogs = 1
            self.active_logsize = 0
            self.active_logfile = ""
            self.logfiles = [ ]

            self.unique_name = find_unique_name(base_name, location=self.location, always_enumerate = True, ending=".yaml")

            # set log names
            self.main_log = self.unique_name + ".yaml"
            self.active_logfile = self.unique_name + "-log_0.yaml"
            self.event_log = self.unique_name + "-events.yaml"

            self.logfiles = [ self.active_logfile ]

            # create empty files
            open(os.path.join(self.location, self.main_log), "x").close()
            open(os.path.join(self.location, self.active_logfile), "x").close()
            open(os.path.join(self.location, self.event_log), "x").close()

            self.write_main_log()
        else:
            with open(load_main_log, "r") as f:
                logdata = yaml.safe_load(f)

            self.location = os.path.dirname(load_main_log)
            self.unique_name = logdata["name"]
            self.logfiles = logdata["logfiles"]
            self.main_log = self.unique_name + ".yaml"
            if self.main_log != os.path.basename(load_main_log):
                raise Exception("It looks like the log file {} was renamed. This is undefined behavior for now!".format(load_main_log))
            self.active_logfile = self.logfiles[-1]
            self.nlogs = logdata["nlogs"]
            self.log_pitch = logdata["log_pitch"]
            self.event_log = logdata["event_log"]
            self.weight = logdata["weight"]

            # sizes assume log_pitch never changes. is that safe?
            with open(os.path.join(self.location, self.active_logfile), "r") as f:
                activelog = yaml.safe_load(f)
                self.active_logsize = len(activelog)
            self.logsize = self.log_pitch * (self.nlogs-1) + self.active_logsize

    def files(self, absolute_path=True):
        rel_files = self.logfiles + [ self.main_log, self.event_log ]
        if absolute_path:
            return [ os.path.join(self.location, x) for x in rel_files ]
        else:
            return rel_files

    def write_main_log(self):
        """Writes main log file, which points to other files for logging information"""
        out = {
            "name" : self.unique_name,
            "logfiles" : self.logfiles,
            "nlogs" : self.nlogs,
            "log_pitch" : self.log_pitch,
            "event_log" : self.event_log,
            "weight" : self.weight
            }

        with open(os.path.join(self.location, self.main_log), "w") as f:
            yaml.safe_dump(out, f)

    def collect(self, trajectory_snapshot: Any) -> None:
        """collect and optionally process data"""
        target_log = self.logsize // self.log_pitch
        isnap = self.logsize + 1

        if target_log != (self.nlogs - 1): # for zero based index, target_log == nlogs means we're out of logs
            self.active_logfile = "{}-log_{:d}.yaml".format(self.unique_name, self.nlogs)
            self.logfiles.append(self.active_logfile)
            self.nlogs += 1
            self.write_main_log()

        with open(os.path.join(self.location, self.active_logfile), "a") as f:
            yaml.safe_dump([trajectory_snapshot], f, explicit_start=False)

        self.logsize += 1

    def hop(self, time: float, hop_from: int, hop_to: int, zeta: float, prob: float) -> None:
        hop_data = {
            "event": "hop",
            "time" : time,
            "from" : hop_from,
            "to"   : hop_to,
            "zeta" : zeta,
            "prob" : prob
            }
        self.record_event(hop_data)

    def record_event(self, event_dict):
        with open(os.path.join(self.location, self.event_log), "a") as f:
            yaml.safe_dump([event_dict], f, explicit_start=False)

    def clone(self):
        out = YAMLTrace(name=self.base_name, weight=float(self.weight), log_pitch=self.log_pitch)

        out.logsize = self.logsize
        out.nlogs = self.nlogs
        out.active_logsize = self.active_logsize

        out.logfiles = [ "{}-log_{}.yaml".format(out.unique_name, i) for i in range(out.nlogs) ]
        for selflog, outlog in zip(self.logfiles, out.logfiles):
            shutil.copy(selflog, outlog)
        out.active_logfile = out.logfiles[-1]

        shutil.copy(self.event_log, out.event_log)

        out.write_main_log()

        return out

    def __iter__(self) -> Iterator:
        for log in (os.path.join(self.location, l) for l in self.logfiles):
            with open(log, "r") as f:
                chunk = yaml.safe_load(f)
                for i in chunk:
                    yield self.form_data(i)

    def __getitem__(self, i: int) -> Any:
        """This is an inefficient way to loop through data"""
        if i < 0:
            i = self.logsize - abs(i)

        if i < 0 or i >= self.logsize:
            raise IndexError("Invalid index specified: {}".format(i))

        target_log = i // self.log_pitch
        target_snap = i - target_log * self.log_pitch
        with open(os.path.join(self.location, self.logfiles[target_log]), "r") as f:
            chunk = yaml.safe_load(f)
            return self.form_data(chunk[target_snap])

    def __len__(self) -> int:
        return self.logsize

    def as_dict(self) -> Dict:
        with open(os.path.join(self.location, self.main_log), "r") as f:
            info = yaml.safe_load(f)

            return {
                    "hops" : info["hops"],
                    "data" : [ x for x in self ],
                    "weight" : self.weight
                    }

def load_log(main_log_name):
    # assuming online yaml logs for now
    out = YAMLTrace(load_main_log=main_log_name)
    return out

DefaultTrace = InMemoryTrace

class TraceManager(object):
    """Manage the collection of observables from a set of trajectories"""
    def __init__(self, TraceType=InMemoryTrace) -> None:
        self.TraceType = TraceType

        self.traces: List = []
        self.outcomes: ArrayLike

    def spawn_tracer(self) -> Trace_:
        """returns a Tracer object that will collect all of the observables for a given trajectory"""
        return self.TraceType()

    def merge_tracer(self, tracer: Trace_) -> None:
        """accepts a Tracer object and adds it to list of traces"""
        self.traces.append(tracer)

    def add_batch(self, traces: List[Trace_]) -> None:
        """merge other manager into self"""
        self.traces.extend(traces)

    def __iter__(self) -> Iterator:
        return self.traces.__iter__()

    def __getitem__(self, i: int) -> Trace_:
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

    def as_dict(self) -> Dict:
        out = {
                "hops" : [],
                "data" : [],
                "weight" : []
            }
        for x in self.traces:
            out["hops"].append(x.as_dict()["hops"])
            out["data"].append(x.as_dict()["data"])
            out["weight"].append(x.as_dict()["weight"])

        return out

