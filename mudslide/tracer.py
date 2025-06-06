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

from .util import find_unique_name, is_string
from .math import RollingAverage
from .constants import fs_to_au

import yaml


class Trace_:
    """Base class for collecting and storing trajectory data.

    This class provides the interface for collecting and storing data from
    molecular dynamics trajectories. It can store snapshots of the system
    state and record events like surface hops.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize the trace object.

        Parameters
        ----------
        weight : float, optional
            Statistical weight of the trajectory, by default 1.0
        """
        self.weight: float = weight

    def collect(self, snapshot: Any) -> None:
        """Add a single snapshot to the trace.

        Parameters
        ----------
        snapshot : Any
            Snapshot data to add to the trace
        """
        return

    def record_event(self, event_dict: Dict, event_type: str = "hop") -> None:
        """Add a single event to the log.

        Parameters
        ----------
        event_dict : Dict
            Dictionary containing event data
        event_type : str, optional
            Type of event, by default "hop"
        """
        return

    def __iter__(self) -> Iterator:
        """Get an iterator over all snapshots.

        Returns
        -------
        Iterator
            Iterator over snapshots
        """
        pass

    def __getitem__(self, i: int) -> Any:
        """Get a particular snapshot by index.

        Parameters
        ----------
        i : int
            Index of snapshot to retrieve

        Returns
        -------
        Any
            Snapshot data
        """
        pass

    def __len__(self) -> int:
        """Get the number of snapshots in the trace.

        Returns
        -------
        int
            Number of snapshots
        """
        return 0

    def frustrated_hop(self, time: float, hop_from: int, hop_to: int, zeta: float, prob: float) -> None:
        """Record a frustrated hop event.

        Parameters
        ----------
        time : float
            Time of the hop
        hop_from : int
            Initial state
        hop_to : int
            Target state
        zeta : float
            Random number used in hop decision
        prob : float
            Hop probability
        """
        hop_data = {"event": "frustrated_hop", "time": time, "from": hop_from, "to": hop_to, "zeta": zeta, "prob": prob}
        self.record_event(hop_data)

    def form_data(self, snap_dict: Dict) -> Dict:
        """Convert snapshot dictionary to appropriate data types.

        Parameters
        ----------
        snap_dict : Dict
            Dictionary containing snapshot data

        Returns
        -------
        Dict
            Processed snapshot data
        """
        out = {}
        for k, v in snap_dict.items():
            if isinstance(v, list):
                o = np.array(v)
                if k in ["density_matrix"]:
                    o = o.view(dtype=np.complex128)
                out[k] = o
            elif isinstance(v, dict):
                out[k] = self.form_data(v)
            else:
                out[k] = v
        return out

    def clone(self) -> 'Trace_':
        """Create a deep copy of the trace.

        Returns
        -------
        Trace_
            Deep copy of the trace object
        """
        return cp.deepcopy(self)

    def print(self, file: Any = sys.stdout) -> None:
        has_electronic_wfn = "density_matrix" in self[0]
        nst = len(self[0]["density_matrix"]) if has_electronic_wfn else 1
        headerlist = ["%12s" % x for x in ["time", "x", "p", "V", "KE", "E"]]
        if has_electronic_wfn:
            headerlist += ["%12s" % x for x in ["rho_{%d,%d}" % (i, i) for i in range(nst)]]
            headerlist += ["%12s" % x for x in ["H_{%d,%d}" % (i, i) for i in range(nst)]]
            headerlist += ["%12s" % "active"]
            headerlist += ["%12s" % "hopping"]
        print("#" + " ".join(headerlist), file=file)
        for i in self:
            line = " {time:12.6f} {position[0]:12.6f} {velocity[0]:12.6f} {potential:12.6f} {kinetic:12.6f} {energy:12.6f} ".format(
                **i)
            if has_electronic_wfn:
                line += " ".join(["%12.6f" % x for x in np.real(np.diag(i["density_matrix"]))])
                line += " " + " ".join(["%12.6f" % x for x in np.real(np.diag(i["electronics"]["hamiltonian"]))])
                line += " {active:12d} {hopping:12e}".format(**i)
            print(line, file=file)

    def print_egylog(self, file: Any = sys.stdout, T_window: int=50) -> None:
        """Prints the energy log of the trajectory"""
        has_electronic_wfn = "density_matrix" in self[0]
        temperature_avg = RollingAverage(T_window)
        nst = len(self[0]["density_matrix"]) if has_electronic_wfn else 1
        headerlist = ["%12s" % x for x in ["time (fs)", "V (H)", "KE (H)", "T (K)", "<T> (K)", "E (H)"]]
        if has_electronic_wfn:
            headerlist += ["%12s" % x for x in ["rho_{%d,%d}" % (i, i) for i in range(nst)]]
            headerlist += ["%12s" % x for x in ["H_{%d,%d}" % (i, i) for i in range(nst)]]
            headerlist += ["%12s" % "active"]
            headerlist += ["%12s" % "hopping"]
        print("#" + " ".join(headerlist), file=file)
        for i in self:
            i["time"] /= fs_to_au
            T = i["temperature"]
            temperature_avg.insert(T)
            avg_T = temperature_avg.get_average()
            i["avg_temperature"] = avg_T
            line = " {time:12.3f} {potential:12.6f} {kinetic:12.6f} {temperature:8.2f} {avg_temperature:8.2f} {energy:12.6f} ".format(**i)
            if has_electronic_wfn:
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

    def write_trajectory(self, filename: str) -> None:
        """Writes trajectory to an xyz file"""
        ndim = self[-1]["position"].shape[0]
        natoms = ndim // 3
        with open(filename, "w") as f: # TODO needs to be fixed for more general cases. How to get the element?
            for i, snap in enumerate(self):
                print(f"{natoms:d}", file=f)
                print(f"energy: {snap['energy']:g}; time: {snap['time']:f}; step: {i:d}", file=f)
                coords = np.array(snap["position"]).reshape(natoms, 3)
                el = "H"
                for j in range(natoms):
                    print(f"{el:3s} {coords[j, 0]:22.16f} {coords[j, 1]:22.16f} {coords[j, 2]:22.16f}", file=f)

class InMemoryTrace(Trace_):
    """Collect results from a single trajectory"""

    def __init__(self, weight: float = 1.0):
        self.data: List = []
        self.hops: List = []
        self.events: Dict = {}
        self.weight: float = weight

    def collect(self, trajectory_snapshot: Any) -> None:
        """collect and optionally process data"""
        self.data.append(trajectory_snapshot)

    def hop(self, time: float, hop_from: int, hop_to: int, zeta: float, prob: float) -> None:
        self.hops.append({"time": time, "from": hop_from, "to": hop_to, "zeta": zeta, "prob": prob})

    def record_event(self, event_dict: Dict, event_type: str = "hop"):
        if event_type == "hop":
            self.hops.append(event_dict)
            return
        if event_type not in self.events:
            self.events[event_type] = []
        self.events[event_type].append(event_dict)

    def __iter__(self) -> Iterator:
        for snap in self.data:
            yield self.form_data(snap)

    def __getitem__(self, i: int) -> Any:
        return self.form_data(self.data[i])

    def __len__(self) -> int:
        return len(self.data)

    def as_dict(self) -> Dict:
        return {"hops": self.hops, "data": self.data, "weight": self.weight}


class YAMLTrace(Trace_):
    """Collect results from a single trajectory and write to yaml files"""

    def __init__(self, base_name: str = "traj", weight: float = 1.0, log_pitch=512, location="", load_main_log=None):

        self.weight: float = weight
        self.log_pitch = log_pitch
        self.base_name = base_name
        self.location = location

        if not os.path.isdir(self.location) and self.location != "":
            os.makedirs(self.location, exist_ok=True)

        if load_main_log is None:  # initialize
            self.logsize = 0
            self.nlogs = 1
            self.active_logsize = 0
            self.active_logfile = ""
            self.logfiles = []

            self.unique_name = find_unique_name(base_name,
                                                location=self.location,
                                                always_enumerate=True,
                                                ending=".yaml")

            # set log names
            self.main_log = self.unique_name + ".yaml"
            self.active_logfile = self.unique_name + "-log_0.yaml"
            self.hop_log = self.unique_name + "-hops.yaml"
            self.event_logs = {}  # Dictionary to store other event logs

            self.logfiles = [self.active_logfile]

            # create empty files
            open(os.path.join(self.location, self.main_log), "x").close()
            open(os.path.join(self.location, self.active_logfile), "x").close()
            open(os.path.join(self.location, self.hop_log), "x").close()

            self.write_main_log()
        else:
            with open(load_main_log, "r") as f:
                logdata = yaml.safe_load(f)

            self.location = os.path.dirname(load_main_log)
            self.unique_name = logdata["name"]
            self.logfiles = logdata["logfiles"]
            self.main_log = self.unique_name + ".yaml"
            if self.main_log != os.path.basename(load_main_log):
                raise Exception(
                    "It looks like the log file {} was renamed. This is undefined behavior for now!".format(
                        load_main_log))
            self.active_logfile = self.logfiles[-1]
            self.nlogs = logdata["nlogs"]
            self.log_pitch = logdata["log_pitch"]
            self.hop_log = logdata["hop_log"]
            self.event_logs = logdata.get("event_logs", {})  # Get event_logs with default empty dict
            self.weight = logdata["weight"]

            # sizes assume log_pitch never changes. is that safe?
            with open(os.path.join(self.location, self.active_logfile), "r") as f:
                activelog = yaml.safe_load(f)
                self.active_logsize = len(activelog)
            self.logsize = self.log_pitch * (self.nlogs - 1) + self.active_logsize

    def files(self, absolute_path=True):
        rel_files = self.logfiles + [self.main_log, self.hop_log] + list(self.event_logs.values())
        if absolute_path:
            return [os.path.join(self.location, x) for x in rel_files]
        else:
            return rel_files

    def write_main_log(self):
        """Writes main log file, which points to other files for logging information"""
        out = {
            "name": self.unique_name,
            "logfiles": self.logfiles,
            "nlogs": self.nlogs,
            "log_pitch": self.log_pitch,
            "hop_log": self.hop_log,
            "event_logs": self.event_logs,
            "weight": self.weight
        }

        with open(os.path.join(self.location, self.main_log), "w") as f:
            yaml.safe_dump(out, f)

    def collect(self, trajectory_snapshot: Any) -> None:
        """collect and optionally process data"""
        target_log = self.logsize // self.log_pitch
        isnap = self.logsize + 1

        if target_log != (self.nlogs - 1):  # for zero based index, target_log == nlogs means we're out of logs
            self.active_logfile = "{}-log_{:d}.yaml".format(self.unique_name, self.nlogs)
            self.logfiles.append(self.active_logfile)
            self.nlogs += 1
            self.write_main_log()

        with open(os.path.join(self.location, self.active_logfile), "a") as f:
            yaml.safe_dump([trajectory_snapshot], f, explicit_start=False)

        self.logsize += 1

    def hop(self, time: float, hop_from: int, hop_to: int, zeta: float, prob: float) -> None:
        hop_data = {"event": "hop", "time": time, "from": hop_from, "to": hop_to, "zeta": zeta, "prob": prob}
        self.record_event(hop_data)

    def record_event(self, event_dict: Dict, event_type: str = "hop"):
        if event_type == "hop":
            log = self.hop_log
        else:
            if event_type not in self.event_logs:
                # Create new event log file if it doesn't exist
                self.event_logs[event_type] = f"{self.unique_name}-{event_type}.yaml"
                open(os.path.join(self.location, self.event_logs[event_type]), "x").close()
                self.write_main_log()  # Update main log with new event log
            log = self.event_logs[event_type]
        with open(os.path.join(self.location, log), "a") as f:
            yaml.safe_dump([event_dict], f, explicit_start=False)

    def clone(self):
        out = YAMLTrace(base_name=self.base_name,
                        weight=float(self.weight),
                        location=self.location,
                        log_pitch=self.log_pitch)

        out.logsize = self.logsize
        out.nlogs = self.nlogs
        out.active_logsize = self.active_logsize

        out.logfiles = ["{}-log_{}.yaml".format(out.unique_name, i) for i in range(out.nlogs)]
        for selflog, outlog in zip(self.logfiles, out.logfiles):
            shutil.copy(os.path.join(self.location, selflog), os.path.join(self.location, outlog))
        out.active_logfile = out.logfiles[-1]

        # Copy hop log
        shutil.copy(os.path.join(self.location, self.hop_log), os.path.join(self.location, out.hop_log))

        # Copy all event logs
        out.event_logs = {}
        for event_type, log_file in self.event_logs.items():
            out.event_logs[event_type] = f"{out.unique_name}-{event_type}.yaml"
            shutil.copy(os.path.join(self.location, log_file), os.path.join(self.location, out.event_logs[event_type]))

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

            return {"hops": info["hops"], "data": [x for x in self], "weight": self.weight}


def load_log(main_log_name):
    # assuming online yaml logs for now
    out = YAMLTrace(load_main_log=main_log_name)
    return out


def trace_factory(trace_type: str = "yaml"):
    if trace_type == "yaml":
        return YAMLTrace
    elif trace_type == "in_memory":
        return InMemoryTrace
    else:
        raise ValueError("Invalid trace type specified: {}".format(trace_type))


def Trace(trace_type, *args, **kwargs):
    if trace_type is None:
        trace_type = "default"

    if is_string(trace_type):
        trace_type = trace_type.lower()
        if trace_type == "default":
            return InMemoryTrace(*args, **kwargs)
        elif trace_type in [ "memory", "inmemory" ]:
            return InMemoryTrace(*args, **kwargs)
        elif trace_type in [ "yaml" ]:
            return YAMLTrace(*args, **kwargs)
    elif isinstance(trace_type, Trace_):
        return trace_type

    raise Exception("Unrecognized Trace option")


class TraceManager:
    """Manage the collection of observables from a set of trajectories"""
    def __init__(self, trace_type="default", trace_args=[], trace_kwargs={}) -> None:
        self.trace_type = trace_type

        self.trace_args = trace_args
        self.trace_kwargs = trace_kwargs

        self.traces: List = []
        self.outcomes: ArrayLike

    def spawn_tracer(self) -> Trace_:
        """returns a Tracer object that will collect all of the observables for a given trajectory"""
        return Trace(self.trace_type, *self.trace_args, **self.trace_kwargs)

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
        weight_norm = sum((t.weight for t in self.traces))
        outcome = sum((t.weight * t.outcome() for t in self.traces)) / weight_norm
        return outcome

    def counts(self) -> ArrayLike:
        out = np.sum((t.outcome() for t in self.traces))
        return out

    def event_list(self) -> List:
        events = set()
        for t in self.traces:
            for e in t.events:
                events.add(e)
        return list(events)

    def summarize(self, verbose: bool = False, file: Any = sys.stdout) -> None:
        norm = sum((t.weight for t in self.traces))
        print("Using mudslide (v{})".format(__version__), file=file)
        print("------------------------------------", file=file)
        print("# of trajectories: {}".format(len(self.traces)), file=file)

        nhops = np.array([len(t.hops) for t in self.traces])
        hop_stats = [np.sum((t.weight for t in self.traces if len(t.hops) == i)) / norm for i in range(max(nhops) + 1)]
        print("{:5s} {:16s}".format("nhops", "percentage"), file=file)
        for i, w in enumerate(hop_stats):
            print("{:5d} {:16.12f}".format(i, w), file=file)

        if verbose:
            print(file=file)
            print("{:>6s} {:>16s} {:>6s} {:>12s}".format("trace", "runtime", "nhops", "weight"), file=file)
            for i, t in enumerate(self.traces):
                print("{:6d} {:16.4f} {:6d} {:12.6f}".format(i, t.data[-1]["time"], len(t.hops), t.weight / norm),
                      file=file)

        event_list = self.event_list()
        if event_list:
            print("Types of events logged: ", ", ".join(event_list))

        for e in event_list:
            print(file=file)

            print("Statistics for {} event".format(e), file=file)
            nevents = np.array([ len(t.events[e]) if e in t.events else 0 for t in self.traces ])
            if verbose:
                print("{:>6s} {:>16s} {:>6s} {:>12s}".format("trace", "runtime", e, "weight"), file=file)
                for i, nevent in enumerate(nevents):
                    print("{:6d} {:16.4f} {:6d} {:12.6f}".format(i, t.data[-1]["time"], nevent, t.weight/norm), file=file)

            print("  {} mean:      {:8.2f}".format(e, np.mean(nevents)),   file=file)
            print("  {} deviation: {:8.2f}".format(e, np.std(nevents)),    file=file)
            print("  {} min:       {:8.2f}".format(e, np.amin(nevents)),   file=file)
            print("  {} max:       {:8.2f}".format(e, np.amax(nevents)),   file=file)
            print("  {} median:    {:8.2f}".format(e, np.median(nevents)), file=file)

    def as_dict(self) -> Dict:
        out = {"hops": [], "data": [], "weight": []}
        for x in self.traces:
            out["hops"].append(x.as_dict()["hops"])
            out["data"].append(x.as_dict()["data"])
            out["weight"].append(x.as_dict()["weight"])

        return out
