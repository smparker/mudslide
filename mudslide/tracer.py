# -*- coding: utf-8 -*-
"""Collect results from single trajectories"""

from __future__ import annotations

import bz2
import copy as cp
import gzip
import lzma
import os
import sys
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

import yaml
import numpy as np
from numpy.typing import ArrayLike

from .constants import fs_to_au
from .util import find_unique_name, is_string
from .math import RollingAverage
from .version import __version__
from .yaml_format import CompactSafeDumper

_COMPRESSORS = {
    "gzip": (gzip, ".gz"),
    "bz2": (bz2, ".bz2"),
    "xz": (lzma, ".xz"),
}

_COMPRESSION_EXTENSIONS = {
    ".gz": gzip,
    ".bz2": bz2,
    ".xz": lzma,
}


def _open_log(path: str, mode: str = "rt") -> Any:
    """Open a log file, automatically handling compression based on extension."""
    for ext, module in _COMPRESSION_EXTENSIONS.items():
        if path.endswith(ext):
            return module.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _sanitize_for_yaml(data: Any) -> Any:
    """Recursively convert numpy types to native Python types for YAML serialization."""
    if isinstance(data, dict):
        return {k: _sanitize_for_yaml(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_for_yaml(v) for v in data]
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


class Trace_(ABC):
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

    @abstractmethod
    def collect(self, snapshot: Any) -> None:
        """Add a single snapshot to the trace.

        Parameters
        ----------
        snapshot : Any
            Snapshot data to add to the trace
        """

    @abstractmethod
    def record_event(self, event_dict: Dict, event_type: str = "hop") -> None:
        """Add a single event to the log.

        Parameters
        ----------
        event_dict : Dict
            Dictionary containing event data
        event_type : str, optional
            Type of event, by default "hop"
        """

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Get an iterator over all snapshots.

        Returns
        -------
        Iterator
            Iterator over snapshots
        """

    @abstractmethod
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

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of snapshots in the trace.

        Returns
        -------
        int
            Number of snapshots
        """

    def form_data(self, snap_dict: Dict) -> Dict[str, Any]:
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
        out: Dict[str, Any] = {}
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
        """print basic information about a trace
        Really only useful for 1D systems

        :param file: file to print to
        """
        has_electronic_wfn = "density_matrix" in self[0]
        nst = len(self[0]["density_matrix"]) if has_electronic_wfn else 1
        headerlist = [f"{x:>12s}" for x in ["time", "x", "p", "V", "KE", "E"]]
        if has_electronic_wfn:
            headerlist += [f"{f'rho_{i},{i}':>12s}" for i in range(nst)]
            headerlist += [f"{f'H_{i},{i}':>12s}" for i in range(nst)]
            headerlist += [f"{'active':>12s}"]
            headerlist += [f"{'hopping':>12s}"]
        print("#" + " ".join(headerlist), file=file)
        for i in self:
            line = (
                f" {i['time']:12.6f} {i['position'][0]:12.6f} {i['velocity'][0]:12.6f}"
                f" {i['potential']:12.6f} {i['kinetic']:12.6f} {i['energy']:12.6f} "
            )
            if has_electronic_wfn:
                line += " ".join([
                    f"{x:12.6f}" for x in np.real(np.diag(i["density_matrix"]))
                ])
                line += " " + " ".join([
                    f"{x:12.6f}"
                    for x in np.real(np.diag(i["electronics"]["hamiltonian"]))
                ])
                line += f" {i['active']:12d} {i['hopping']:12e}"
            print(line, file=file)

    def print_egylog(self, file: Any = sys.stdout, T_window: int = 50) -> None:
        """Prints the energy log of the trajectory"""
        has_electronic_wfn = "density_matrix" in self[0]
        temperature_avg = RollingAverage(T_window)
        nst = len(self[0]["density_matrix"]) if has_electronic_wfn else 1
        headerlist = [
            f"{x:>12s}" for x in
            ["time (fs)", "V (H)", "KE (H)", "T (K)", "<T> (K)", "E (H)"]
        ]
        if has_electronic_wfn:
            headerlist += [f"{f'rho_{i},{i}':>12s}" for i in range(nst)]
            headerlist += [f"{f'H_{i},{i}':>12s}" for i in range(nst)]
            headerlist += [f"{'active':>12s}"]
            headerlist += [f"{'hopping':>12s}"]
        print("#" + " ".join(headerlist), file=file)
        for i in self:
            i["time"] /= fs_to_au
            T = i["temperature"]
            temperature_avg.insert(T)
            avg_T = temperature_avg.get_average()
            i["avg_temperature"] = avg_T

            line = (
                f" {i['time']:12.3f} {i['potential']:12.6f} {i['kinetic']:12.6f}"
                f" {i['temperature']:12.2f} {i['avg_temperature']:12.2f} {i['energy']:12.6f} "
            )

            if has_electronic_wfn:
                line += " ".join([
                    f"{x:12.6f}" for x in np.real(np.diag(i["density_matrix"]))
                ])
                line += " " + " ".join([
                    f"{x:12.6f}"
                    for x in np.real(np.diag(i["electronics"]["hamiltonian"]))
                ])
                line += f" {i['active']:12d} {i['hopping']:12e}"

            print(line, file=file)

    def outcome(self) -> np.ndarray:
        """Classifies end of simulation: 2*state + [0 for left, 1 for right]"""
        last_snapshot = self[-1]
        ndof = len(last_snapshot["position"])
        nst = len(last_snapshot["density_matrix"])
        position = last_snapshot["position"][0]
        active = last_snapshot["active"]

        out = np.zeros([nst, 2], dtype=np.float64)

        if ndof != 1:
            return out

        lr = 0 if position < 0.0 else 1
        # first bit is left (0) or right (1), second bit is electronic state
        out[active, lr] = 1.0

        return out

    def write_trajectory(self, filename: str) -> None:
        """Writes trajectory to an xyz file"""
        ndof = self[-1]["position"].shape[0]
        natoms = ndof // 3
        with open(filename, "w",
                  encoding="utf-8") as f:  # TODO fix for more general cases.
            for i, snap in enumerate(self):
                print(f"{natoms:d}", file=f)
                print(
                    f"energy: {snap['energy']:g}; time: {snap['time']:f}; step: {i:d}",
                    file=f)
                coords = np.array(snap["position"]).reshape(natoms, 3)
                el = "H"
                for j in range(natoms):
                    print(
                        f"{el:3s} "
                        f"{coords[j, 0]:22.16f} {coords[j, 1]:22.16f} {coords[j, 2]:22.16f}",
                        file=f)


class InMemoryTrace(Trace_):
    """Collect results from a single trajectory"""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight)
        self.data: List = []
        self.hops: List = []
        self.events: Dict = {}

    def collect(self, snapshot: Any) -> None:
        """collect and optionally process data"""
        self.data.append(snapshot)

    def record_event(self, event_dict: Dict, event_type: str = "hop") -> None:
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
        """Return the trace object as a dictionary.

        Returns
        -------
        dict
            Dictionary containing:
                hops : list
                    List of hop events
                data : list
                    List of trajectory snapshots
                weight : float
                    Statistical weight of the trajectory
        """
        return {"hops": self.hops, "data": self.data, "weight": self.weight}


class YAMLTrace(Trace_):
    """Collect results from a single trajectory and write to yaml files"""

    def __init__(self,
                 base_name: str = "traj",
                 weight: float = 1.0,
                 log_pitch: int = 512,
                 location: str = "",
                 load_main_log: Optional[str] = None,
                 compression: Optional[str] = "xz"):
        """Initialize a YAML trace object.

        Parameters
        ----------
        base_name : str, optional
            Base name for log files, by default "traj"
        weight : float, optional
            Statistical weight of the trajectory, by default 1.0
        log_pitch : int, optional
            Number of snapshots per log file, by default 512
        location : str, optional
            Directory to store log files, by default ""
        load_main_log : str, optional
            Path to existing main log file to load from, by default None
        compression : str, optional
            Compression algorithm for completed log chunks.
            Supported values: "gzip", "bz2", "xz", or None (no compression).
            The active log file is always kept uncompressed; only completed
            chunks are compressed when rolling over to a new log file.

        Notes
        -----
        If load_main_log is provided, the trace will load existing log files.
        Otherwise, new log files will be created with the given base_name.
        """
        super().__init__(weight=weight)

        if compression is not None and compression not in _COMPRESSORS:
            raise ValueError(f"Unknown compression type: {compression}. "
                             f"Supported types: {list(_COMPRESSORS.keys())}")

        self.weight: float = weight
        self.log_pitch = log_pitch
        self.base_name = base_name
        self.location = location
        self.compression = compression

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

            with open(os.path.join(self.location, self.main_log),
                      "x",
                      encoding='utf-8') as f:
                pass
            with open(os.path.join(self.location, self.active_logfile),
                      "x",
                      encoding='utf-8') as f:
                pass
            with open(os.path.join(self.location, self.hop_log),
                      "x",
                      encoding='utf-8') as f:
                pass

            self.write_main_log()
        else:
            with open(load_main_log, "r", encoding='utf-8') as f:
                logdata = yaml.safe_load(f)

            self.location = os.path.dirname(load_main_log)
            self.unique_name = logdata["name"]
            self.logfiles = logdata["logfiles"]
            self.main_log = self.unique_name + ".yaml"
            if self.main_log != os.path.basename(load_main_log):
                raise RuntimeError(
                    f"It looks like the log file {load_main_log} was renamed. "
                    "This is undefined behavior for now!")
            self.active_logfile = self.logfiles[-1]
            self.nlogs = logdata["nlogs"]
            self.log_pitch = logdata["log_pitch"]
            self.hop_log = logdata["hop_log"]
            self.event_logs = logdata.get("event_logs", {})
            self.weight = logdata["weight"]

            # sizes assume log_pitch never changes. is that safe?
            with open(os.path.join(self.location, self.active_logfile),
                      "r",
                      encoding='utf-8') as f:
                activelog = yaml.safe_load(f)
                self.active_logsize = len(activelog)
            self.logsize = self.log_pitch * (self.nlogs -
                                             1) + self.active_logsize

    def files(self, absolute_path: bool = True) -> List[str]:
        """returns a list of all files associated with this trace

        :param absolute_path: if True, returns the absolute path to the files,
            otherwise returns the relative path
        :return: list of files
        """
        rel_files = self.logfiles + [self.main_log, self.hop_log] + list(
            self.event_logs.values())
        if absolute_path:
            return [os.path.join(self.location, x) for x in rel_files]
        return rel_files

    def write_main_log(self) -> None:
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

        with open(os.path.join(self.location, self.main_log),
                  "w",
                  encoding='utf-8') as f:
            yaml.dump(out,
                      f,
                      Dumper=CompactSafeDumper,
                      default_flow_style=False)

    def _compress_log(self, log_index: int) -> None:
        """Compress a completed log file in place.

        Parameters
        ----------
        log_index : int
            Index into self.logfiles of the log to compress
        """
        assert self.compression is not None
        module, ext = _COMPRESSORS[self.compression]
        old_name = self.logfiles[log_index]
        new_name = old_name + ext
        old_path = os.path.join(self.location, old_name)
        new_path = os.path.join(self.location, new_name)

        with open(old_path, "rb") as f_in:
            with module.open(new_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(old_path)
        self.logfiles[log_index] = new_name

    def collect(self, snapshot: Any) -> None:
        """collect and optionally process data"""
        target_log = self.logsize // self.log_pitch

        if target_log != (self.nlogs - 1):
            if self.compression is not None:
                self._compress_log(self.nlogs - 1)

            # for zero based index, target_log == nlogs means we're out of logs
            self.active_logfile = f"{self.unique_name}-log_{self.nlogs}.yaml"
            self.logfiles.append(self.active_logfile)
            self.nlogs += 1
            self.write_main_log()

        with open(os.path.join(self.location, self.active_logfile),
                  "a",
                  encoding='utf-8') as f:
            yaml.dump([_sanitize_for_yaml(snapshot)],
                      f,
                      Dumper=CompactSafeDumper,
                      default_flow_style=False,
                      explicit_start=False)

        self.logsize += 1

    def record_event(self, event_dict: Dict, event_type: str = "hop") -> None:
        if event_type == "hop":
            log = self.hop_log
        else:
            if event_type not in self.event_logs:
                # Create new event log file if it doesn't exist
                self.event_logs[
                    event_type] = f"{self.unique_name}-{event_type}.yaml"
                with open(os.path.join(self.location,
                                       self.event_logs[event_type]),
                          "x",
                          encoding='utf-8') as f:
                    pass
                self.write_main_log()  # Update main log with new event log
            log = self.event_logs[event_type]
        with open(os.path.join(self.location, log), "a", encoding='utf-8') as f:
            yaml.dump([_sanitize_for_yaml(event_dict)],
                      f,
                      Dumper=CompactSafeDumper,
                      default_flow_style=False,
                      explicit_start=False)

    def clone(self) -> YAMLTrace:
        """Create a deep copy of the trace.

        Returns
        -------
        YAMLTrace
            Deep copy of the trace object
        """
        out = YAMLTrace(base_name=self.base_name,
                        weight=float(self.weight),
                        location=self.location,
                        log_pitch=self.log_pitch,
                        compression=self.compression)

        out.logsize = self.logsize
        out.nlogs = self.nlogs
        out.active_logsize = self.active_logsize

        # Build logfile names preserving compression extensions from source
        initial_log = out.logfiles[0]  # empty file created by __init__
        out.logfiles = []
        for i in range(out.nlogs):
            src_name = self.logfiles[i]
            suffix = ""
            for ext in _COMPRESSION_EXTENSIONS:
                if src_name.endswith(ext):
                    suffix = ext
                    break
            out_name = f"{out.unique_name}-log_{i}.yaml{suffix}"
            out.logfiles.append(out_name)
            shutil.copy(os.path.join(self.location, src_name),
                        os.path.join(self.location, out_name))

        # Clean up initial empty log if replaced by compressed version
        if out.logfiles[0] != initial_log:
            initial_path = os.path.join(self.location, initial_log)
            if os.path.exists(initial_path):
                os.remove(initial_path)

        out.active_logfile = out.logfiles[-1]

        # Copy hop log
        shutil.copy(os.path.join(self.location, self.hop_log),
                    os.path.join(self.location, out.hop_log))

        # Copy all event logs
        out.event_logs = {}
        for event_type, log_file in self.event_logs.items():
            out.event_logs[event_type] = f"{out.unique_name}-{event_type}.yaml"
            shutil.copy(os.path.join(self.location, log_file),
                        os.path.join(self.location, out.event_logs[event_type]))

        out.write_main_log()

        return out

    def __iter__(self) -> Iterator:
        for log in (os.path.join(self.location, l) for l in self.logfiles):
            with _open_log(log, "rt") as f:
                chunk = yaml.safe_load(f)
                for i in chunk:
                    yield self.form_data(i)

    def __getitem__(self, i: int) -> Any:
        """This is an inefficient way to loop through data"""
        if i < 0:
            i = self.logsize - abs(i)

        if i < 0 or i >= self.logsize:
            raise IndexError(f"Invalid index specified: {i}")

        target_log = i // self.log_pitch
        target_snap = i - target_log * self.log_pitch
        with _open_log(os.path.join(self.location, self.logfiles[target_log]),
                       "rt") as f:
            chunk = yaml.safe_load(f)
            return self.form_data(chunk[target_snap])

    def __len__(self) -> int:
        return self.logsize

    def as_dict(self) -> Dict:
        """return the object as a dictionary"""
        with open(os.path.join(self.location, self.main_log),
                  "r",
                  encoding='utf-8') as f:
            info = yaml.safe_load(f)

            return {
                "hops": info["hops"],
                "data": list(self),
                "weight": self.weight
            }


def load_log(main_log_name: str) -> YAMLTrace:
    """prepare a trace object from a main log file"""
    # assuming online yaml logs for now
    out = YAMLTrace(load_main_log=main_log_name)
    return out


def trace_factory(trace_type: str = "yaml") -> type:
    """returns the appropriate trace class based on the type specified

    Parameters
    ----------
    trace_type : str
        Type of trace to create (yaml, in_memory)
    """
    if trace_type == "yaml":
        return YAMLTrace
    if trace_type == "in_memory":
        return InMemoryTrace
    raise ValueError(f"Invalid trace type specified: {trace_type}")


def Trace(trace_type: Any, *args: Any, **kwargs: Any) -> Trace_:
    """Create a trace object based on the type specified.

    Parameters
    ----------
    trace_type : str
        Type of trace to create (default, memory, yaml)
    *args : Any
        Additional arguments to pass to the trace constructor
    **kwargs : Any
        Additional keyword arguments to pass to the trace constructor

    Returns
    -------
    Trace_
        A trace object of the specified type

    Raises
    ------
    ValueError
        If the trace type is not recognized
    """
    if trace_type is None:
        trace_type = "default"

    if is_string(trace_type):
        trace_type = trace_type.lower()
        if trace_type == "default":
            return InMemoryTrace(*args, **kwargs)
        elif trace_type in ["memory", "inmemory"]:
            return InMemoryTrace(*args, **kwargs)
        elif trace_type in ["yaml"]:
            return YAMLTrace(*args, **kwargs)
    elif isinstance(trace_type, Trace_):
        return trace_type

    raise ValueError("Unrecognized Trace option")


class TraceManager:
    """Manage the collection of observables from a set of trajectories"""

    def __init__(self,
                 trace_type: str = "default",
                 trace_args: Optional[List[Any]] = None,
                 trace_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.trace_type = trace_type

        self.trace_args = trace_args if trace_args is not None else []
        self.trace_kwargs = trace_kwargs if trace_kwargs is not None else {}

        self.traces: List = []
        self.outcomes: np.ndarray

    def spawn_tracer(self) -> Trace_:
        """returns a Tracer object that collects all of the observables for a given trajectory"""
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

    def outcome(self) -> np.ndarray:
        """summarize outcomes from entire set of traces"""
        weight_norm = sum((t.weight for t in self.traces))
        outcome = sum(
            (t.weight * t.outcome() for t in self.traces)) / weight_norm
        return outcome

    def counts(self) -> np.ndarray:
        """summarize outcomes from entire set of traces"""
        out = sum(t.outcome() for t in self.traces)
        return out

    def event_list(self) -> List:
        """return a list of all events logged"""
        events = set()
        for t in self.traces:
            for e in t.events:
                events.add(e)
        return list(events)

    def summarize(self, verbose: bool = False, file: Any = sys.stdout) -> None:
        """print a summary of the traces"""
        norm = sum((t.weight for t in self.traces))
        print(f"Using mudslide (v{__version__})", file=file)
        print("------------------------------------", file=file)
        print(f"# of trajectories: {len(self.traces)}", file=file)

        nhops = np.array([len(t.hops) for t in self.traces])
        hop_stats = [
            sum(t.weight for t in self.traces if len(t.hops) == i) /
            norm for i in range(max(nhops) + 1)
        ]
        print(f"{'nhops':5s} {'percentage':16s}", file=file)
        for i, w in enumerate(hop_stats):
            print(f"{i:5d} {w:16.12f}", file=file)

        if verbose:
            print(file=file)
            print(
                f"{'trace':>6s} {'runtime':>16s} {'nhops':>6s} {'weight':>12s}",
                file=file)
            for i, t in enumerate(self.traces):
                print(
                    f"{i:6d} {t.data[-1]['time']:16.4f} {len(t.hops):6d} {t.weight/norm:12.6f}",
                    file=file)
        event_list = self.event_list()
        if event_list:
            print("Types of events logged: ", ", ".join(event_list))

        for e in event_list:
            print(file=file)

            print(f"Statistics for {e} event", file=file)
            nevents = np.array(
                [len(t.events[e]) if e in t.events else 0 for t in self.traces])
            if verbose:
                print(f"{'trace':>6s} {'runtime':>16s} {e:>6s} {'weight':>12s}",
                      file=file)
                for i, nevent in enumerate(nevents):
                    print(
                        f"{i:6d} "
                        f"{self[i][-1]['time']:16.4f} "
                        f"{nevent:6d} "
                        f"{self[i].weight/norm:12.6f}",
                        file=file)
            print(f"  {e} mean:      {np.mean(nevents):8.2f}", file=file)
            print(f"  {e} deviation: {np.std(nevents):8.2f}", file=file)
            print(f"  {e} min:       {np.amin(nevents):8.2f}", file=file)
            print(f"  {e} max:       {np.amax(nevents):8.2f}", file=file)
            print(f"  {e} median:    {np.median(nevents):8.2f}", file=file)

    def as_dict(self) -> Dict:
        """return the object as a dictionary"""
        out: Dict[str, list[Any]] = {"hops": [], "data": [], "weight": []}
        for x in self.traces:
            out["hops"].append(x.as_dict()["hops"])
            out["data"].append(x.as_dict()["data"])
            out["weight"].append(x.as_dict()["weight"])

        return out
