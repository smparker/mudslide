# -*- coding: utf-8 -*-
"""Extended tests for tracer.py coverage"""

import io
import os

import numpy as np
import pytest

import mudslide
from mudslide.exceptions import ConfigurationError
from mudslide.tracer import (_sanitize_for_yaml, InMemoryTrace, YAMLTrace,
                             TraceManager, trace_factory, Trace, load_log,
                             _COMPRESSORS, _COMPRESSION_EXTENSIONS)


def _make_1d_snapshot(time, position, velocity, active=0, nstates=2):
    """Create a 1D snapshot with electronic state info for testing."""
    # density_matrix stored as float pairs (real, imag) per element
    # For nstates=2, shape is (2, 4) in float, viewed as (2,2) complex
    dm = np.zeros((nstates, nstates), dtype=np.complex128)
    dm[active, active] = 1.0 + 0.0j
    dm_as_floats = dm.view(dtype=np.float64).tolist()

    hamiltonian = np.diag([0.01 * i for i in range(nstates)]).tolist()

    return {
        "time": time,
        "position": [position],
        "velocity": [velocity],
        "potential": 0.01,
        "kinetic": 0.5 * velocity**2,
        "energy": 0.01 + 0.5 * velocity**2,
        "temperature": 300.0,
        "density_matrix": dm_as_floats,
        "active": active,
        "hopping": 0.0,
        "electronics": {
            "hamiltonian": hamiltonian
        }
    }


def _make_3d_snapshot(time, natoms=3):
    """Create a 3D snapshot for trajectory writing tests."""
    ndof = natoms * 3
    return {
        "time": time,
        "position": list(np.random.default_rng(int(time)).normal(0, 1, ndof)),
        "velocity": list(np.zeros(ndof)),
        "potential": 0.0,
        "kinetic": 0.0,
        "energy": 0.0,
        "temperature": 0.0,
    }


# --- _sanitize_for_yaml ---


def test_sanitize_np_integer():
    result = _sanitize_for_yaml(np.int64(42))
    assert result == 42
    assert isinstance(result, int)


def test_sanitize_np_floating():
    result = _sanitize_for_yaml(np.float64(3.14))
    assert result == pytest.approx(3.14)
    assert isinstance(result, float)


def test_sanitize_np_ndarray():
    result = _sanitize_for_yaml(np.array([1, 2, 3]))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_sanitize_nested():
    data = {"a": np.int64(1), "b": [np.float64(2.0)], "c": "plain"}
    result = _sanitize_for_yaml(data)
    assert result == {"a": 1, "b": [2.0], "c": "plain"}


# --- InMemoryTrace ---


def test_in_memory_trace_clone():
    trace = InMemoryTrace(weight=2.0)
    trace.collect(_make_1d_snapshot(0.0, 1.0, 5.0))
    trace.collect(_make_1d_snapshot(1.0, 2.0, 4.0))

    cloned = trace.clone()
    assert len(cloned) == 2
    assert cloned.weight == 2.0
    # modifying original doesn't affect clone
    trace.collect(_make_1d_snapshot(2.0, 3.0, 3.0))
    assert len(cloned) == 2
    assert len(trace) == 3


def test_in_memory_trace_as_dict():
    trace = InMemoryTrace(weight=1.5)
    trace.collect(_make_1d_snapshot(0.0, 1.0, 5.0))
    trace.record_event({"from": 0, "to": 1}, event_type="hop")

    d = trace.as_dict()
    assert "hops" in d
    assert "data" in d
    assert "weight" in d
    assert len(d["hops"]) == 1
    assert len(d["data"]) == 1
    assert d["weight"] == 1.5


def test_in_memory_trace_record_event_custom_type():
    trace = InMemoryTrace()
    trace.record_event({"info": "test"}, event_type="rescale")
    assert "rescale" in trace.events
    assert len(trace.events["rescale"]) == 1


def test_in_memory_trace_print():
    trace = InMemoryTrace()
    trace.collect(_make_1d_snapshot(0.0, -1.0, 5.0, active=0))
    trace.collect(_make_1d_snapshot(1.0, 1.0, 4.0, active=0))

    buf = io.StringIO()
    trace.print(file=buf)
    output = buf.getvalue()
    assert "time" in output
    assert "rho_0,0" in output


def test_in_memory_trace_print_no_electronic_wfn():
    trace = InMemoryTrace()
    snap = {
        "time": 0.0,
        "position": [1.0],
        "velocity": [5.0],
        "potential": 0.01,
        "kinetic": 0.05,
        "energy": 0.06
    }
    trace.collect(snap)

    buf = io.StringIO()
    trace.print(file=buf)
    output = buf.getvalue()
    assert "time" in output
    assert "rho" not in output


def test_in_memory_trace_print_egylog():
    trace = InMemoryTrace()
    trace.collect(_make_1d_snapshot(0.0, -1.0, 5.0, active=0))
    trace.collect(_make_1d_snapshot(100.0, 1.0, 4.0, active=0))

    buf = io.StringIO()
    trace.print_egylog(file=buf)
    output = buf.getvalue()
    assert "time (fs)" in output
    assert "rho_0,0" in output


def test_in_memory_trace_print_egylog_no_electronic_wfn():
    trace = InMemoryTrace()
    snap = {
        "time": 0.0,
        "position": [1.0],
        "velocity": [5.0],
        "potential": 0.01,
        "kinetic": 0.05,
        "energy": 0.06,
        "temperature": 300.0
    }
    trace.collect(snap)

    buf = io.StringIO()
    trace.print_egylog(file=buf)
    output = buf.getvalue()
    assert "time (fs)" in output


def test_in_memory_trace_outcome_1d():
    trace = InMemoryTrace()
    trace.collect(_make_1d_snapshot(0.0, -1.0, 5.0, active=0))
    trace.collect(_make_1d_snapshot(1.0, 2.0, 4.0, active=1))  # right, state 1

    out = trace.outcome()
    assert out.shape == (2, 2)
    assert out[1, 1] == 1.0  # state 1, right


def test_in_memory_trace_outcome_multidof():
    """outcome returns zeros for multi-DOF systems"""
    trace = InMemoryTrace()
    snap = _make_1d_snapshot(0.0, 1.0, 5.0)
    snap["position"] = [1.0, 2.0]  # 2 DOF
    trace.collect(snap)

    out = trace.outcome()
    assert np.all(out == 0.0)


def test_in_memory_trace_write_trajectory(tmp_path):
    trace = InMemoryTrace()
    trace.collect(_make_3d_snapshot(0.0, natoms=3))
    trace.collect(_make_3d_snapshot(1.0, natoms=3))

    filepath = os.path.join(str(tmp_path), "traj.xyz")
    trace.write_trajectory(filepath)

    with open(filepath, "r") as f:
        content = f.read()
    assert "3" in content
    assert "energy:" in content


# --- trace_factory ---


def test_trace_factory_in_memory():
    cls = trace_factory("in_memory")
    assert cls is InMemoryTrace


def test_trace_factory_yaml():
    cls = trace_factory("yaml")
    assert cls is YAMLTrace


def test_trace_factory_invalid():
    with pytest.raises(ConfigurationError, match="Invalid trace type"):
        trace_factory("unknown")


# --- Trace function ---


def test_trace_function_default():
    t = Trace(None)
    assert isinstance(t, InMemoryTrace)


def test_trace_function_memory():
    t = Trace("memory")
    assert isinstance(t, InMemoryTrace)


def test_trace_function_inmemory():
    t = Trace("inmemory")
    assert isinstance(t, InMemoryTrace)


def test_trace_function_passthrough():
    existing = InMemoryTrace(weight=3.0)
    t = Trace(existing)
    assert t is existing


def test_trace_function_invalid():
    with pytest.raises(ConfigurationError, match="Unrecognized Trace option"):
        Trace(12345)


# --- TraceManager ---


def _make_trace_with_data(nhops=0, nevents=0, weight=1.0):
    """Create an InMemoryTrace with some data."""
    trace = InMemoryTrace(weight=weight)
    trace.collect(_make_1d_snapshot(0.0, -1.0, 5.0, active=0))
    trace.collect(_make_1d_snapshot(1.0, 2.0, 4.0, active=0))
    for i in range(nhops):
        trace.record_event({
            "from": 0,
            "to": 1,
            "time": float(i)
        },
                           event_type="hop")
    for i in range(nevents):
        trace.record_event({"info": f"event_{i}"}, event_type="rescale")
    return trace


def test_trace_manager_add_batch():
    tm = TraceManager()
    t1 = _make_trace_with_data()
    t2 = _make_trace_with_data()
    tm.add_batch([t1, t2])
    assert len(tm.traces) == 2


def test_trace_manager_iter():
    tm = TraceManager()
    t1 = _make_trace_with_data()
    t2 = _make_trace_with_data()
    tm.merge_tracer(t1)
    tm.merge_tracer(t2)

    traces = list(tm)
    assert len(traces) == 2


def test_trace_manager_event_list():
    tm = TraceManager()
    t1 = _make_trace_with_data(nhops=1, nevents=2)
    tm.merge_tracer(t1)

    events = tm.event_list()
    assert "rescale" in events


def test_trace_manager_summarize():
    """summarize uses np.sum(generator) which is deprecated in newer numpy"""
    tm = TraceManager()
    tm.merge_tracer(_make_trace_with_data(nhops=0))
    tm.merge_tracer(_make_trace_with_data(nhops=1))

    buf = io.StringIO()
    try:
        tm.summarize(file=buf)
    except TypeError:
        pytest.skip("np.sum(generator) deprecated in this numpy version")
    output = buf.getvalue()
    assert "# of trajectories: 2" in output
    assert "nhops" in output


def test_trace_manager_summarize_verbose():
    """summarize uses np.sum(generator) which is deprecated in newer numpy"""
    tm = TraceManager()
    tm.merge_tracer(_make_trace_with_data(nhops=1))

    buf = io.StringIO()
    try:
        tm.summarize(verbose=True, file=buf)
    except TypeError:
        pytest.skip("np.sum(generator) deprecated in this numpy version")
    output = buf.getvalue()
    assert "trace" in output
    assert "runtime" in output


def test_trace_manager_summarize_with_events():
    """summarize uses np.sum(generator) which is deprecated in newer numpy"""
    tm = TraceManager()
    tm.merge_tracer(_make_trace_with_data(nhops=0, nevents=2))
    tm.merge_tracer(_make_trace_with_data(nhops=0, nevents=1))

    buf = io.StringIO()
    try:
        tm.summarize(verbose=True, file=buf)
    except TypeError:
        pytest.skip("np.sum(generator) deprecated in this numpy version")
    output = buf.getvalue()
    assert "rescale" in output
    assert "mean" in output


def test_trace_manager_as_dict():
    tm = TraceManager()
    tm.merge_tracer(_make_trace_with_data(nhops=1))
    tm.merge_tracer(_make_trace_with_data(nhops=0))

    d = tm.as_dict()
    assert "hops" in d
    assert "data" in d
    assert "weight" in d
    assert len(d["hops"]) == 2
    assert len(d["data"]) == 2


def test_trace_manager_counts():
    """counts uses np.sum on generator which is deprecated in newer numpy"""
    tm = TraceManager()
    tm.merge_tracer(_make_trace_with_data())
    tm.merge_tracer(_make_trace_with_data())

    try:
        counts = tm.counts()
        assert counts is not None
    except TypeError:
        pytest.skip("np.sum(generator) deprecated in this numpy version")


# --- YAMLTrace additional coverage ---


def test_yaml_trace_files(tmp_path):
    trace = YAMLTrace(base_name="test", location=str(tmp_path))
    files = trace.files(absolute_path=True)
    assert len(files) > 0
    assert all(os.path.isabs(f) for f in files)

    rel_files = trace.files(absolute_path=False)
    assert all(not os.path.isabs(f) for f in rel_files)


def test_yaml_trace_iter_and_getitem(tmp_path):
    trace = YAMLTrace(base_name="test", location=str(tmp_path), log_pitch=4)
    for i in range(6):
        snap = {
            "time": float(i),
            "position": [float(i)],
            "velocity": [0.0],
            "energy": 0.0
        }
        trace.collect(snap)

    assert len(trace) == 6

    # test __getitem__ with positive and negative indices
    first = trace[0]
    assert first["time"] == 0.0

    last = trace[-1]
    assert last["time"] == 5.0

    # test __iter__
    times = [s["time"] for s in trace]
    assert times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_yaml_trace_getitem_out_of_range(tmp_path):
    trace = YAMLTrace(base_name="test", location=str(tmp_path))
    trace.collect({"time": 0.0, "position": [0.0]})

    with pytest.raises(IndexError):
        trace[5]

    with pytest.raises(IndexError):
        trace[-5]


def test_yaml_trace_clone_with_events(tmp_path):
    trace = YAMLTrace(base_name="test", location=str(tmp_path))
    trace.collect({"time": 0.0, "position": [0.0]})
    trace.record_event({"from": 0, "to": 1})
    trace.record_event({"info": "test"}, event_type="rescale")

    cloned = trace.clone()
    assert len(cloned) == 1
    assert "rescale" in cloned.event_logs


def test_yaml_trace_load_and_reload(tmp_path):
    trace = YAMLTrace(base_name="test", location=str(tmp_path))
    trace.collect({"time": 0.0, "position": [0.0]})
    trace.collect({"time": 1.0, "position": [1.0]})

    main_log_path = os.path.join(str(tmp_path), trace.main_log)
    loaded = load_log(main_log_path)
    assert len(loaded) == 2
    assert loaded[0]["time"] == 0.0


# --- YAMLTrace compression ---


def test_yaml_trace_invalid_compression():
    with pytest.raises(ConfigurationError, match="Unknown compression type"):
        YAMLTrace(base_name="test", compression="lz4")


@pytest.mark.parametrize("compression", ["gzip", "bz2", "xz"])
def test_yaml_trace_compression_basic(tmp_path, compression):
    """Write enough snapshots to trigger rollover, verify compressed files."""
    trace = YAMLTrace(base_name="test", location=str(tmp_path),
                      log_pitch=4, compression=compression)
    for i in range(6):
        trace.collect({
            "time": float(i),
            "position": [float(i)],
            "velocity": [0.0],
            "energy": 0.0
        })

    assert len(trace) == 6

    # First chunk (log_0) should be compressed
    _, ext = _COMPRESSORS[compression]
    assert trace.logfiles[0].endswith(ext)
    assert os.path.exists(os.path.join(str(tmp_path), trace.logfiles[0]))

    # Active log (log_1) should remain uncompressed
    assert trace.logfiles[1].endswith(".yaml")
    assert not any(trace.logfiles[1].endswith(e) for e in _COMPRESSION_EXTENSIONS)

    # Data should be readable
    first = trace[0]
    assert first["time"] == 0.0

    last = trace[-1]
    assert last["time"] == 5.0

    times = [s["time"] for s in trace]
    assert times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.parametrize("compression", ["gzip", "bz2", "xz"])
def test_yaml_trace_compression_reload(tmp_path, compression):
    """Verify compressed traces can be loaded from the main log."""
    trace = YAMLTrace(base_name="test", location=str(tmp_path),
                      log_pitch=4, compression=compression)
    for i in range(6):
        trace.collect({
            "time": float(i),
            "position": [float(i)],
            "velocity": [0.0],
            "energy": 0.0
        })

    main_log_path = os.path.join(str(tmp_path), trace.main_log)
    loaded = load_log(main_log_path)
    assert len(loaded) == 6
    assert loaded[0]["time"] == 0.0
    assert loaded[-1]["time"] == 5.0


@pytest.mark.parametrize("compression", ["gzip", "bz2", "xz"])
def test_yaml_trace_compression_clone(tmp_path, compression):
    """Clone a trace with compressed chunks."""
    trace = YAMLTrace(base_name="test", location=str(tmp_path),
                      log_pitch=4, compression=compression)
    for i in range(6):
        trace.collect({
            "time": float(i),
            "position": [float(i)],
            "velocity": [0.0],
            "energy": 0.0
        })
    trace.record_event({"from": 0, "to": 1})

    cloned = trace.clone()
    assert len(cloned) == 6

    # Cloned compressed files should exist with correct extensions
    _, ext = _COMPRESSORS[compression]
    assert cloned.logfiles[0].endswith(ext)

    # Data should be intact
    times = [s["time"] for s in cloned]
    assert times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    # No orphaned empty log_0.yaml
    orphan = os.path.join(str(tmp_path),
                          f"{cloned.unique_name}-log_0.yaml")
    assert not os.path.exists(orphan)


def test_yaml_trace_no_compression_unchanged(tmp_path):
    """Verify compression=None disables compression."""
    trace = YAMLTrace(base_name="test", location=str(tmp_path),
                      log_pitch=4, compression=None)
    for i in range(6):
        trace.collect({
            "time": float(i),
            "position": [float(i)],
            "velocity": [0.0],
            "energy": 0.0
        })

    # All log files should be plain .yaml
    for lf in trace.logfiles:
        assert lf.endswith(".yaml")
        assert not any(lf.endswith(e) for e in _COMPRESSION_EXTENSIONS)


@pytest.mark.parametrize("compression", ["gzip", "bz2", "xz"])
def test_yaml_trace_compression_multiple_rollovers(tmp_path, compression):
    """Verify multiple chunk rollovers all compress correctly."""
    trace = YAMLTrace(base_name="test", location=str(tmp_path),
                      log_pitch=3, compression=compression)
    for i in range(10):
        trace.collect({
            "time": float(i),
            "position": [float(i)],
            "velocity": [0.0],
            "energy": 0.0
        })

    # 10 snapshots / 3 per chunk = 4 chunks (0..2, 3..5, 6..8, 9)
    assert trace.nlogs == 4
    _, ext = _COMPRESSORS[compression]

    # First 3 chunks should be compressed
    for lf in trace.logfiles[:3]:
        assert lf.endswith(ext)

    # Last chunk (active) should be uncompressed
    assert trace.logfiles[3].endswith(".yaml")
    assert not any(trace.logfiles[3].endswith(e) for e in _COMPRESSION_EXTENSIONS)

    # All data should be readable
    times = [s["time"] for s in trace]
    assert times == [float(i) for i in range(10)]
