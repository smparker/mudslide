#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for mudslide/collect.py"""

import os
import argparse

import pytest
import yaml

from mudslide.collect import add_collect_parser, collect, collect_wrapper, legend, legend_format


def test_parser_added_to_subparsers():
    """Test that collect parser is added correctly"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    add_collect_parser(subparsers)

    # Parse a collect command
    args = parser.parse_args(['collect', 'mylog'])
    assert args.logname == 'mylog'
    assert args.keys == 'tkpea'  # default value


def test_custom_keys():
    """Test that custom keys can be specified"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    add_collect_parser(subparsers)

    args = parser.parse_args(['collect', 'mylog', '-k', 'te'])
    assert args.keys == 'te'


def test_func_set_to_collect_wrapper():
    """Test that func is set to collect_wrapper"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    add_collect_parser(subparsers)

    args = parser.parse_args(['collect', 'mylog'])
    assert args.func == collect_wrapper


def _create_yaml_trace(basename, snapshots):
    """Create a minimal YAML trace structure for testing.

    Creates the main log file and associated data log file.
    """
    log_file = f"{basename}-log_0.yaml"
    hop_file = f"{basename}-hops.yaml"
    main_log = f"{basename}.yaml"

    # Write main log
    main_data = {
        "name": basename,
        "logfiles": [log_file],
        "nlogs": 1,
        "log_pitch": 512,
        "hop_log": hop_file,
        "event_logs": {},
        "weight": 1.0
    }
    with open(main_log, "w") as f:
        yaml.dump(main_data, f)

    # Write data log with snapshots
    with open(log_file, "w") as f:
        yaml.dump(snapshots, f)

    # Write empty hop log
    with open(hop_file, "w") as f:
        pass

    return main_log


@pytest.fixture()
def working_dir(tmp_path, monkeypatch):
    """Change to a temporary directory for tests that need file I/O"""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_collect_basic(working_dir):
    """Test basic collect functionality with default keys"""
    snapshots = [
        {
            "time": 0.0,
            "kinetic": 0.5,
            "potential": -1.0,
            "energy": -0.5,
            "active": 0
        },
        {
            "time": 1.0,
            "kinetic": 0.6,
            "potential": -1.1,
            "energy": -0.5,
            "active": 0
        },
        {
            "time": 2.0,
            "kinetic": 0.7,
            "potential": -1.2,
            "energy": -0.5,
            "active": 1
        },
    ]
    main_log = _create_yaml_trace("test-traj-0", snapshots)

    collect(main_log)

    output_file = main_log + ".dat"
    assert os.path.exists(output_file)

    with open(output_file, "r") as f:
        lines = f.readlines()

    # Check header
    assert lines[0].startswith("#")
    assert "time" in lines[0]
    assert "kinetic" in lines[0]
    assert "potential" in lines[0]
    assert "energy" in lines[0]
    assert "active" in lines[0]

    # Check data lines (should be 3)
    assert len(lines) == 4  # 1 header + 3 data


def test_collect_custom_keys(working_dir):
    """Test collect with subset of keys"""
    snapshots = [
        {
            "time": 0.0,
            "kinetic": 0.5,
            "potential": -1.0,
            "energy": -0.5,
            "active": 0
        },
        {
            "time": 1.0,
            "kinetic": 0.6,
            "potential": -1.1,
            "energy": -0.5,
            "active": 1
        },
    ]
    main_log = _create_yaml_trace("test-traj-0", snapshots)

    collect(main_log, keys="te")

    output_file = main_log + ".dat"
    with open(output_file, "r") as f:
        lines = f.readlines()

    # Header should only have time and energy
    header = lines[0]
    assert "time" in header
    assert "energy" in header
    assert "kinetic" not in header
    assert "potential" not in header
    assert "active" not in header


def test_collect_output_values(working_dir):
    """Test that output values match input data"""
    snapshots = [
        {
            "time": 1.5,
            "kinetic": 0.123,
            "potential": -0.456,
            "energy": -0.333,
            "active": 2
        },
    ]
    main_log = _create_yaml_trace("test-traj-0", snapshots)

    collect(main_log)

    output_file = main_log + ".dat"
    with open(output_file, "r") as f:
        lines = f.readlines()

    # Parse data line
    data_line = lines[1].strip().split()
    assert float(data_line[0]) == pytest.approx(1.5, abs=1e-6)
    assert float(data_line[1]) == pytest.approx(0.123, abs=1e-6)
    assert float(data_line[2]) == pytest.approx(-0.456, abs=1e-6)
    assert float(data_line[3]) == pytest.approx(-0.333, abs=1e-6)
    assert int(data_line[4]) == 2


def test_wrapper_calls_collect(working_dir):
    """Test that wrapper correctly calls collect with args"""
    snapshots = [
        {
            "time": 0.0,
            "kinetic": 0.5,
            "potential": -1.0,
            "energy": -0.5,
            "active": 0
        },
    ]
    main_log = _create_yaml_trace("test-traj-0", snapshots)

    # Create mock args object
    class MockArgs:
        logname = main_log
        keys = "te"

    collect_wrapper(MockArgs())

    output_file = main_log + ".dat"
    assert os.path.exists(output_file)

    with open(output_file, "r") as f:
        content = f.read()

    assert "time" in content
    assert "energy" in content


def test_legend_keys_match_format_keys():
    """Test that legend and legend_format have the same keys"""
    assert set(legend.keys()) == set(legend_format.keys())


def test_expected_keys_present():
    """Test that expected keys are present in legend"""
    expected_keys = ['t', 'k', 'p', 'e', 'a']
    for key in expected_keys:
        assert key in legend


def test_legend_values():
    """Test legend mappings are correct"""
    assert legend['t'] == 'time'
    assert legend['k'] == 'kinetic'
    assert legend['p'] == 'potential'
    assert legend['e'] == 'energy'
    assert legend['a'] == 'active'
