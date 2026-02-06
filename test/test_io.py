#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for mudslide/io.py"""

import os
import io

import numpy as np
import pytest

from mudslide.io import write_xyz, write_trajectory_xyz
from mudslide.constants import bohr_to_angstrom


def test_basic_xyz_output():
    """Test basic XYZ format output with simple coordinates"""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    atom_types = ["H", "H", "O"]
    output = io.StringIO()

    write_xyz(coords, atom_types, output, comment="test molecule")

    result = output.getvalue()
    lines = result.strip().split("\n")

    assert lines[0] == "3"
    assert lines[1] == "test molecule"
    assert len(lines) == 5  # natom + comment + 3 atoms


def test_bohr_to_angstrom_conversion():
    """Test that coordinates are converted from bohr to angstrom"""
    # 1 bohr should become ~0.529 angstrom
    coords = np.array([[1.0, 0.0, 0.0]])
    atom_types = ["H"]
    output = io.StringIO()

    write_xyz(coords, atom_types, output)

    result = output.getvalue()
    lines = result.strip().split("\n")
    # Parse the coordinate line
    parts = lines[2].split()
    x_coord = float(parts[1])

    assert x_coord == pytest.approx(bohr_to_angstrom, abs=1e-10)


def test_atom_type_capitalization():
    """Test that atom types are capitalized correctly"""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atom_types = ["h", "he"]  # lowercase input
    output = io.StringIO()

    write_xyz(coords, atom_types, output)

    result = output.getvalue()
    lines = result.strip().split("\n")

    # Check atom labels are capitalized
    assert lines[2].startswith("H  ")
    assert lines[3].startswith("He ")


def test_empty_comment():
    """Test with empty comment string"""
    coords = np.array([[0.0, 0.0, 0.0]])
    atom_types = ["C"]
    output = io.StringIO()

    write_xyz(coords, atom_types, output, comment="")

    result = output.getvalue()
    lines = result.split("\n")

    assert lines[0] == "1"
    assert lines[1] == ""  # empty comment line


def test_coordinate_precision():
    """Test that coordinates are written with proper precision"""
    coords = np.array([[1.123456789012, 2.234567890123, 3.345678901234]])
    atom_types = ["N"]
    output = io.StringIO()

    write_xyz(coords, atom_types, output)

    result = output.getvalue()
    lines = result.strip().split("\n")
    parts = lines[2].split()

    # Check we have 12 decimal places (20.12f format)
    # The actual values are converted to angstrom
    expected_x = 1.123456789012 * bohr_to_angstrom
    assert float(parts[1]) == pytest.approx(expected_x, abs=1e-10)


def _make_mock_model(natom=3, ndim=3, atom_types=None):
    """Create a mock model object"""

    class MockModel:

        def __init__(self, natom, ndim, atom_types):
            self.dimensionality = (natom, ndim)
            self.atom_types = atom_types

    return MockModel(natom, ndim, atom_types)


def _make_trace(nframes=5, natom=3, ndim=3):
    """Create a mock trace (list of frame dictionaries)"""
    trace = []
    for i in range(nframes):
        frame = {
            "energy": -100.0 + i * 0.1,
            "time": i * 10.0,
            "position": np.random.randn(natom * ndim)
        }
        trace.append(frame)
    return trace


def test_basic_trajectory_output(tmp_path):
    """Test basic trajectory XYZ file output"""
    model = _make_mock_model(natom=2, ndim=3, atom_types=["H", "O"])
    trace = _make_trace(nframes=3, natom=2, ndim=3)
    filename = os.path.join(str(tmp_path), "test_traj.xyz")

    write_trajectory_xyz(model, trace, filename)

    assert os.path.exists(filename)

    with open(filename, "r") as f:
        content = f.read()

    lines = content.strip().split("\n")
    # 3 frames * (1 natom + 1 comment + 2 atoms) = 12 lines
    assert len(lines) == 12


def test_every_parameter(tmp_path):
    """Test that 'every' parameter skips frames correctly"""
    model = _make_mock_model(natom=1, ndim=3, atom_types=["C"])
    trace = _make_trace(nframes=10, natom=1, ndim=3)
    filename = os.path.join(str(tmp_path), "test_every.xyz")

    write_trajectory_xyz(model, trace, filename, every=3)

    with open(filename, "r") as f:
        content = f.read()

    lines = content.strip().split("\n")
    # Frames 0, 3, 6, 9 should be written (4 frames)
    # Each frame: 1 natom + 1 comment + 1 atom = 3 lines
    assert len(lines) == 12


def test_fallback_to_x_atom_types(tmp_path):
    """Test fallback to 'X' when model.atom_types is None"""
    model = _make_mock_model(natom=2, ndim=3, atom_types=None)
    trace = _make_trace(nframes=1, natom=2, ndim=3)
    filename = os.path.join(str(tmp_path), "test_fallback.xyz")

    write_trajectory_xyz(model, trace, filename)

    with open(filename, "r") as f:
        content = f.read()

    lines = content.strip().split("\n")
    # Both atoms should be labeled "X"
    assert lines[2].startswith("X  ")
    assert lines[3].startswith("X  ")


def test_comment_line_format(tmp_path):
    """Test that comment line contains energy and time"""
    model = _make_mock_model(natom=1, ndim=3, atom_types=["H"])
    trace = [{
        "energy": -123.456,
        "time": 42.0,
        "position": np.array([1.0, 2.0, 3.0])
    }]
    filename = os.path.join(str(tmp_path), "test_comment.xyz")

    write_trajectory_xyz(model, trace, filename)

    with open(filename, "r") as f:
        lines = f.readlines()

    comment = lines[1].strip()
    assert "E=" in comment
    assert "t=" in comment
    assert "-123.456" in comment
    assert "42" in comment


def test_position_reshaping(tmp_path):
    """Test that flat position array is reshaped correctly"""
    model = _make_mock_model(natom=2, ndim=3, atom_types=["H", "O"])
    # Position as flat array: [x1, y1, z1, x2, y2, z2]
    trace = [{
        "energy": 0.0,
        "time": 0.0,
        "position": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }]
    filename = os.path.join(str(tmp_path), "test_reshape.xyz")

    write_trajectory_xyz(model, trace, filename)

    with open(filename, "r") as f:
        lines = f.readlines()

    # First atom should have coords (1, 2, 3) * bohr_to_angstrom
    parts1 = lines[2].split()
    assert float(parts1[1]) == pytest.approx(1.0 * bohr_to_angstrom, abs=1e-8)
    assert float(parts1[2]) == pytest.approx(2.0 * bohr_to_angstrom, abs=1e-8)
    assert float(parts1[3]) == pytest.approx(3.0 * bohr_to_angstrom, abs=1e-8)

    # Second atom should have coords (4, 5, 6) * bohr_to_angstrom
    parts2 = lines[3].split()
    assert float(parts2[1]) == pytest.approx(4.0 * bohr_to_angstrom, abs=1e-8)
    assert float(parts2[2]) == pytest.approx(5.0 * bohr_to_angstrom, abs=1e-8)
    assert float(parts2[3]) == pytest.approx(6.0 * bohr_to_angstrom, abs=1e-8)
