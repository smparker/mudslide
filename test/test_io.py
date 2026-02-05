#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for mudslide/io.py"""

import unittest
import os
import tempfile
import io

import numpy as np

from mudslide.io import write_xyz, write_trajectory_xyz
from mudslide.constants import bohr_to_angstrom


class TestWriteXYZ(unittest.TestCase):
    """Tests for the write_xyz function"""

    def test_basic_xyz_output(self):
        """Test basic XYZ format output with simple coordinates"""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom_types = ["H", "H", "O"]
        output = io.StringIO()

        write_xyz(coords, atom_types, output, comment="test molecule")

        result = output.getvalue()
        lines = result.strip().split("\n")

        self.assertEqual(lines[0], "3")
        self.assertEqual(lines[1], "test molecule")
        self.assertEqual(len(lines), 5)  # natom + comment + 3 atoms

    def test_bohr_to_angstrom_conversion(self):
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

        self.assertAlmostEqual(x_coord, bohr_to_angstrom, places=10)

    def test_atom_type_capitalization(self):
        """Test that atom types are capitalized correctly"""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        atom_types = ["h", "he"]  # lowercase input
        output = io.StringIO()

        write_xyz(coords, atom_types, output)

        result = output.getvalue()
        lines = result.strip().split("\n")

        # Check atom labels are capitalized
        self.assertTrue(lines[2].startswith("H  "))
        self.assertTrue(lines[3].startswith("He "))

    def test_empty_comment(self):
        """Test with empty comment string"""
        coords = np.array([[0.0, 0.0, 0.0]])
        atom_types = ["C"]
        output = io.StringIO()

        write_xyz(coords, atom_types, output, comment="")

        result = output.getvalue()
        lines = result.split("\n")

        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[1], "")  # empty comment line

    def test_coordinate_precision(self):
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
        self.assertAlmostEqual(float(parts[1]), expected_x, places=10)


class TestWriteTrajectoryXYZ(unittest.TestCase):
    """Tests for the write_trajectory_xyz function"""

    def setUp(self):
        """Create a mock model and trace for testing"""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.tmpdir)

    def _make_mock_model(self, natom=3, ndim=3, atom_types=None):
        """Create a mock model object"""

        class MockModel:

            def __init__(self, natom, ndim, atom_types):
                self.dimensionality = (natom, ndim)
                self.atom_types = atom_types

        return MockModel(natom, ndim, atom_types)

    def _make_trace(self, nframes=5, natom=3, ndim=3):
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

    def test_basic_trajectory_output(self):
        """Test basic trajectory XYZ file output"""
        model = self._make_mock_model(natom=2, ndim=3, atom_types=["H", "O"])
        trace = self._make_trace(nframes=3, natom=2, ndim=3)
        filename = os.path.join(self.tmpdir, "test_traj.xyz")

        write_trajectory_xyz(model, trace, filename)

        self.assertTrue(os.path.exists(filename))

        with open(filename, "r") as f:
            content = f.read()

        lines = content.strip().split("\n")
        # 3 frames * (1 natom + 1 comment + 2 atoms) = 12 lines
        self.assertEqual(len(lines), 12)

    def test_every_parameter(self):
        """Test that 'every' parameter skips frames correctly"""
        model = self._make_mock_model(natom=1, ndim=3, atom_types=["C"])
        trace = self._make_trace(nframes=10, natom=1, ndim=3)
        filename = os.path.join(self.tmpdir, "test_every.xyz")

        write_trajectory_xyz(model, trace, filename, every=3)

        with open(filename, "r") as f:
            content = f.read()

        lines = content.strip().split("\n")
        # Frames 0, 3, 6, 9 should be written (4 frames)
        # Each frame: 1 natom + 1 comment + 1 atom = 3 lines
        self.assertEqual(len(lines), 12)

    def test_fallback_to_x_atom_types(self):
        """Test fallback to 'X' when model.atom_types is None"""
        model = self._make_mock_model(natom=2, ndim=3, atom_types=None)
        trace = self._make_trace(nframes=1, natom=2, ndim=3)
        filename = os.path.join(self.tmpdir, "test_fallback.xyz")

        write_trajectory_xyz(model, trace, filename)

        with open(filename, "r") as f:
            content = f.read()

        lines = content.strip().split("\n")
        # Both atoms should be labeled "X"
        self.assertTrue(lines[2].startswith("X  "))
        self.assertTrue(lines[3].startswith("X  "))

    def test_comment_line_format(self):
        """Test that comment line contains energy and time"""
        model = self._make_mock_model(natom=1, ndim=3, atom_types=["H"])
        trace = [{
            "energy": -123.456,
            "time": 42.0,
            "position": np.array([1.0, 2.0, 3.0])
        }]
        filename = os.path.join(self.tmpdir, "test_comment.xyz")

        write_trajectory_xyz(model, trace, filename)

        with open(filename, "r") as f:
            lines = f.readlines()

        comment = lines[1].strip()
        self.assertIn("E=", comment)
        self.assertIn("t=", comment)
        self.assertIn("-123.456", comment)
        self.assertIn("42", comment)

    def test_position_reshaping(self):
        """Test that flat position array is reshaped correctly"""
        model = self._make_mock_model(natom=2, ndim=3, atom_types=["H", "O"])
        # Position as flat array: [x1, y1, z1, x2, y2, z2]
        trace = [{
            "energy": 0.0,
            "time": 0.0,
            "position": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        }]
        filename = os.path.join(self.tmpdir, "test_reshape.xyz")

        write_trajectory_xyz(model, trace, filename)

        with open(filename, "r") as f:
            lines = f.readlines()

        # First atom should have coords (1, 2, 3) * bohr_to_angstrom
        parts1 = lines[2].split()
        self.assertAlmostEqual(float(parts1[1]),
                               1.0 * bohr_to_angstrom,
                               places=8)
        self.assertAlmostEqual(float(parts1[2]),
                               2.0 * bohr_to_angstrom,
                               places=8)
        self.assertAlmostEqual(float(parts1[3]),
                               3.0 * bohr_to_angstrom,
                               places=8)

        # Second atom should have coords (4, 5, 6) * bohr_to_angstrom
        parts2 = lines[3].split()
        self.assertAlmostEqual(float(parts2[1]),
                               4.0 * bohr_to_angstrom,
                               places=8)
        self.assertAlmostEqual(float(parts2[2]),
                               5.0 * bohr_to_angstrom,
                               places=8)
        self.assertAlmostEqual(float(parts2[3]),
                               6.0 * bohr_to_angstrom,
                               places=8)


if __name__ == '__main__':
    unittest.main()
