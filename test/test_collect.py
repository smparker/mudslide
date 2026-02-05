#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for mudslide/collect.py"""

import unittest
import os
import tempfile
import shutil
import argparse

import yaml

from mudslide.collect import add_collect_parser, collect, collect_wrapper, legend, legend_format


class TestAddCollectParser(unittest.TestCase):
    """Tests for the add_collect_parser function"""

    def test_parser_added_to_subparsers(self):
        """Test that collect parser is added correctly"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        add_collect_parser(subparsers)

        # Parse a collect command
        args = parser.parse_args(['collect', 'mylog'])
        self.assertEqual(args.logname, 'mylog')
        self.assertEqual(args.keys, 'tkpea')  # default value

    def test_custom_keys(self):
        """Test that custom keys can be specified"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        add_collect_parser(subparsers)

        args = parser.parse_args(['collect', 'mylog', '-k', 'te'])
        self.assertEqual(args.keys, 'te')

    def test_func_set_to_collect_wrapper(self):
        """Test that func is set to collect_wrapper"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        add_collect_parser(subparsers)

        args = parser.parse_args(['collect', 'mylog'])
        self.assertEqual(args.func, collect_wrapper)


class TestCollect(unittest.TestCase):
    """Tests for the collect function"""

    def setUp(self):
        """Create temporary directory and YAML trace files for testing"""
        self.tmpdir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        """Clean up temporary directory"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.tmpdir)

    def _create_yaml_trace(self, basename, snapshots):
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

    def test_collect_basic(self):
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
        main_log = self._create_yaml_trace("test-traj-0", snapshots)

        collect(main_log)

        output_file = main_log + ".dat"
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, "r") as f:
            lines = f.readlines()

        # Check header
        self.assertTrue(lines[0].startswith("#"))
        self.assertIn("time", lines[0])
        self.assertIn("kinetic", lines[0])
        self.assertIn("potential", lines[0])
        self.assertIn("energy", lines[0])
        self.assertIn("active", lines[0])

        # Check data lines (should be 3)
        self.assertEqual(len(lines), 4)  # 1 header + 3 data

    def test_collect_custom_keys(self):
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
        main_log = self._create_yaml_trace("test-traj-0", snapshots)

        collect(main_log, keys="te")

        output_file = main_log + ".dat"
        with open(output_file, "r") as f:
            lines = f.readlines()

        # Header should only have time and energy
        header = lines[0]
        self.assertIn("time", header)
        self.assertIn("energy", header)
        self.assertNotIn("kinetic", header)
        self.assertNotIn("potential", header)
        self.assertNotIn("active", header)

    def test_collect_output_values(self):
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
        main_log = self._create_yaml_trace("test-traj-0", snapshots)

        collect(main_log)

        output_file = main_log + ".dat"
        with open(output_file, "r") as f:
            lines = f.readlines()

        # Parse data line
        data_line = lines[1].strip().split()
        self.assertAlmostEqual(float(data_line[0]), 1.5, places=6)
        self.assertAlmostEqual(float(data_line[1]), 0.123, places=6)
        self.assertAlmostEqual(float(data_line[2]), -0.456, places=6)
        self.assertAlmostEqual(float(data_line[3]), -0.333, places=6)
        self.assertEqual(int(data_line[4]), 2)


class TestCollectWrapper(unittest.TestCase):
    """Tests for the collect_wrapper function"""

    def setUp(self):
        """Create temporary directory for testing"""
        self.tmpdir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        """Clean up temporary directory"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.tmpdir)

    def _create_yaml_trace(self, basename, snapshots):
        """Create a minimal YAML trace structure for testing."""
        log_file = f"{basename}-log_0.yaml"
        hop_file = f"{basename}-hops.yaml"
        main_log = f"{basename}.yaml"

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

        with open(log_file, "w") as f:
            yaml.dump(snapshots, f)

        with open(hop_file, "w") as f:
            pass

        return main_log

    def test_wrapper_calls_collect(self):
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
        main_log = self._create_yaml_trace("test-traj-0", snapshots)

        # Create mock args object
        class MockArgs:
            logname = main_log
            keys = "te"

        collect_wrapper(MockArgs())

        output_file = main_log + ".dat"
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, "r") as f:
            content = f.read()

        self.assertIn("time", content)
        self.assertIn("energy", content)


class TestLegendMappings(unittest.TestCase):
    """Tests for the legend and legend_format dictionaries"""

    def test_legend_keys_match_format_keys(self):
        """Test that legend and legend_format have the same keys"""
        self.assertEqual(set(legend.keys()), set(legend_format.keys()))

    def test_expected_keys_present(self):
        """Test that expected keys are present in legend"""
        expected_keys = ['t', 'k', 'p', 'e', 'a']
        for key in expected_keys:
            self.assertIn(key, legend)

    def test_legend_values(self):
        """Test legend mappings are correct"""
        self.assertEqual(legend['t'], 'time')
        self.assertEqual(legend['k'], 'kinetic')
        self.assertEqual(legend['p'], 'potential')
        self.assertEqual(legend['e'], 'energy')
        self.assertEqual(legend['a'], 'active')


if __name__ == '__main__':
    unittest.main()
