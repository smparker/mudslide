#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for header"""

import unittest
import re
from io import StringIO
from unittest.mock import patch

import mudslide
from mudslide.header import print_header, BANNER


class TestHeader(unittest.TestCase):
    """Test Suite for header functions"""

    def _capture_header(self) -> str:
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print_header()
        return mock_stdout.getvalue()

    def test_print_header_runs(self) -> None:
        """print_header should run without error"""
        output = self._capture_header()
        self.assertIn("MUDSLIDE", output)

    def test_header_contains_version(self) -> None:
        """print_header should contain the mudslide version"""
        output = self._capture_header()
        self.assertIn(mudslide.__version__, output)

    def test_header_contains_python_version(self) -> None:
        """print_header should contain the Python version"""
        import sys
        output = self._capture_header()
        self.assertIn(sys.version.split()[0], output)

    def test_header_contains_numpy_version(self) -> None:
        """print_header should contain the NumPy version"""
        import numpy
        output = self._capture_header()
        self.assertIn(numpy.__version__, output)

    def test_header_contains_scipy_version(self) -> None:
        """print_header should contain the SciPy version"""
        import scipy
        output = self._capture_header()
        self.assertIn(scipy.__version__, output)

    def test_header_contains_platform(self) -> None:
        """print_header should contain the platform"""
        import platform
        output = self._capture_header()
        self.assertIn(platform.platform(), output)

    def test_header_contains_path(self) -> None:
        """print_header should contain the install path"""
        from mudslide.version import get_install_path
        output = self._capture_header()
        self.assertIn(get_install_path(), output)

    def test_header_contains_date(self) -> None:
        """print_header should contain a date string"""
        output = self._capture_header()
        self.assertRegex(output, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_header_contains_banner(self) -> None:
        """print_header should contain the ASCII banner"""
        output = self._capture_header()
        self.assertIn(".-'~~~`-.", output)

    def test_accessible_from_mudslide(self) -> None:
        """print_header should be accessible as mudslide.print_header"""
        self.assertIs(mudslide.print_header, print_header)


if __name__ == '__main__':
    unittest.main()
