#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for header"""

import re
from io import StringIO
from unittest.mock import patch

import mudslide
from mudslide.header import print_header, BANNER


def _capture_header() -> str:
    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        print_header()
    return mock_stdout.getvalue()


def test_print_header_runs() -> None:
    """print_header should run without error"""
    output = _capture_header()
    assert "MUDSLIDE" in output


def test_header_contains_version() -> None:
    """print_header should contain the mudslide version"""
    output = _capture_header()
    assert mudslide.__version__ in output


def test_header_contains_python_version() -> None:
    """print_header should contain the Python version"""
    import sys
    output = _capture_header()
    assert sys.version.split()[0] in output


def test_header_contains_numpy_version() -> None:
    """print_header should contain the NumPy version"""
    import numpy
    output = _capture_header()
    assert numpy.__version__ in output


def test_header_contains_scipy_version() -> None:
    """print_header should contain the SciPy version"""
    import scipy
    output = _capture_header()
    assert scipy.__version__ in output


def test_header_contains_platform() -> None:
    """print_header should contain the platform"""
    import platform
    output = _capture_header()
    assert platform.platform() in output


def test_header_contains_path() -> None:
    """print_header should contain the install path"""
    from mudslide.version import get_install_path
    output = _capture_header()
    assert get_install_path() in output


def test_header_contains_date() -> None:
    """print_header should contain a date string"""
    output = _capture_header()
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", output)


def test_header_contains_banner() -> None:
    """print_header should contain the ASCII banner"""
    output = _capture_header()
    assert ".-'~~~`-." in output


def test_accessible_from_mudslide() -> None:
    """print_header should be accessible as mudslide.print_header"""
    assert mudslide.print_header is print_header
