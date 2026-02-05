#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for CLI entry points.

These tests verify the CLI wiring (argument parsing, subcommand dispatch)
without re-testing the underlying functionality which is covered elsewhere.
"""

import os
import io
import pytest

from mudslide.mud import mud_main
from mudslide.turboparse.parse import parse
from mudslide.version import get_version_info

testdir = os.path.dirname(__file__)
exampledir = os.path.join(os.path.dirname(testdir), "examples", "turboparse")


class TestMudCLI:
    """Tests for the main mudslide CLI (mud.py)"""

    def test_version_flag(self):
        """Test --version flag exits cleanly with version info"""
        with pytest.raises(SystemExit) as exc_info:
            mud_main(["--version"])
        assert exc_info.value.code == 0

    def test_help_flag(self):
        """Test --help flag exits cleanly"""
        with pytest.raises(SystemExit) as exc_info:
            mud_main(["--help"])
        assert exc_info.value.code == 0

    def test_no_subcommand_fails(self):
        """Test that missing subcommand produces error"""
        with pytest.raises(SystemExit) as exc_info:
            mud_main([])
        assert exc_info.value.code != 0

    def test_invalid_subcommand_fails(self):
        """Test that invalid subcommand produces error"""
        with pytest.raises(SystemExit) as exc_info:
            mud_main(["nonexistent-command"])
        assert exc_info.value.code != 0

    def test_collect_subcommand_recognized(self):
        """Test that collect subcommand is recognized (fails due to missing args)"""
        with pytest.raises(SystemExit) as exc_info:
            mud_main(["collect"])
        # Exits with error because required args are missing, but command is recognized
        assert exc_info.value.code != 0

    def test_collect_help(self):
        """Test collect subcommand help"""
        with pytest.raises(SystemExit) as exc_info:
            mud_main(["collect", "--help"])
        assert exc_info.value.code == 0

    def test_surface_subcommand_recognized(self):
        """Test that surface subcommand is recognized"""
        with pytest.raises(SystemExit) as exc_info:
            mud_main(["surface", "--help"])
        assert exc_info.value.code == 0

    def test_surface_runs(self, tmp_path):
        """Test surface subcommand runs with minimal valid args"""
        outfile = tmp_path / "surface.out"
        mud_main([
            "surface", "-m", "simple", "-r", "-5", "5", "-n", "10",
            f"--output={outfile}"
        ])
        assert outfile.exists()
        content = outfile.read_text()
        assert len(content) > 0

    def test_make_harmonic_subcommand_recognized(self):
        """Test that make_harmonic subcommand is recognized"""
        with pytest.raises(SystemExit) as exc_info:
            mud_main(["make_harmonic", "--help"])
        assert exc_info.value.code == 0

    def test_debug_flag_accepted(self):
        """Test that -d/--debug flag is accepted"""
        with pytest.raises(SystemExit) as exc_info:
            mud_main(["--debug", "--help"])
        assert exc_info.value.code == 0


class TestTurboparseCLI:
    """Tests for the turboparse CLI (turboparse/parse.py)"""

    def test_version_flag(self):
        """Test --version flag exits cleanly"""
        with pytest.raises(SystemExit) as exc_info:
            parse(["--version"])
        assert exc_info.value.code == 0

    def test_help_flag(self):
        """Test --help flag exits cleanly"""
        with pytest.raises(SystemExit) as exc_info:
            parse(["--help"])
        assert exc_info.value.code == 0

    def test_missing_file_arg_fails(self):
        """Test that missing file argument produces error"""
        with pytest.raises(SystemExit) as exc_info:
            parse([])
        assert exc_info.value.code != 0

    def test_nonexistent_file_fails(self):
        """Test that nonexistent file produces error"""
        with pytest.raises(FileNotFoundError):
            parse(["/nonexistent/file.txt"])

    def test_parse_yaml_output(self, capsys):
        """Test parsing a file with YAML output (default)"""
        filepath = os.path.join(exampledir, "H.NAC.txt")
        parse([filepath])
        captured = capsys.readouterr()
        assert "egrad:" in captured.out
        assert "excited_state:" in captured.out

    def test_parse_json_output(self, capsys):
        """Test parsing a file with JSON output"""
        filepath = os.path.join(exampledir, "H.NAC.txt")
        parse([filepath, "--format", "json"])
        captured = capsys.readouterr()
        assert '"egrad":' in captured.out
        assert '"excited_state":' in captured.out

    def test_parse_json_short_flag(self, capsys):
        """Test parsing with -f short flag for format"""
        filepath = os.path.join(exampledir, "H.NAC.txt")
        parse([filepath, "-f", "json"])
        captured = capsys.readouterr()
        assert '"egrad":' in captured.out

    def test_invalid_format_fails(self):
        """Test that invalid format option produces error"""
        filepath = os.path.join(exampledir, "H.NAC.txt")
        with pytest.raises(SystemExit) as exc_info:
            parse([filepath, "--format", "invalid"])
        assert exc_info.value.code != 0
