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


class TestMainCLI:
    """Tests for mudslide.__main__ CLI options.

    These tests cover CLI argument paths that weren't previously exercised.
    """

    def test_published_simple_model(self, tmp_path):
        """Test --published flag with simple model"""
        import mudslide.__main__
        outfile = tmp_path / "output.txt"
        with open(outfile, "w") as f:
            mudslide.__main__.main([
                "-m", "simple", "--published", "-n", "1", "-s", "1",
                "-T", "100", "-z", "42", "-o", "averaged"
            ], file=f)
        content = outfile.read_text()
        assert len(content) > 0
        # Check output has expected columns (momentum + state outcomes)
        lines = [l for l in content.strip().split('\n') if not l.startswith('#')]
        assert len(lines) == 1

    def test_published_dual_model(self, tmp_path):
        """Test --published flag with dual model"""
        import mudslide.__main__
        outfile = tmp_path / "output.txt"
        with open(outfile, "w") as f:
            mudslide.__main__.main([
                "-m", "dual", "--published", "-n", "1", "-s", "1",
                "-T", "100", "-z", "42", "-o", "averaged"
            ], file=f)
        content = outfile.read_text()
        lines = [l for l in content.strip().split('\n') if not l.startswith('#')]
        assert len(lines) == 1

    def test_published_extended_model(self, tmp_path):
        """Test --published flag with extended model"""
        import mudslide.__main__
        outfile = tmp_path / "output.txt"
        with open(outfile, "w") as f:
            mudslide.__main__.main([
                "-m", "extended", "--published", "-n", "1", "-s", "1",
                "-T", "100", "-z", "42", "-o", "averaged"
            ], file=f)
        content = outfile.read_text()
        lines = [l for l in content.strip().split('\n') if not l.startswith('#')]
        assert len(lines) == 1

    def test_published_super_model(self, tmp_path):
        """Test --published flag with super model"""
        import mudslide.__main__
        outfile = tmp_path / "output.txt"
        with open(outfile, "w") as f:
            mudslide.__main__.main([
                "-m", "super", "--published", "-n", "1", "-s", "1",
                "-T", "100", "-z", "42", "-o", "averaged"
            ], file=f)
        content = outfile.read_text()
        lines = [l for l in content.strip().split('\n') if not l.startswith('#')]
        assert len(lines) == 1

    def test_published_unknown_model_warning(self, tmp_path, capsys):
        """Test --published flag with model that has no published bounds"""
        import mudslide.__main__
        import sys
        outfile = tmp_path / "output.txt"
        with open(outfile, "w") as f:
            mudslide.__main__.main([
                "-m", "shin-metiu", "--published", "-n", "1", "-s", "1",
                "-T", "100", "-z", "42", "-o", "averaged"
            ], file=f)
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_kspacing_log(self, tmp_path):
        """Test --kspacing log option"""
        import mudslide.__main__
        outfile = tmp_path / "output.txt"
        with open(outfile, "w") as f:
            # Using log spacing with small range
            mudslide.__main__.main([
                "-m", "simple", "-n", "2", "-s", "1", "-l", "log",
                "-k", "0.5", "1.0",  # log10 range
                "-T", "100", "-z", "42", "-o", "averaged"
            ], file=f)
        content = outfile.read_text()
        lines = [l for l in content.strip().split('\n') if not l.startswith('#')]
        assert len(lines) == 2

    def test_ksampling_normal(self, tmp_path):
        """Test --ksampling normal option"""
        import mudslide.__main__
        outfile = tmp_path / "output.txt"
        with open(outfile, "w") as f:
            mudslide.__main__.main([
                "-m", "simple", "-n", "1", "-s", "2",
                "-K", "normal", "-f", "5.0",  # normal sampling with std dev
                "-k", "10", "10", "-T", "100", "-z", "42", "-o", "averaged"
            ], file=f)
        content = outfile.read_text()
        lines = [l for l in content.strip().split('\n') if not l.startswith('#')]
        assert len(lines) == 1

    def test_output_swarm(self, tmp_path, monkeypatch):
        """Test --output swarm option"""
        import mudslide.__main__
        # Change to tmp_path so swarm files are written there
        monkeypatch.chdir(tmp_path)
        mudslide.__main__.main([
            "-m", "simple", "-n", "1", "-s", "2",
            "-k", "10", "10", "-T", "50", "-z", "42", "-o", "swarm"
        ])
        # Check that state files were created
        assert (tmp_path / "state_0.trace").exists()
        assert (tmp_path / "state_1.trace").exists()
        # Verify content is valid (positions are formatted correctly)
        content = (tmp_path / "state_0.trace").read_text()
        lines = [l for l in content.strip().split('\n') if l.strip()]
        assert len(lines) > 0
        # Each line should be a valid float
        for line in lines:
            if float(line) != -9999999:  # Skip placeholder lines
                float(line)  # Should not raise

    def test_output_pickle(self, tmp_path):
        """Test --output pickle option"""
        import mudslide.__main__
        import pickle
        outfile = tmp_path / "output.txt"
        pickle_file = tmp_path / "results.pickle"
        with open(outfile, "w") as f:
            mudslide.__main__.main([
                "-m", "simple", "-n", "1", "-s", "1",
                "-k", "10", "10", "-T", "100", "-z", "42",
                "-o", "pickle", "-O", str(pickle_file)
            ], file=f)
        assert pickle_file.exists()
        with open(pickle_file, "rb") as pf:
            results = pickle.load(pf)
        assert len(results) == 1
        assert results[0][0] == 10.0  # momentum value

    def test_output_hack(self, tmp_path, capsys):
        """Test --output hack option"""
        import mudslide.__main__
        outfile = tmp_path / "output.txt"
        with open(outfile, "w") as f:
            mudslide.__main__.main([
                "-m", "simple", "-n", "1", "-s", "1",
                "-k", "10", "10", "-T", "100", "-z", "42", "-o", "hack"
            ], file=f)
        content = outfile.read_text()
        assert "Hack something here" in content
