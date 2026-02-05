# -*- coding: utf-8 -*-
"""Tests for turbo_make_harmonic module"""

import io
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

import mudslide
from mudslide.models.turbomole_model import turbomole_is_installed
from mudslide.config import get_config

testdir = os.path.dirname(__file__)
exampledir = os.path.join(os.path.dirname(testdir), "examples", "make_harmonic_water")


def _turbomole_available():
    return (turbomole_is_installed() or
            "MUDSLIDE_TURBOMOLE_PREFIX" in os.environ or
            get_config("turbomole.command_prefix") is not None)


pytestmark = pytest.mark.skipif(not _turbomole_available(),
                                reason="Turbomole must be installed")


@pytest.fixture
def setup_workdir():
    """Setup a temporary directory with the example control file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the control file
        shutil.copy(os.path.join(exampledir, "control"), tmpdir)
        yield tmpdir


class TestTurboControl:
    """Tests for TurboControl methods used by make_harmonic"""

    def test_read_coords(self, setup_workdir):
        """Test reading coordinates from control file"""
        from mudslide.models.turbomole_model import TurboControl

        control = TurboControl(workdir=setup_workdir)
        symbols, coords = control.read_coords()

        assert symbols == ["o", "h", "h"]
        assert coords.shape == (3, 3)

        # Reference values from control file (in Bohr)
        # O at (0, 0, -0.77235515880385)
        # H at (1.43182237599739, 0, 0.38617757940192)
        # H at (-1.43182237599739, 0, 0.38617757940192)
        expected_coords = np.array([
            [0.00000000000000, 0.00000000000000, -0.77235515880385],
            [1.43182237599739, 0.00000000000000, 0.38617757940192],
            [-1.43182237599739, 0.00000000000000, 0.38617757940192]
        ])
        np.testing.assert_allclose(coords, expected_coords)

    def test_get_masses(self, setup_workdir):
        """Test getting masses from symbols"""
        from mudslide.models.turbomole_model import TurboControl
        from mudslide.periodic_table import masses as atomic_masses
        from mudslide.constants import amu_to_au

        control = TurboControl(workdir=setup_workdir)
        symbols, _ = control.read_coords()
        masses = control.get_masses(symbols)

        assert masses.shape == (9,)

        # O mass is ~15.99 amu, H mass is ~1.008 amu
        # Masses are repeated 3 times for each atom (x, y, z)
        o_mass_au = atomic_masses['o'] * amu_to_au
        h_mass_au = atomic_masses['h'] * amu_to_au

        expected_masses = np.array([
            o_mass_au, o_mass_au, o_mass_au,
            h_mass_au, h_mass_au, h_mass_au,
            h_mass_au, h_mass_au, h_mass_au
        ])
        np.testing.assert_allclose(masses, expected_masses, rtol=1e-10)

    def test_read_hessian(self, setup_workdir):
        """Test reading Hessian from control file"""
        from mudslide.models.turbomole_model import TurboControl

        control = TurboControl(workdir=setup_workdir)
        hessian = control.read_hessian()

        assert hessian.shape == (9, 9)

        # Check symmetry
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-10)

        # Check some reference values from the control file
        # First diagonal element
        assert np.isclose(hessian[0, 0], 0.6154198265)
        # Off-diagonal element
        assert np.isclose(hessian[0, 3], -0.3077099133)
        # Last diagonal element
        assert np.isclose(hessian[8, 8], 0.2102869440)

    def test_hessian_eigenvalues(self, setup_workdir):
        """Test that Hessian has correct eigenvalue structure"""
        from mudslide.models.turbomole_model import TurboControl

        control = TurboControl(workdir=setup_workdir)
        hessian = control.read_hessian()
        eigenvalues = np.linalg.eigvalsh(hessian)
        sorted_eigenvalues = sorted(eigenvalues, key=abs)

        # Should have 6 near-zero eigenvalues (translations + rotations)
        # and 3 positive eigenvalues (vibrations)
        near_zero = [ev for ev in sorted_eigenvalues if abs(ev) < 1e-6]
        positive = [ev for ev in sorted_eigenvalues if ev > 0.01]

        assert len(near_zero) >= 5  # At least 5 near-zero (linear molecule has 5)
        assert len(positive) == 3  # 3 vibrational modes


class TestMakeHarmonicMain:
    """Tests for the make_harmonic_main function"""

    def test_make_harmonic_creates_model(self, setup_workdir):
        """Test that make_harmonic_main creates a valid model file"""
        from mudslide.turbo_make_harmonic import make_harmonic_main

        model_dest = os.path.join(setup_workdir, "test_harmonic.json")
        output = io.StringIO()

        make_harmonic_main(
            control=os.path.join(setup_workdir, "control"),
            model_dest=model_dest,
            output=output
        )

        # Check model file was created
        assert os.path.exists(model_dest)

        # Load and validate the model
        model = mudslide.models.HarmonicModel.from_file(model_dest)

        assert model.nstates == 1
        assert model.ndims == 3
        assert model.nparticles == 3
        assert model.ndof == 9
        assert model.E0 == 0.0

    def test_make_harmonic_geometry(self, setup_workdir):
        """Test that make_harmonic outputs correct geometry"""
        from mudslide.turbo_make_harmonic import make_harmonic_main

        model_dest = os.path.join(setup_workdir, "test_harmonic.json")
        output = io.StringIO()

        make_harmonic_main(
            control=os.path.join(setup_workdir, "control"),
            model_dest=model_dest,
            output=output
        )

        model = mudslide.models.HarmonicModel.from_file(model_dest)

        # Check reference geometry (should match control file coordinates)
        expected_x0 = np.array([
            0.00000000000000, 0.00000000000000, -0.77235515880385,
            1.43182237599739, 0.00000000000000, 0.38617757940192,
            -1.43182237599739, 0.00000000000000, 0.38617757940192
        ])
        np.testing.assert_allclose(model.x0, expected_x0)

    def test_make_harmonic_atom_types(self, setup_workdir):
        """Test that make_harmonic sets correct atom types"""
        from mudslide.turbo_make_harmonic import make_harmonic_main

        model_dest = os.path.join(setup_workdir, "test_harmonic.json")
        output = io.StringIO()

        make_harmonic_main(
            control=os.path.join(setup_workdir, "control"),
            model_dest=model_dest,
            output=output
        )

        model = mudslide.models.HarmonicModel.from_file(model_dest)
        assert model.atom_types == ["o", "h", "h"]

    def test_make_harmonic_hessian(self, setup_workdir):
        """Test that make_harmonic outputs correct Hessian"""
        from mudslide.turbo_make_harmonic import make_harmonic_main

        model_dest = os.path.join(setup_workdir, "test_harmonic.json")
        output = io.StringIO()

        make_harmonic_main(
            control=os.path.join(setup_workdir, "control"),
            model_dest=model_dest,
            output=output
        )

        model = mudslide.models.HarmonicModel.from_file(model_dest)

        # Hessian should be 9x9 and symmetric
        assert model.H0.shape == (9, 9)
        np.testing.assert_allclose(model.H0, model.H0.T, atol=1e-10)

        # Check specific reference values
        assert np.isclose(model.H0[0, 0], 0.6154198265)

    def test_make_harmonic_masses(self, setup_workdir):
        """Test that make_harmonic sets correct masses"""
        from mudslide.turbo_make_harmonic import make_harmonic_main
        from mudslide.periodic_table import masses as atomic_masses
        from mudslide.constants import amu_to_au

        model_dest = os.path.join(setup_workdir, "test_harmonic.json")
        output = io.StringIO()

        make_harmonic_main(
            control=os.path.join(setup_workdir, "control"),
            model_dest=model_dest,
            output=output
        )

        model = mudslide.models.HarmonicModel.from_file(model_dest)

        o_mass_au = atomic_masses['o'] * amu_to_au
        h_mass_au = atomic_masses['h'] * amu_to_au

        expected_masses = np.array([
            o_mass_au, o_mass_au, o_mass_au,
            h_mass_au, h_mass_au, h_mass_au,
            h_mass_au, h_mass_au, h_mass_au
        ])
        np.testing.assert_allclose(model.mass, expected_masses, rtol=1e-10)

    def test_make_harmonic_yaml_output(self, setup_workdir):
        """Test that make_harmonic can output YAML format"""
        from mudslide.turbo_make_harmonic import make_harmonic_main

        model_dest = os.path.join(setup_workdir, "test_harmonic.yaml")
        output = io.StringIO()

        make_harmonic_main(
            control=os.path.join(setup_workdir, "control"),
            model_dest=model_dest,
            output=output
        )

        # Check model file was created
        assert os.path.exists(model_dest)

        # Load and validate
        model = mudslide.models.HarmonicModel.from_file(model_dest)
        assert model.nstates == 1
        assert model.nparticles == 3

    def test_make_harmonic_output_messages(self, setup_workdir):
        """Test that make_harmonic writes expected output messages"""
        from mudslide.turbo_make_harmonic import make_harmonic_main

        model_dest = os.path.join(setup_workdir, "test_harmonic.json")
        output = io.StringIO()

        make_harmonic_main(
            control=os.path.join(setup_workdir, "control"),
            model_dest=model_dest,
            output=output
        )

        output_text = output.getvalue()

        # Check for expected output sections
        assert "Reading Turbomole control file" in output_text
        assert "Reference geometry:" in output_text
        assert "Hessian loaded with eigenvalues" in output_text
        assert "Writing harmonic model" in output_text


class TestMakeHarmonicModelUsage:
    """Test that the generated harmonic model can be used correctly"""

    def test_harmonic_model_compute(self, setup_workdir):
        """Test computing energy and forces with the harmonic model"""
        from mudslide.turbo_make_harmonic import make_harmonic_main

        model_dest = os.path.join(setup_workdir, "test_harmonic.json")
        output = io.StringIO()

        make_harmonic_main(
            control=os.path.join(setup_workdir, "control"),
            model_dest=model_dest,
            output=output
        )

        model = mudslide.models.HarmonicModel.from_file(model_dest)

        # At equilibrium, energy should be 0 and force should be 0
        model.compute(model.x0)
        assert np.isclose(model.energies[0], 0.0)
        np.testing.assert_allclose(model.force(0), np.zeros(9), atol=1e-10)

    def test_harmonic_model_displaced(self, setup_workdir):
        """Test energy at displaced geometry"""
        from mudslide.turbo_make_harmonic import make_harmonic_main

        model_dest = os.path.join(setup_workdir, "test_harmonic.json")
        output = io.StringIO()

        make_harmonic_main(
            control=os.path.join(setup_workdir, "control"),
            model_dest=model_dest,
            output=output
        )

        model = mudslide.models.HarmonicModel.from_file(model_dest)

        # Displace from equilibrium
        x = np.copy(model.x0)
        x[0] += 0.1  # Small displacement in x

        model.compute(x)

        # Energy should be positive
        assert model.energies[0] > 0

        # Force should be non-zero and restoring (opposite sign of displacement)
        force = model.force(0)
        assert force[0] < 0  # Restoring force
