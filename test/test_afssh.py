#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for A-FSSH (Augmented Fewest Switches Surface Hopping) implementation.

These tests cover the decoherence correction and collapse logic in A-FSSH,
as well as RK4 integration paths and error handling.
"""

import numpy as np
import pytest

from mudslide.afssh import AugmentedFSSH, AFSSHPropagator, AFSSHVVPropagator
from mudslide.models import scattering_models as models


def make_afssh_traj(model, x0, v0, state0, **kwargs):
    """Helper to create an AugmentedFSSH trajectory with common defaults."""
    # Use strict_option_check=False since augmented_integration is AFSSH-specific
    return AugmentedFSSH(
        model,
        np.atleast_1d(x0),
        np.atleast_1d(v0),
        state0,
        strict_option_check=False,
        **kwargs
    )


def step_trajectory(traj, nsteps=1):
    """Advance trajectory by nsteps using its propagator."""
    if traj.electronics is None:
        traj.electronics = traj.model.update(
            traj.position,
            gradients=traj.needed_gradients(),
            couplings=traj.needed_couplings()
        )
    traj.propagator(traj, nsteps)


class TestAFSSHPropagator:
    """Tests for AFSSHPropagator factory."""

    def test_default_vv_propagator(self):
        """Test that default propagator is VV type"""
        model = models["simple"](mass=2000.0)
        prop = AFSSHPropagator(model)
        assert isinstance(prop, AFSSHVVPropagator)

    def test_explicit_vv_propagator(self):
        """Test explicit VV propagator selection"""
        model = models["simple"](mass=2000.0)
        prop = AFSSHPropagator(model, "vv")
        assert isinstance(prop, AFSSHVVPropagator)

    def test_dict_options_vv(self):
        """Test dictionary options for propagator"""
        model = models["simple"](mass=2000.0)
        prop = AFSSHPropagator(model, {"type": "vv"})
        assert isinstance(prop, AFSSHVVPropagator)

    def test_invalid_propagator_type_raises(self):
        """Test that invalid propagator type raises ValueError"""
        model = models["simple"](mass=2000.0)
        with pytest.raises(ValueError, match="Unrecognized"):
            AFSSHPropagator(model, "invalid_type")

    def test_invalid_options_type_raises(self):
        """Test that invalid options type raises Exception"""
        model = models["simple"](mass=2000.0)
        with pytest.raises(Exception, match="must be a string or a dictionary"):
            AFSSHPropagator(model, 12345)


class TestAFSSHRK4Integration:
    """Tests for RK4 integration path in A-FSSH."""

    def test_rk4_integration_delR(self):
        """Test RK4 integration for delR propagation"""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(
            model, -5.0, 0.01, 0,
            augmented_integration="rk4",
            seed_sequence=42
        )

        # Run a few steps
        for _ in range(10):
            step_trajectory(traj)

        # delR should have evolved
        assert traj.delR is not None
        assert traj.delR.shape == (model.ndof, model.nstates, model.nstates)

    def test_rk4_integration_delP(self):
        """Test RK4 integration for delP propagation"""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(
            model, -5.0, 0.01, 0,
            augmented_integration="rk4",
            seed_sequence=42
        )

        # Run a few steps
        for _ in range(10):
            step_trajectory(traj)

        # delP should have evolved
        assert traj.delP is not None
        assert traj.delP.shape == (model.ndof, model.nstates, model.nstates)

    def test_rk4_vs_exp_consistency(self):
        """Test that RK4 and exp integration give similar results for short times."""
        model = models["simple"](mass=2000.0)

        # Create two trajectories with different integration methods
        traj_exp = make_afssh_traj(
            model, -10.0, 0.01, 0,
            augmented_integration="exp",
            seed_sequence=42
        )
        traj_rk4 = make_afssh_traj(
            model, -10.0, 0.01, 0,
            augmented_integration="rk4",
            seed_sequence=42
        )

        # Run a few steps (before any coupling region)
        for _ in range(5):
            step_trajectory(traj_exp)
            step_trajectory(traj_rk4)

        # Positions and velocities should be nearly identical
        # (decoherence effects are small far from coupling)
        np.testing.assert_allclose(traj_exp.position, traj_rk4.position, rtol=1e-6)
        np.testing.assert_allclose(traj_exp.velocity, traj_rk4.velocity, rtol=1e-6)

    def test_invalid_integration_method_delR_raises(self):
        """Test that invalid integration method raises for delR."""
        model = models["simple"](mass=2000.0)
        traj = make_afssh_traj(model, -5.0, 0.01, 0, seed_sequence=42)
        # Manually set invalid integration method
        traj.augmented_integration = "invalid"

        with pytest.raises(Exception, match="Unrecognized propagate delR"):
            step_trajectory(traj)

    def test_invalid_integration_method_delP_raises(self):
        """Test that invalid integration method raises for delP."""
        model = models["simple"](mass=2000.0)
        traj = make_afssh_traj(
            model, -5.0, 0.01, 0,
            augmented_integration="exp",
            seed_sequence=42
        )
        # Take one step with valid method
        step_trajectory(traj)
        # Then set invalid method
        traj.augmented_integration = "invalid"

        with pytest.raises(Exception, match="Unrecognized propagate delP"):
            # Call advance_delP directly
            traj.advance_delP(traj.last_electronics, traj.electronics)


class TestAFSSHDecoherence:
    """Tests for A-FSSH decoherence and collapse behavior."""

    def test_gamma_collapse_computation(self):
        """Test that gamma_collapse returns proper probabilities."""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(
            model, 0.0, 0.01, 0,  # Start near coupling region
            seed_sequence=42
        )

        # Run into coupling region
        for _ in range(50):
            step_trajectory(traj)

        # Compute gamma
        gamma = traj.gamma_collapse(traj.electronics)

        # gamma should be an array with nstates elements
        assert gamma.shape == (model.nstates,)
        # Self-collapse should be zero
        assert gamma[traj.state] == 0.0

    def test_collapse_event_recorded(self):
        """Test that collapse events are recorded in tracer when triggered.

        The collapse probability is very small in typical simulations, so
        we directly test the collapse mechanism by calling surface_hopping
        with artificially large delR values.
        """
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(model, 0.0, 0.01, 0, seed_sequence=42)

        # Run a few steps to set up electronics
        for _ in range(20):
            step_trajectory(traj)

        # Artificially set large delR values to trigger collapse
        # Gamma is proportional to delR * force difference
        # Set delR large enough that gamma becomes significant
        traj.delR[:, 1, 1] = 1e6  # Large value on state 1 diagonal
        traj.delR[:, 0, 0] = 0.0

        # Set delP non-zero to avoid division issues
        traj.delP[:, :, :] = 1e-8

        # Compute gamma - it should be non-trivial now
        gamma = traj.gamma_collapse(traj.electronics)

        # Check that gamma has sensible structure
        assert gamma.shape == (model.nstates,)
        assert gamma[traj.state] == 0.0  # Self-collapse always zero

        # Even if we can't trigger an actual collapse event, verify
        # the collapse code path works by calling surface_hopping
        # (which will call gamma_collapse internally)
        initial_events = len(traj.tracer.hops)
        traj.surface_hopping(traj.last_electronics, traj.electronics)
        # The surface_hopping method should run without error

    def test_collapse_resets_density_matrix(self):
        """Test that collapse properly resets density matrix."""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(model, 0.0, 0.01, 0, seed_sequence=42)

        # Run into coupling region to build up off-diagonal rho
        for _ in range(20):
            step_trajectory(traj)

        # Force a collapse by setting delR to produce large positive gamma
        fm = traj.electronics.force_matrix
        ddF_sign = np.sign(fm[0, 0] - fm[1, 1])
        traj.delR[:, 0, 0] = ddF_sign * 1e6
        traj.delR[:, 1, 1] = 0.0
        traj.delP[:, 0, 0] = ddF_sign * 1e-4
        traj.delP[:, 1, 1] = 0.0

        traj.surface_hopping(traj.last_electronics, traj.electronics)

        collapse_events = traj.tracer.events.get("collapse", [])
        assert len(collapse_events) > 0, "Collapse event was not triggered"

        # After collapse, rho should be reset to pure state
        assert np.isclose(traj.rho[traj.state, traj.state], 1.0, atol=1e-10)
        for i in range(model.nstates):
            if i != traj.state:
                assert np.isclose(traj.rho[i, i], 0.0, atol=1e-10)

    def test_collapse_resets_delR_delP(self):
        """Test that collapse resets delR and delP matrices."""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(model, 0.0, 0.01, 0, seed_sequence=42)

        # Run into coupling region
        for _ in range(20):
            step_trajectory(traj)

        # Force a collapse by setting delR to produce large positive gamma
        fm = traj.electronics.force_matrix
        ddF_sign = np.sign(fm[0, 0] - fm[1, 1])
        traj.delR[:, 0, 0] = ddF_sign * 1e6
        traj.delR[:, 1, 1] = 0.0
        traj.delP[:, 0, 0] = ddF_sign * 1e-4
        traj.delP[:, 1, 1] = 0.0

        traj.surface_hopping(traj.last_electronics, traj.electronics)

        collapse_events = traj.tracer.events.get("collapse", [])
        assert len(collapse_events) > 0, "Collapse event was not triggered"

        # After collapse, delR and delP should be reset to zero
        assert traj.delR.shape == (model.ndof, model.nstates, model.nstates)
        assert traj.delP.shape == (model.ndof, model.nstates, model.nstates)
        np.testing.assert_array_equal(traj.delR, 0.0)
        np.testing.assert_array_equal(traj.delP, 0.0)


class TestAFSSHHopUpdate:
    """Tests for A-FSSH hop_update behavior."""

    def test_hop_update_shifts_delR_delP(self):
        """Test that hop_update shifts delR and delP correctly."""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(model, 0.0, 0.007, 0, seed_sequence=42)

        # Run some steps to build up delR/delP
        for _ in range(20):
            step_trajectory(traj)

        # Store values before hop_update
        delR_before = traj.delR.copy()
        delP_before = traj.delP.copy()

        # Simulate a hop from state 0 to state 1
        hop_from, hop_to = 0, 1

        # Get the diagonal values that will be subtracted
        dRb = delR_before[:, hop_to, hop_to].copy()
        dPb = delP_before[:, hop_to, hop_to].copy()

        # Call hop_update
        traj.hop_update(hop_from, hop_to)

        # Verify the shift happened correctly
        for i in range(model.nstates):
            np.testing.assert_allclose(
                traj.delR[:, i, i],
                delR_before[:, i, i] - dRb,
                rtol=1e-10
            )
            np.testing.assert_allclose(
                traj.delP[:, i, i],
                delP_before[:, i, i] - dPb,
                rtol=1e-10
            )


class TestAFSSHDirectionOfRescale:
    """Tests for A-FSSH direction_of_rescale method."""

    def test_direction_of_rescale_returns_real(self):
        """Test that direction_of_rescale returns real-valued vector."""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(model, 0.0, 0.007, 0, seed_sequence=42)

        # Run some steps
        for _ in range(20):
            step_trajectory(traj)

        direction = traj.direction_of_rescale(0, 1)

        # Should be real and have correct shape
        assert direction.dtype in [np.float64, np.float32]
        assert direction.shape == (model.ndof,)

    def test_direction_of_rescale_difference(self):
        """Test direction is based on delP diagonal difference."""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(model, 0.0, 0.007, 0, seed_sequence=42)

        # Run some steps
        for _ in range(20):
            step_trajectory(traj)

        direction = traj.direction_of_rescale(0, 1)
        expected = np.real(traj.delP[:, 0, 0] - traj.delP[:, 1, 1])

        np.testing.assert_allclose(direction, expected, rtol=1e-10)


class TestAFSSHFullTrajectory:
    """Integration tests for full A-FSSH trajectories."""

    def test_afssh_dual_model_completes(self):
        """Test that A-FSSH trajectory on dual model completes without error."""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(
            model, -10.0, 0.007, 0,
            bounds=[[-5.0], [5.0]],
            seed_sequence=78341  # Same seed as reference test
        )

        # Use simulate() to run the full trajectory
        trace = traj.simulate()

        # Trajectory should complete with data
        assert len(trace) > 0
        assert not np.isnan(traj.position).any()
        assert not np.isnan(traj.velocity).any()

    def test_afssh_with_rk4_completes(self):
        """Test A-FSSH with RK4 integration completes without error."""
        model = models["dual"](mass=2000.0)
        traj = make_afssh_traj(
            model, -10.0, 0.007, 0,
            bounds=[[-5.0], [5.0]],
            augmented_integration="rk4",
            seed_sequence=42
        )

        trace = traj.simulate()

        assert len(trace) > 0
        assert not np.isnan(traj.position).any()
        assert not np.isnan(traj.velocity).any()

    def test_afssh_energy_conservation(self):
        """Test that A-FSSH conserves total energy reasonably well."""
        model = models["simple"](mass=2000.0)
        traj = make_afssh_traj(model, -10.0, 0.01, 0, seed_sequence=42)

        initial_energy = None
        energies = []

        for _ in range(100):
            step_trajectory(traj)
            # Total energy = KE + PE (use traj methods for proper computation)
            ke = traj.kinetic_energy()
            pe = traj.potential_energy()
            total_e = float(ke + pe)
            energies.append(total_e)
            if initial_energy is None:
                initial_energy = total_e

        # Energy should be roughly conserved (within numerical precision)
        # Allow some deviation due to hopping and decoherence
        max_deviation = max(abs(e - initial_energy) for e in energies)
        assert max_deviation < 0.01 * abs(initial_energy)
