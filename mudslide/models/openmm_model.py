# -*- coding: utf-8 -*-
""" OpenMM interface for mudslide """

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .electronics import ElectronicModel_
from ..constants import bohr_to_angstrom, amu_to_au, Hartree_to_kJmol
from ..periodic_table import masses

try:
    # Suppress OpenMM warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*SwigPyPacked.*")
    warnings.filterwarnings("ignore", message=".*SwigPyObject.*")
    warnings.filterwarnings("ignore", message=".*swigvarlink.*")
    import openmm
    import openmm.app
    OPENMM_INSTALLED = True
except ImportError:
    OPENMM_INSTALLED = False


def openmm_is_installed():
    """Check if OpenMM is installed"""
    return OPENMM_INSTALLED


class OpenMM(ElectronicModel_):
    """OpenMM interface"""

    def __init__(self,
                 pdb,
                 ff,
                 system,
                 platform_name: str = "Reference",
                 properties: dict = None):
        """Initialize OpenMM interface"""
        super().__init__(representation="adiabatic",
                         nstates=1,
                         nparticles=pdb.topology.getNumAtoms(),
                         ndims=3)

        self._pdb = pdb
        self._ff = ff
        self._system = system

        # check if there are constraints
        num_constraints = self._system.getNumConstraints()
        if num_constraints > 0:
            raise ValueError(
                "OpenMM system has constraints, which are not supported by mudslide. "
                +
                "Please remove constraints from the system and use the rigidWater=True option."
            )

        # check if there are virtual sites
        if any(
                self._system.isVirtualSite(i)
                for i in range(self._system.getNumParticles())):
            raise ValueError(
                "OpenMM system has virtual sites, which are not supported by mudslide."
            )

        # get charges
        try:
            nonbonded = [
                f for f in self._system.getForces()
                if isinstance(f, openmm.NonbondedForce)
            ][0]
            self._charges = np.array([
                nonbonded.getParticleParameters(i)[0].value_in_unit(
                    openmm.unit.elementary_charge)
                for i in range(self.nparticles)
            ])
        except IndexError as exc:
            raise ValueError(
                "Can't find charges from OpenMM,"
                " probably because mudslide only understands Amber-like forces"
            ) from exc

        # make dummy integrator
        self._integrator = openmm.VerletIntegrator(0.001 *
                                                   openmm.unit.picoseconds)

        # create simulation
        platform = openmm.Platform.getPlatformByName(platform_name)
        properties = properties if properties is not None else {}
        self._simulation = openmm.app.Simulation(self._pdb.topology,
                                                 self._system, self._integrator,
                                                 platform, properties)

        xyz = np.array(self._convert_openmm_position_to_au(
            pdb.getPositions())).reshape(-1)
        self._position = xyz
        self.atom_types = [
            atom.element.symbol.lower() for atom in self._pdb.topology.atoms()
        ]
        self.mass = np.array(
            [masses[e] for e in self.atom_types for i in range(3)])
        self.mass *= amu_to_au
        self.energies = np.zeros([1], dtype=np.float64)

    def _convert_au_position_to_openmm(self, xyz):
        """Convert position from bohr to nanometer using OpenMM units"""
        nm = openmm.unit.nanometer
        return (xyz.reshape(-1, 3) * bohr_to_angstrom * 0.1) * nm

    def _convert_openmm_position_to_au(self, xyz):
        """Convert position from nanometer to bohr using OpenMM units"""
        nm = openmm.unit.nanometer
        return np.array(xyz / nm).reshape(-1) * (10.0 / bohr_to_angstrom)

    def _convert_openmm_force_to_au(self, force):
        kjmol = openmm.unit.kilojoules_per_mole
        nm = openmm.unit.nanometer
        return np.array(force * nm / kjmol).reshape(
            1, -1) / Hartree_to_kJmol * 0.1 * bohr_to_angstrom

    def _convert_energy_to_au(self, energy):
        kjmol = openmm.unit.kilojoules_per_mole
        return energy / kjmol / Hartree_to_kJmol

    def compute(self,
                X: ArrayLike,
                couplings: Any = None,
                gradients: Any = None,
                reference: Any = None):
        """Compute energy and forces"""
        self._position = X
        xyz = self._convert_au_position_to_openmm(X)

        self._simulation.context.setPositions(xyz)
        state = self._simulation.context.getState(getPositions=True,
                                                  getEnergy=True,
                                                  getForces=True)

        self.energies = np.array(
            [self._convert_energy_to_au(state.getPotentialEnergy())])
        self._hamiltonian = np.zeros([1, 1])
        self._hamiltonian[0, 0] = self.energies[0]

        self._force = self._convert_openmm_force_to_au(
            state.getForces(asNumpy=True))
        self._forces_available[:] = True
