import numpy as np

from typing import Any
from mudslide.typing import ArrayLike
from mudslide.models import ElectronicModel_
from .. import periodic_table

try:
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

    def __init__(self, pdb, ff, system):
        """Initialize OpenMM interface"""
        super().__init__(representation="adiabatic",
                         nstates=1,
                         ndim=pdb.topology.getNumAtoms() * 3)

        self._pdb = pdb
        self._ff = ff
        self._system = system

        self._natoms = pdb.topology.getNumAtoms()

        # check if there are constraints
        num_constraints = self._system.getNumConstraints()
        if num_constraints > 0:
            raise ValueError(
                "OpenMM system has constraints, which are not supported by mudslide. "
                +
                "Please remove constraints from the system and use the rigidWater=True option."
            )

        # check if there are virtual sites
        if any([ self._system.isVirtualSite(i) for i in range(self._system.getNumParticles()) ]):
            raise ValueError(
                "OpenMM system has virtual sites, which are not supported by mudslide."
            )

        # get charges
        try:
            nonbonded = [ f for f in self._system.getForces() if isinstance(f, openmm.NonbondedForce) ][0]
            self._charges = np.array([ nonbonded.getParticleParameters(i)[0].value_in_unit(openmm.unit.elementary_charge) for i in range(self._natoms)])
        except IndexError:
            raise ValueError("Can't find charges from OpenMM, probably because mudslide only understands Amber-like forces")

        # make dummy integrator
        self._integrator = openmm.VerletIntegrator(0.001 *
                                                   openmm.unit.picoseconds)

        # create simulation
        self._simulation = openmm.app.Simulation(self._pdb.topology,
                                                 self._system, self._integrator)

        xyz = np.array(
            self._convert_openmm_position_to_au(
                pdb.getPositions())).reshape(-1)
        self._position = xyz
        self._elements = [
            atom.element.symbol.lower() for atom in self._pdb.topology.atoms()
        ]
        self.mass = np.array([
            periodic_table.masses[e]
            for e in self._elements
            for i in range(3)
        ])
        self.mass *= mudslide.constants.amu_to_au

    def _convert_au_position_to_openmm(self, xyz):
        """Convert position from bohr to nanometer using OpenMM units"""
        nm = openmm.unit.nanometer
        return (xyz.reshape(-1, 3) * mudslide.constants.bohr_to_angstrom *
                0.1) * nm

    def _convert_openmm_position_to_au(self, xyz):
        """Convert position from nanometer to bohr using OpenMM units"""
        nm = openmm.unit.nanometer
        return np.array(
            xyz / nm).reshape(-1) * (10.0 / mudslide.constants.bohr_to_angstrom)

    def _convert_openmm_force_to_au(self, force):
        kjmol = openmm.unit.kilojoules_per_mole
        nm = openmm.unit.nanometer
        return np.array(force * nm / kjmol).reshape(
            1, -1
        ) / mudslide.constants.Hartree_to_kJmol * 0.1 * mudslide.constants.bohr_to_angstrom

    def _convert_energy_to_au(self, energy):
        kjmol = openmm.unit.kilojoules_per_mole
        return energy / kjmol / mudslide.constants.Hartree_to_kJmol

    def compute(self,
                position: ArrayLike,
                couplings: Any = None,
                gradients: Any = None,
                reference: Any = None):
        """Compute energy and forces"""
        self._position = position
        xyz = self._convert_au_position_to_openmm(position)

        self._simulation.context.setPositions(xyz)
        state = self._simulation.context.getState(getPositions=True,
                                                  getEnergy=True,
                                                  getForces=True)

        self.energies = np.array([self._convert_energy_to_au(state.getPotentialEnergy())])
        self._hamiltonian = np.zeros([1, 1])
        self._hamiltonian[0,0] = self.energies[0]

        self._force = self._convert_openmm_force_to_au(
            state.getForces(asNumpy=True))
        self._forces_available[:] = True

