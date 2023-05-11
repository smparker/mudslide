
import numpy as np

try:
    import openmm
    import openmm.app
    OPENMM_INSTALLED = True
except ImportError:
    OPENMM_INSTALLED = False

import mudslide.electronics

def openmm_is_installed():
    """Check if OpenMM is installed"""
    return OPENMM_INSTALLED

class OpenMM(mudslide.electronics.ElectronicModel_):
    """OpenMM interface"""

    def __init__(self, pdb, ff, system):
        """Initialize OpenMM interface"""
        super().__init__(representation="adiabatic", nstates=1,
                         ndim=pdb.topology.getNumAtoms() * 3)

        self._pdb = pdb
        self._ff = ff
        self._system = system

        num_constraints = self._system.getNumConstraints()
        if num_constraints > 0:
            raise ValueError("OpenMM system has constraints, which are not supported by mudslide. " +
            "Please remove constraints from the system and use the rigidWater=True option.")

        # make dummy integrator
        self._integrator = openmm.VerletIntegrator(0.001 * openmm.unit.picoseconds)

        # create simulation
        self._simulation = openmm.app.Simulation(self._pdb.topology, self._system, self._integrator)

        xyz = np.array(self._convert_openmm_position_to_au(pdb.positions)).reshape(-1)
        self._position = xyz

    def _convert_au_position_to_openmm(self, xyz):
        """Convert position from bohr to nanometer using OpenMM units"""
        nm = openmm.unit.nanometer
        return (xyz.reshape(-1, 3) * mudslide.constants.bohr_to_angstrom * 0.1) * nm

    def _convert_openmm_position_to_au(self, xyz):
        """Convert position from nanometer to bohr using OpenMM units"""
        nm = openmm.unit.nanometer
        return np.array(xyz / nm).reshape(-1) * (10.0 / mudslide.constants.bohr_to_angstrom)

    def _convert_openmm_force_to_au(self, force):
        kjmol = openmm.unit.kilojoules_per_mole
        nm = openmm.unit.nanometer
        return np.array(force * nm / kjmol).reshape(1, -1) / mudslide.constants.Hartree_to_kJmol * 0.1 * mudslide.constants.bohr_to_angstrom

    def _convert_energy_to_au(self, energy):
        kjmol = openmm.unit.kilojoules_per_mole
        return energy / kjmol / mudslide.constants.Hartree_to_kJmol

    def compute(self, position):
        """Compute energy and forces"""
        self._position = position
        xyz = self._convert_au_position_to_openmm(position)

        self._simulation.context.setPositions(xyz)
        state = self._simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
        self._force = self._convert_openmm_force_to_au(state.getForces(asNumpy=True))
        self._forces_available[:] = True
        self._energy = self._convert_energy_to_au(state.getPotentialEnergy())

