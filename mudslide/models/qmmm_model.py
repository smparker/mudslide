# -*- coding: utf-8 -*-
"""QM/MM model using turbomole and OpenMM"""

import os

import numpy as np

try:
    import openmm
    import openmm.app
    OPENMM_INSTALLED = True
except ImportError:
    OPENMM_INSTALLED = False

from mudslide.models import ElectronicModel_
from mudslide.typing import ArrayLike
from typing import Any

class QMMM(ElectronicModel_):
    """A QM/MM model"""
    def __init__(self, qm_model, mm_model):
        self._qm_model = qm_model
        self._mm_model = mm_model

        # initialize position with mm_model
        self._position = np.copy(mm_model._position)

        self._ndof_qm = qm_model.ndof()
        self._ndof_mm = mm_model.ndof() - self._ndof_qm
        self._qm_atoms = list(range(self._ndof_qm//3))
        self._nqm = len(self._qm_atoms)

        # update position for qm atoms, just in case they are different
        self._position[:self._ndof_qm] = qm_model._position

        super().__init__(nstates=qm_model.nstates(), ndof=mm_model.ndof())

        if not self.check_qm_and_mm_regions(self._qm_atoms):
            raise ValueError("QM atoms must have the same elements in the QM and MM models.")
        self.remove_qm_interactions(self._qm_atoms)

    def check_qm_and_mm_regions(self, qm_atoms):
        """Check to make sure that the atoms labelled QM
        have at least the same elements listed in the QM model
        and the MM model.
        """
        qm_elements = self._qm_model._elements
        qm_elements_in_mm = [ self._mm_model.atom_types[i] for i in qm_atoms ]
        return qm_elements == qm_elements_in_mm

    def remove_qm_interactions(self, qm_atoms):
        """Remove bonded interactions of QM atoms from the MM model

        Args:
            indices (list): list of indices of QM atoms
        """
        num_bond_removed = 0
        num_angl_removed = 0
        num_tors_removed = 0
        for force in self._mm_model._system.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                for n in range(force.getNumBonds()):
                    a, b, r, k = force.getBondParameters(n)
                    if a in qm_atoms and b in qm_atoms:
                        force.setBondParameters(n, a, b, r, k*0.000)
                        num_bond_removed += 1
                    if (a in qm_atoms) != (b in qm_atoms):
                        raise ValueError(f"Bonded interactions between QM and MM regions not allowed."
                            "Atoms {a:d} and {b:d} are bonded across regions.")
            elif isinstance(force, openmm.HarmonicAngleForce):
                for n in range(force.getNumAngles()):
                    a, b, c, t, k = force.getAngleParameters(n)
                    in_qm = [x in qm_atoms for x in [a, b, c]]
                    if all(in_qm):
                        force.setAngleParameters(n, a, b, c, t, k*0.000)
                        num_angl_removed += 1
                    elif any(in_qm):
                        raise ValueError(f"Bonded interactions between QM and MM regions not allowed."
                            "Atoms {a:d}, {b:d}, and {c:d} are bonded across regions.")
            elif isinstance(force, openmm.PeriodicTorsionForce):
                for n in range(force.getNumTorsions()):
                    a, b, c, d, mult, phi, k = force.getTorsionParameters(n)
                    in_qm = [x in qm_atoms for x in [a, b, c, d]]
                    if all(in_qm):
                        force.setTorsionParameters(n, a, b, c, d, mult, phi, k*0.000)
                        num_tors_removed  += 1
                    elif any(in_qm):
                        raise ValueError(f"Bonded interactions between QM and MM regions not allowed."
                            "Atoms {a:d}, {b:d}, {c:d}, and {d:d} are bonded across regions.")
            elif isinstance(force, openmm.NonbondedForce):
                for n in range(force.getNumParticles()):
                    chg, sig, eps = force.getParticleParameters(n)
                    if n in qm_atoms:
                        force.setParticleParameters(n, chg*0, sig, eps)

                # add exceptions for qm-qm nonbonded interactions
                for n in range(force.getNumExceptions()):
                    i, j, chgprod, sig, eps = force.getExceptionParameters(n)
                for i in qm_atoms:
                    for j in range(i):
                        force.addException(i, j, 0, 1, 0, replace=True)
            elif isinstance(force, openmm.CMMotionRemover):
                raise ValueError(f"Cannot use CMMotionRemover in QM/MM model. Turn it off by setting removeCMMotion=False when preparing the System().")
            else:
                raise ValueError(f"Force {force.__class__.__name__} not supported in QM/MM model.")

    def compute(self, X: ArrayLike, couplings: Any=None, gradients: Any=None, reference: Any=None) -> None:
        """Computes QM/MM energy by calling OpenMM and Turbomole and stitching together the results"""
        self._position = X
        qmxyz = X[:self._ndof_qm]

        self._mm_model.compute(X)

        only_mm_xyz = X[self._ndof_qm:]
        only_mm_charges = self._mm_model._charges[self._nqm:]
        self._qm_model.control.add_point_charges(only_mm_xyz.reshape(-1,3), only_mm_charges)
        self._qm_model.compute(qmxyz)

        self._hamiltonian = self._mm_model.hamiltonian() + self._qm_model.hamiltonian()

        mmforce = self._mm_model._force
        qmforce = self._qm_model._force
        self._force = np.zeros([self.nstates(), self.ndof()])

        self._force[:,:] = mmforce
        self._force[:,:self._ndof_qm] += qmforce
        self._forces_available = self._qm_model._forces_available

        a, b, qm_on_mm_force = self._qm_model.control.read_point_charge_gradients()
        self._force[:,self._ndof_qm:] -= qm_on_mm_force.reshape(1,-1)

        def clone(self):
            """Return a copy of the QMMM object"""
            return QMMM(self._qm_model.clone(), self._mm_model.clone())

