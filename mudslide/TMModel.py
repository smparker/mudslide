# -*- coding: utf-8 -*-
"""Implementations of the one-dimensional two-state models Tully demonstrated FSSH on in Tully, J.C. <I>J. Chem. Phys.</I> 1990 <B>93</B> 1061."""

import numpy as np
import math
from scipy.special import erf
import subprocess
import turboparse
import re
import copy as cp

from pathlib import Path

from .electronics import ElectronicModel_ 

from typing import Tuple, Any

from .typing import ArrayLike, DtypeLike
from .constants import eVtoHartree, amu_to_au
from .periodic_table import masses

class TMModel(ElectronicModel_):
    def __init__(
        self,
        turbomole_dir: str,
        states: ArrayLike,
        sub_dir_stem:str = "traj",
        representation: str = "adiabatic",
        reference: Any = None,
        turbomole_modules = {"gs_energy": "ridft", "gs_grads": "rdgrad", "es_grads": "egrad"}
    ):
        ElectronicModel_.__init__(self, representation=representation, reference=reference)

        self.turbomole_dir = turbomole_dir
        self.states = states
        self.nstates_ = len(self.states)
        
        self.sub_dir_stem = sub_dir_stem
        self.sub_dir_num = 0

        self.turbomole_modules = turbomole_modules
        self.turbomole_init()

    
    def nstates(self):
            return self.nstates_

    def ndim(self):
            return self.ndim_

    def turbomole_init(self):
        self.coord_path = Path(self.turbomole_dir)/"control"  
        self.get_coords()

    def get_coords(self):
        coords = []
        self.atom_order = []
        self.mass = []
        coord = subprocess.run(["sdg", "coord"], capture_output= True, text = True).stdout
        coord_list = coord.rstrip().split("\n")

        for c in coord_list[1:]:
            c_list = c.split()
            self.atom_order.append(c_list[3])
            coords.append([float(val) for val in c_list[:3]])
            self.mass.append(3 * [masses[c_list[3]]])

        self.ndim_ = 3 * len(coords)
        self.X = np.array(coords, dtype=np.float64).reshape(self.ndim())
        self.mass = np.array(self.mass, dtype=np.float64).reshape(self.ndim()) * amu_to_au


    def update_coords(self, X):
        self.coord_path = subprocess.run(["sdg", "-f", "coord"], capture_output=True, text=True).stdout.rstrip()
        X = X.reshape((self.ndim() // 3, 3))
        with open(self.coord_path, "r") as f:
            lines = f.readlines()
        regex = re.compile(r"\$coord\n|\$coord\s")
        read_start = np.argwhere(np.array([bool(regex.match(line)) for line in lines]))[0][0]

        current_line = read_start +1
        for i, coord_list in enumerate(X):
            lines[current_line] = "    {: .14f}      {: .14f}      {: .14f}      {}\n".format(
                                  coord_list[0], coord_list[1], coord_list[2], self.atom_order[i]
                    )
            current_line += 1
        
        with open(self.coord_path, "w") as coord_file:
            coord_file.write("".join(lines))            


    def call_turbomole(self, outname="turbo.out"):
        # Open file/run Turbomole
        with open(outname, "w") as f:
            for turbomole_module in self.turbomole_modules.values():
                tur_output = subprocess.run(turbomole_module, stdout=f) 

        # Parse results with Turboparse
        with open(outname, "r") as f:
            data_dict = turboparse.parse_turbo(f)

        # Now add results to model
    def call_turbomole(self, outname="turbo.out"):

        with open(outname, "w") as f:
            for turbomole_module in self.turbomole_modules.values():
                tur_output = subprocess.run(turbomole_module, stdout=f) 

        # Parse results with Turboparse
        with open(outname, "r") as f:
            data_dict = turboparse.parse_turbo(f)

        # Now add results to model
        energy = data_dict[self.turbomole_modules["gs_energy"]]["energy"]
        self.energies = [energy]
        parsed_gradients = data_dict[self.turbomole_modules["gs_grads"]]["gradient"]

        # Check for presence of egrad turbomole module

        if "egrad" in self.turbomole_modules.values():
            # egrad updates to energy
            parsed_energies = data_dict["egrad"]["excited_state"][0]["energy"]
            excited_energies = [
                data_dict["egrad"]["excited_state"][i]["energy"]
                + energy for i in range(len(data_dict["egrad"]["excited_state"]))
            ]
            self.energies.extend(excited_energies)

            # egrad couplings
            parsed_nac_coupling = data_dict["egrad"]["coupling"]
            self.derivative_coupling = np.zeros((self.nstates(), self.nstates(), self.ndim()))
            for dct in parsed_nac_coupling:
                i = dct["bra_state"]
                j = dct["ket_state"]

                self.derivative_coupling[i][j] = np.array(dct["d/dR"]).reshape(self.ndim(), order="F")
                self.derivative_coupling[j][i] = self.derivative_coupling[i][j]

            # egrad updates to gradients
            parsed_gradients.extend(data_dict["egrad"]["gradient"])

        # Reshape gradients
        self.gradients = np.zeros((self.nstates(),self.ndim()))
        for state in self.states:
            grads = []
            for i in range(self.ndim() // 3):
                grads.extend(
                [
                parsed_gradients[state]["d_dx"][i],
                parsed_gradients[state]["d_dy"][i],
                parsed_gradients[state]["d_dz"][i],
                ]
                        )
            grads = np.array(grads)
            self.gradients[state] = grads

        self.force = -(self.gradients) 

    def V(self, X: ArrayLike) -> ArrayLike:
        out = np.zeros([self.nstates(), self.nstates()])
        for i, e in enumerate(self.states):
            out[i][i] = self.energies[e]
        return out

    def compute(self, X, couplings, gradients, reference):
        """
        Calls Turbomole/Turboparse to generate electronic properties including
        gradients, couplings, energies. For this to work, this class
        needs to know where the "original" files/data sit, so that they
        can get properly passed to Turbomole. (__init__() can get these
        file locations.)
        """
        self.call_turbomole()
        self.reference, self.hamiltonian = self._compute_basis_states(self.V(X), reference=reference)


    def update(self, X: ArrayLike, couplings: Any = None, gradients: Any = None): 
        out = cp.copy(self)
        out.position = X
        out.update_coords(X)
        out.compute(X, couplings=couplings, gradients=gradients, reference=self.reference)
        return out

    def _compute_basis_states(self, V: ArrayLike, reference: Any = None) -> Tuple[ArrayLike,ArrayLike]:
        """Computes coefficient matrix for basis states
        if a diabatic representation is chosen, no transformation takes place
        :param V: potential matrix
        :param reference: ElectronicStates from previous step used only to fix phase
        """
        if self.representation == "adiabatic":
            en, co = np.linalg.eigh(V)
            nst = self.nstates()
            coeff = co[:,:nst]
            energies = en[:nst]

            if reference is not None:
                try:
                    for mo in range(self.nstates()):
                        if (np.dot(coeff[:,mo], reference[:,mo]) < 0.0):
                            coeff[:,mo] *= -1.0
                except:
                    raise Exception("Failed to regularize new ElectronicStates from a reference object %s" % (reference))
            return coeff, np.diag(energies)
        elif self.representation == "diabatic":
            raise Exception("Adiabatic models can only be run in adiabatic mode")
            return None
        else:
            raise Exception("Unrecognized representation")

    def clone(self):
        this_sub_dir = self.sub_dir_stem + "_" + str(self.sub_dir_num)
        p = Path(self.turbomole_dir)/this_sub_dir
        p.mkdir(parents = True, exist_ok = True)
        control_path = Path(self.turbomole_dir)/"control"
        subprocess.run(["cpc",p, control_path])          

