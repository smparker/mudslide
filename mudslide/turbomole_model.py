# -*- coding: utf-8 -*-
"""Implementations of the one-dimensional two-state models Tully demonstrated FSSH on in Tully, J.C. <I>J. Chem. Phys.</I> 1990 <B>93</B> 1061."""
import numpy as np
import math
from scipy.special import erf
import subprocess
import turboparse
import re
import copy as cp

import os, sys, io, shutil

from pathlib import Path

from .electronics import ElectronicModel_

from typing import Tuple, Any

from .typing import ArrayLike, DtypeLike
from .constants import eVtoHartree, amu_to_au
from .periodic_table import masses


def turbomole_is_installed():
    # needs to have turbodir set
    has_turbodir = "TURBODIR" in os.environ
    # check that the scripts directory is available by testing for `sysname`
    has_scripts = shutil.which("sysname") is not None
    # check that the bin directory is available by testing for `sdg`
    has_bin = shutil.which("sdg") is not None

    return has_turbodir and has_scripts and has_bin


class TMModel(ElectronicModel_):

    def __init__(
        self,
        turbomole_dir: str,
        states: ArrayLike,
        sub_dir_stem: str = "traj",
        representation: str = "adiabatic",
        reference: Any = None,
        expert=False,  # when False, will update turbomole parameters for best NAMD performance
        turbomole_modules={
            "gs_energy": "ridft",
            "gs_grads": "rdgrad",
            "es_grads": "egrad"
        }):
        ElectronicModel_.__init__(self, representation=representation, reference=reference)

        self.turbomole_dir = turbomole_dir
        self.states = states
        self.nstates_ = len(self.states)

        self.sub_dir_stem = sub_dir_stem
        self.sub_dir_num = 0

        assert turbomole_is_installed()

        self.turbomole_modules = turbomole_modules
        assert all([shutil.which(x) is not None for x in self.turbomole_modules.values()])

        self.turbomole_init()

        if not expert:
            self.apply_suggested_parameters()

    def nstates(self):
        return self.nstates_

    def ndim(self):
        return self.ndim_

    def sdg(self,
            dg,
            file=None,
            show_keyword=False,
            show_body=False,
            show_filename_only=False,
            discard_comments=True,
            quiet=False):
        sdg_command = "sdg"
        if file is not None:
            sdg_command += " -s {}".format(file)
        if show_keyword:
            sdg_command += " -t"
        if show_body:
            sdg_command += " -b"
        if show_filename_only:
            sdg_command += " -f"
        if discard_comments:
            sdg_command += " -c"
        if quiet:
            sdg_command += " -q"

        sdg_command += " {}".format(dg)

        result = subprocess.run(sdg_command.split(), capture_output=True, text=True)
        return result.stdout.rstrip()

    def adg(self, dg, data, newline=False):
        if not isinstance(data, list):
            data = [data]
        lines = "\\n".join(["{}".format(x) for x in data])
        if newline:
            lines = "\\n" + lines
        adg_command = "adg {} {}".format(dg, lines)
        result = subprocess.run(adg_command.split(), capture_output=True, text=True)

    def apply_suggested_parameters(self):
        # weight derivatives are mandatory
        sdg_dft = self.sdg("dft", show_body=True)
        if "weight derivatives" not in sdg_dft:
            current_dft = sdg_dft.split("\n")
            self.adg("dft", current_dft + [" weight derivatives"], newline=True)

        # prefer psuedowavefunction couplings with ETFs
        sdg_nac = self.sdg("nacme")[6:].strip()  # skip past $nacme
        update_nac = False
        if "response" in sdg_nac:
            update_nac = True
            sdg_nac = sdg_nac.replace("response", "pseudo")
        if "do_etf" not in sdg_nac:
            update_nac = True
            sdg_nac += " do_etf"
        if update_nac:
            self.adg("nacme", sdg_nac)

        # probably force phaser on as well

    def run_single(self, module, stdout=sys.stdout):
        output = subprocess.run(module, capture_output=True, text=True)
        print(output.stdout, file=stdout)
        if "abnormal" in output.stderr:
            raise Exception("Call to {} ended abnormally".format(module))

    def turbomole_init(self):
        self.coord_path = self.sdg("coord", show_filename_only=True)
        self.get_coords()

    def get_coords(self):
        coords = []
        self.atom_order = []
        self.mass = []
        coord = self.sdg("coord")
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
        X = X.reshape((self.ndim() // 3, 3))

        with open(self.coord_path, "r") as f:
            lines = f.readlines()

        regex = re.compile(r"\s*\$coord")
        coordline = 0
        for i, line in enumerate(lines):
            if regex.match(line) is not None:
                coordline = i
                break
        # Reached end of file without finding $coord.
        if line == "":
            raise ValueError(f"$coord entry not found in file: {self.coord_path}!")

        coordline += 1
        for i, coord_list in enumerate(X):
            lines[coordline] = "{:26.16e}{:28.16e}{:28.16e}{:>7}\n".format(coord_list[0], coord_list[1], coord_list[2],
                                                                           self.atom_order[i])
            coordline += 1

        with open(self.coord_path, "w") as coord_file:
            coord_file.write("".join(lines))

        # Now add results to model
    def call_turbomole(self, outname="turbo.out"):
        with open(outname, "w") as f:
            for turbomole_module in self.turbomole_modules.values():
                self.run_single(turbomole_module, stdout=f)

        # Parse results with Turboparse
        with open(outname, "r") as f:
            data_dict = turboparse.parse_turbo(f)

        # Now add results to model
        energy = data_dict[self.turbomole_modules["gs_energy"]]["energy"]
        self.energies = [energy]
        parsed_gradients = [data_dict[self.turbomole_modules["gs_grads"]]["gradient"][0]["gradients"]]

        # Check for presence of egrad turbomole module
        if "egrad" in self.turbomole_modules.values():
            # egrad updates to energy
            excited_energies = [
                data_dict["egrad"]["excited_state"][i]["energy"] + energy
                for i in range(len(data_dict["egrad"]["excited_state"]))
            ]
            self.energies.extend(excited_energies)

            # egrad couplings
            parsed_nac_coupling = data_dict["egrad"]["coupling"]
            self.derivative_coupling = np.zeros((self.nstates(), self.nstates(), self.ndim()))
            for dct in parsed_nac_coupling:
                i = dct["bra_state"]
                j = dct["ket_state"]

                self.derivative_coupling[i][j] = np.array(dct["d/dR"]).reshape(self.ndim(), order="F")
                self.derivative_coupling[j][i] = -(self.derivative_coupling[i][j])

            # egrad updates to gradients
            for i in range(len(data_dict["egrad"]["gradient"])):
                parsed_gradients.extend([data_dict["egrad"]["gradient"][i]["gradients"]])

        # Reshape gradients

        self.gradients = np.array(parsed_gradients)

        self.force = -(self.gradients)

    def compute(self, X, couplings, gradients, reference):
        """
        Calls Turbomole/Turboparse to generate electronic properties including
        gradients, couplings, energies. For this to work, this class
        needs to know where the "original" files/data sit, so that they
        can get properly passed to Turbomole. (__init__() can get these
        file locations.)
        """
        self.call_turbomole()
        self.hamiltonian = np.zeros([self.nstates(), self.nstates()])
        for i, e in enumerate(self.states):
            self.hamiltonian[i][i] = self.energies[e]
        return self.hamiltonian

    def update(self, X: ArrayLike, couplings: Any = None, gradients: Any = None):
        out = cp.copy(self)
        out.position = X
        out.update_coords(X)
        out.compute(X, couplings=couplings, gradients=gradients, reference=self.reference)
        return out
