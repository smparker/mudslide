# -*- coding: utf-8 -*-
"""Implementations of the interface between turbomole and mudslide."""

import os
import sys
import shutil
import re
import copy as cp
import subprocess

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike

from .. import turboparse

from ..util import find_unique_name
from ..constants import amu_to_au
from ..periodic_table import masses
from .electronics import ElectronicModel_


def turbomole_is_installed():
    """ Check if turbomole is installed by checking for environment variable TURBODIR and
    checking that the scripts and bin directories are available.

    :return: True if turbomole is installed, False otherwise
    """
    # needs to have turbodir set
    has_turbodir = "TURBODIR" in os.environ
    # check that the scripts directory is available by testing for `sysname`
    has_scripts = shutil.which("sysname") is not None
    # check that the bin directory is available by testing for `sdg`
    has_bin = shutil.which("sdg") is not None

    return has_turbodir and has_scripts and has_bin

class TurboControl:
    """A class to handle the control file for turbomole"""

    # default filenames
    pcgrad_file = "pcgrad"
    control_file = "control"

    def __init__(self, control_file="control", workdir=None,
                 command_prefix: Optional[List[str]] = None):
        self.command_prefix = command_prefix if command_prefix is not None else []
        # workdir is directory of control file
        if control_file is not None:
            self.workdir = os.path.abspath(os.path.dirname(control_file))
        elif workdir is not None:
            self.workdir = os.path.abspath(workdir)
        else:
            raise ValueError("Must provide either control_file or workdir")
        self.control_file = control_file

        # make sure control file exists
        if not os.path.exists(os.path.join(self.workdir, self.control_file)):
            raise RuntimeError(f"control file not found in working directory {self.workdir:s}")

        # list of data groups and which file they are in, used to avoid rerunning sdg too much
        self.dg_in_file = {}

    def _build_command(self, cmd: list) -> list:
        """Prepend command_prefix to a command list"""
        return self.command_prefix + cmd

    def check_turbomole_is_installed(self):
        """Check that turbomole is installed, raise exception if not"""
        if not turbomole_is_installed():
            raise RuntimeError("Turbomole is not installed")

    def where_is_dg(self, dg, absolute_path=False):
        """Find which file a data group is in"""
        loc = self.dg_in_file[dg] if dg in self.dg_in_file \
                else self.sdg(dg, show_filename_only=True)
        if absolute_path:
            loc = os.path.join(self.workdir, loc)
        return loc

    def sdg(
        self,
        dg,
        file=None,
        show_keyword=False,
        show_body=False,
        show_filename_only=False,
        discard_comments=True,
        quiet=False,
    ):
        """Convenience function to run show data group (sdg) on a control"""
        sdg_command = "sdg"
        if file is not None:
            sdg_command += f" -s {file}"
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

        sdg_command += f" {dg}"

        result = subprocess.run(self._build_command(sdg_command.split()), capture_output=True,
                                text=True, cwd=self.workdir, check=False)
        return result.stdout.rstrip()

    def adg(self, dg, data, newline=False):
        """Convenience function to run add data group (adg) on a control"""
        if not isinstance(data, list):
            data = [data]
        lines = "\\n".join([f"{x}" for x in data])
        if newline:
            lines = "\\n" + lines
        adg_command = f"adg {dg} {lines}"
        result = subprocess.run(self._build_command(adg_command.split()), capture_output=True,
                                text=True, cwd=self.workdir, check=True)
        # check that the command ran successfully
        if "abnormal" in result.stderr:
            raise RuntimeError(f"Call to adg ended abnormally: {result.stderr}")

    def cpc(self, dest):
        """Copy the control file and other files to a new directory"""
        subprocess.run(self._build_command(["cpc", dest]), cwd=self.workdir, check=False)
        file_list = ['ciss_a','exspectrum', 'statistics', 'dipl_a',
                     'excitationlog.1', 'moments', 'vecsao', 'control',
                     'gradient', 'energy', 'moments' ]
        for f in file_list:
            if os.path.exists(os.path.join(os.path.abspath(self.workdir), f)) and not \
                    os.path.exists(os.path.join(dest,f)):
                shutil.copy(os.path.join(os.path.abspath(self.workdir), f), dest)

    def use_weight_derivatives(self, use=True):
        """Check if weight derivatives are used in the control file"""
        sdg_dft = self.sdg("dft", show_body=True)
        if use: # make sure weight derivatives turned on
            if "weight derivatives" not in sdg_dft:
                current_dft = sdg_dft.split("\n")
                self.adg("dft", current_dft + [" weight derivatives"], newline=True)
        else: # remove weight derivatives
            if "weight derivatives" in sdg_dft:
                current_dft = sdg_dft.split("\n")
                self.adg("dft",
                         [x for x in current_dft if "weight derivatives" not in x],
                         newline=True)

    def run_single(self, module, stdout=sys.stdout):
        """Run a single turbomole module"""
        output = subprocess.run(self._build_command([module]), capture_output=True, text=True,
                                cwd=self.workdir, check=False)
        print(output.stdout, file=stdout)
        if "abnormal" in output.stderr:
            raise RuntimeError(f"Call to {module} ended abnormally")

    def read_coords(self):
        """Read the coordinates from the control file

        :return: (symbols, X) where symbols is a list of element symbols
                 and X is a numpy array of shape (n_atoms, 3)
        """
        coords = []
        symbols = []
        coord = self.sdg("coord")
        coord_list = coord.rstrip().split("\n")

        for c in coord_list[1:]:
            c_list = c.split()
            symbols.append(c_list[3])
            coords.append([float(val) for val in c_list[:3]])

        X = np.array(coords, dtype=np.float64)
        return symbols, X

    def get_masses(self, symbols):
        """Get the masses of the atoms in the system"""
        atomic_masses = np.array([masses[s] for s in symbols for _ in range(3)], dtype=np.float64)
        atomic_masses *= amu_to_au
        return atomic_masses

    def read_hessian(self):
        """
        Projected Hessian has a structure of
        $hessian (projected)
        1 1 0.000 0.000 0.000 0.000 0.000
        1 2 0.000 0.000 0.000 0.000 0.000
        2 1 0.000 0.000 0.000 0.000 0.000
        ...
        """
        hessian = []
        hess = self.sdg("hessian", show_body=True)
        hess_list = hess.rstrip().split("\n")
        ndof = 0
        for h in hess_list:
            h_list = h.split()
            ndof = max(ndof, int(h_list[0]))
            hessian.extend([float(val) for val in h_list[2:]])
        H = np.array(hessian, dtype=np.float64).reshape(ndof, ndof)
        return H

    def add_point_charges(self, coords: ArrayLike, charges: ArrayLike):
        """Add point charges to the control file

        point_charges data group has the structure:
        $point_charges nocheck list pcgrad
        <x> <y> <z> <q>
        ...
        """
        nq = len(charges)
        assert coords.shape == (nq, 3)

        self.adg("point_charges", ["file=point_charges"])
        with open(os.path.join(self.workdir, "point_charges"), "w", encoding='utf-8') as f:
            print("$point_charges nocheck list pcgrad", file=f)
            for xyz, q in zip(coords, charges):
                print(f"{xyz[0]:22.16g} {xyz[1]:22.16g} {xyz[2]:22.16g} {q:22.16f}", file=f)
            print("$end", file=f)

        # make sure point charge gradients are requested
        drvopt = self.sdg("drvopt", show_body=True, show_keyword=False)
        if "point charges" not in drvopt:
            drvopt = drvopt.rstrip().split("\n")
            drvopt += [" point charges"]
            self.adg("drvopt", drvopt, newline=True)
        self.adg("point_charge_gradients", [f"file={self.pcgrad_file}"])

    def read_point_charge_gradients(self):
        """Read point charges and gradients from the control file

        point charges in dg $point_charges
        gradients in dg $point_charge_gradients

        Returns: (coords, charges, gradients) where coords is a numpy
                 array of shape (nq, 3) with coordinates in Bohr
        """

        # read point charges
        pc = self.sdg("point_charges", show_body=True, show_keyword=False)
        pc_list = pc.rstrip().split("\n")
        nq = len(pc_list)
        coords = np.zeros((nq, 3))
        charges = np.zeros(nq)
        for i, p in enumerate(pc_list):
            p_list = p.split()
            coords[i, :] = [float(val) for val in p_list[:3]]
            charges[i] = float(p_list[3])

        # read point charge gradients
        pcgrad = self.sdg("point_charge_gradients", show_body=True, show_keyword=False)
        pcgrad_list = pcgrad.rstrip().split("\n")
        gradients = np.zeros((nq, 3))
        for i, p in enumerate(pcgrad_list):
            p_list = p.split()
            gradients[i, :] = [float(val.replace('D','e')) for val in p_list[:3]]

        return coords, charges, gradients

class TMModel(ElectronicModel_):
    """A class to handle the electronic model for excited state Turbomole calculations"""
    def __init__(
        self,
        states: ArrayLike,
        run_turbomole_dir: str = ".",
        workdir_stem: str = "run_turbomole",
        representation: str = "adiabatic",
        reference: Any = None,
        expert: bool=False,  # when False, update turbomole parameters for NAMD
        turbomole_modules: Dict=None,
        command_prefix: Optional[List[str]] = None
    ):
        self.command_prefix = command_prefix if command_prefix is not None else []
        self.workdir_stem = workdir_stem
        self.run_turbomole_dir = run_turbomole_dir
        unique_workdir = find_unique_name(self.workdir_stem, self.run_turbomole_dir,
                                          always_enumerate=True)
        work = os.path.join(os.path.abspath(self.run_turbomole_dir), unique_workdir)
        subprocess.run(self.command_prefix + ["cpc", work], cwd=self.run_turbomole_dir,
                       check=False)
        self.control = TurboControl(workdir=work, command_prefix=self.command_prefix)

        # read coordinates and elements from the control file
        elements, X = self.control.read_coords()
        nparts, ndims = X.shape
        self._position = X.flatten()
        self._elements = elements
        self.states = states
        ElectronicModel_.__init__(self, nstates=len(self.states), ndims=ndims, nparticles=nparts,
                                representation=representation, reference=reference)
        self.mass = self.control.get_masses(self._elements)

        self.expert = expert

        self.energies = np.zeros(self._nstates, dtype=np.float64)

        if not self.command_prefix:
            if not turbomole_is_installed():
                raise RuntimeError("Turbomole is not installed")

        if turbomole_modules is None:
            # always need energy and gradients
            mod = { "gs_energy": "ridft", "gs_grads": "rdgrad" }
            if any(s != 0 for s in self.states):
                mod["es_grads"] = "egrad"
            self.turbomole_modules = mod
        else:
            self.turbomole_modules = turbomole_modules
        if not self.command_prefix:
            if not all(shutil.which(x) is not None for x in self.turbomole_modules.values()):
                raise RuntimeError("Turbomole modules not found")

        # self.turbomole_init()

        if not self.expert:
            self.apply_suggested_parameters()


    def apply_suggested_parameters(self):
        """ Apply suggested parameters for Turbomole to work well with NAMD

        This function will update the control file to ensure that Turbomole
        is set up to work well with NAMD, including:
        - using pseudowavefunction couplings with ETFs
        - using weight derivatives
        - using phaser to adjust phases
        """

        # weight derivatives are mandatory
        self.control.use_weight_derivatives(use=True)

        # prefer psuedowavefunction couplings with ETFs
        sdg_nac = self.control.sdg("nacme")[6:].strip()  # skip past $nacme
        update_nac = False
        if "response" in sdg_nac:
            update_nac = True
            sdg_nac = sdg_nac.replace("response", "pseudo")
        if "do_etf" not in sdg_nac:
            update_nac = True
            sdg_nac += " do_etf"
        if update_nac:
            self.control.adg("nacme", sdg_nac)

        # probably force phaser on as well

    def update_coords(self, X):
        """ Update the coordinates in the control file

        :param X: numpy array of shape (n_atoms * 3) with coordinates in Bohr
        """
        X = X.reshape((self.ndof // 3, 3))
        coord_path = self.control.where_is_dg("coord", absolute_path=True)

        with open(coord_path, "r", encoding='utf-8') as f:
            lines = f.readlines()

        regex = re.compile(r"\s*\$coord")
        coordline = 0
        line = ""
        for i, line in enumerate(lines):
            if regex.match(line) is not None:
                coordline = i
                break

        # Reached end of file without finding $coord.
        if line == "":
            raise ValueError(f"$coord entry not found in file: {coord_path}!")

        coordline += 1
        for i, coord_list in enumerate(X):
            x, y, z = coord_list[:3]
            s = self._elements[i]
            lines[coordline] = f"{x:26.16e} {y:28.16e} {z:28.16e} {s:>7}\n"
            coordline += 1

        with open(coord_path, "w", encoding='utf-8') as coord_file:
            coord_file.write("".join(lines))

        # Now add results to model

    def call_turbomole(self, outname="turbo.out") -> None:
        """Call Turbomole to run the calculation"""
        # which forces are actually found?
        self._force = np.zeros((self.nstates, self.ndof))
        self._forces_available = np.zeros(self.nstates, dtype=bool)

        with open(outname, "w", encoding='utf-8') as f:
            for turbomole_module in self.turbomole_modules.values():
                self.control.run_single(turbomole_module, stdout=f)

        # Parse results with Turboparse
        with open(outname, "r", encoding='utf-8') as f:
            data_dict = turboparse.parse_turbo(f)

        # Now add results to model
        energy = data_dict[self.turbomole_modules["gs_energy"]]["energy"]
        self.energies[:] = 0.0
        self.energies[0] = energy
        dE0 = np.array(data_dict[self.turbomole_modules["gs_grads"]]["gradient"][0]["d/dR"])
        self._force[0,:] = -dE0.flatten()
        self._forces_available[0] = True

        # Check for presence of egrad turbomole module
        if "egrad" in self.turbomole_modules.values():
            # egrad updates to energy
            excited_energies = [
                data_dict["egrad"]["excited_state"][i]["energy"] + energy
                for i in range(len(data_dict["egrad"]["excited_state"]))
            ]
            for i, e in enumerate(excited_energies):
                self.energies[i+1] = e

            # egrad couplings
            parsed_nac_coupling = data_dict["egrad"]["coupling"]
            self._derivative_coupling = np.zeros((self.nstates, self.nstates, self.ndof))
            self._derivative_couplings_available = np.zeros((self.nstates, self.nstates),
                                                            dtype=bool)
            # Set diagonal elements to True since they are always zero
            for i in range(self.nstates):
                self._derivative_couplings_available[i,i] = True

            for dct in parsed_nac_coupling:
                i = dct["bra_state"]
                j = dct["ket_state"]

                ddr = np.array(dct["d/dR"]).flatten()
                self._derivative_coupling[i, j, :] = ddr
                self._derivative_coupling[j, i, :] = -(self._derivative_coupling[i, j, :])
                self._derivative_couplings_available[i, j] = True
                self._derivative_couplings_available[j, i] = True

            # egrad updates to gradients
            for i in range(len(data_dict["egrad"]["gradient"])):
                dE = np.array(data_dict["egrad"]["gradient"][i]["d/dR"])
                self._force[i+1,:] = -dE.flatten()
                self._forces_available[i+1] = True


    def compute(self, X, couplings: Any=None, gradients: Any=None, reference: Any=None) -> None:
        """
        Calls Turbomole/Turboparse to generate electronic properties including
        gradients, couplings, energies. For this to work, this class
        needs to know where the "original" files/data sit, so that they
        can get properly passed to Turbomole. (__init__() can get these
        file locations.)
        """
        self._position = X
        self.update_coords(X)
        self.call_turbomole(outname = Path(self.control.workdir)/"turbo.out")

        self._hamiltonian = np.zeros([self.nstates, self.nstates])
        self._hamiltonian = np.diag(self.energies)

    def clone(self):
        model_clone = cp.deepcopy(self)
        unique_workdir = find_unique_name(self.workdir_stem, self.run_turbomole_dir,
                                          always_enumerate=True)
        workdir = os.path.join(os.path.abspath(self.run_turbomole_dir), unique_workdir)
        model_clone.control.workdir = workdir
        os.makedirs(model_clone.control.workdir, exist_ok = True)
        self.control.cpc(model_clone.control.workdir)

        # necessary?
        # model_clone.turbomole_init()
        return model_clone
