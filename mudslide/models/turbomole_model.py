# -*- coding: utf-8 -*-
"""Implementations of the interface between turbomole and mudslide."""

import glob
import io
import os
import sys
import shlex
import shutil
import re
import copy as cp
import subprocess
import warnings

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from .. import turboparse

from ..util import find_unique_name
from ..constants import amu_to_au
from ..periodic_table import masses
from ..config import get_config
from .electronics import ElectronicModel_


def _resolve_command_prefix(explicit: Optional[str]) -> Optional[str]:
    """Resolve command_prefix: explicit arg > env var > config file."""
    if explicit is not None:
        return explicit
    env = os.environ.get("MUDSLIDE_TURBOMOLE_PREFIX")
    if env:
        return env
    return get_config("turbomole.command_prefix")


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

def turbomole_is_installed_or_prefixed():
    """ Check if turbomole is installed or a command prefix is set.

    :return: True if turbomole is installed or a command prefix is set, False otherwise
    """
    command_prefix = _resolve_command_prefix(None)
    return turbomole_is_installed() or (command_prefix is not None)


def verify_module_output(module: str, data_dict: dict, stderr: str) -> None:
    """Verify that a turbomole module ran through correctly.

    Checks stderr for normal termination and the parsed output to verify
    convergence of iterative procedures. Dispatches to module-specific
    verification functions.
    """
    if "ended normally" not in stderr:
        warnings.warn(f"{module} did not report normal termination")

    if module in ("ridft", "dscf"):
        _verify_scf(module, data_dict)
    elif module in ("egrad", "escf"):
        _verify_response(module, data_dict)


def _verify_scf(module: str, data_dict: dict) -> None:
    """Verify SCF convergence for ridft/dscf."""
    try:
        converged = data_dict[module]["converged"]
    except KeyError:
        warnings.warn(f"Convergence information not found for {module}")
        return
    if not converged:
        raise RuntimeError(f"{module} SCF did not converge")


def _verify_response(module: str, data_dict: dict) -> None:
    """Verify Davidson and CPKS convergence for egrad/escf."""
    for solver in ("davidson", "cpks"):
        try:
            converged = data_dict[module][solver]["converged"]
        except KeyError:
            warnings.warn(
                f"Convergence information for {solver} not found in {module} output"
            )
            continue
        if not converged:
            raise RuntimeError(f"{module} {solver} did not converge")


class TurboControl:
    """A class to handle the control file for turbomole"""

    # default filenames
    pcgrad_file = "pcgrad"
    control_file = "control"

    def __init__(self,
                 control_file=None,
                 workdir=None,
                 command_prefix: Optional[str] = None):
        self.command_prefix = _resolve_command_prefix(command_prefix)
        # workdir is directory of control file
        if control_file is not None:
            self.workdir = os.path.abspath(os.path.dirname(control_file))
        elif workdir is not None:
            self.workdir = os.path.abspath(workdir)
        else:
            raise ValueError("Must provide either control_file or workdir")
        self.control_file = control_file or "control"

        # make sure control file exists
        if not os.path.exists(os.path.join(self.workdir, self.control_file)):
            raise RuntimeError(
                f"control file not found in working directory {self.workdir:s}")

        # list of data groups and which file they are in, used to avoid rerunning sdg too much
        self.dg_in_file = {}

    def _build_command(self, cmd: list, cwd: Optional[str] = None) -> tuple:
        """Build command with prefix and working directory handling.

        When command_prefix is set, the entire command is run through
        ``sh -c`` so that shell constructs in the prefix (e.g. ``$(pwd)``)
        are evaluated at execution time. The working directory is embedded
        via ``cd`` so that it is respected inside containers.

        Returns (command_list, effective_cwd) tuple.
        """
        if self.command_prefix:
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
            if cwd is not None:
                shell_cmd = f"cd {shlex.quote(cwd)} && {self.command_prefix} {cmd_str}"
            else:
                shell_cmd = f"{self.command_prefix} {cmd_str}"
            return ["sh", "-c", shell_cmd], None
        return cmd, cwd

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

        full_cmd, effective_cwd = self._build_command(sdg_command.split(),
                                                      cwd=self.workdir)
        result = subprocess.run(full_cmd,
                                capture_output=True,
                                text=True,
                                cwd=effective_cwd,
                                check=False)
        return result.stdout.rstrip()

    def adg(self, dg, data, newline=False):
        """Convenience function to run add data group (adg) on a control"""
        if not isinstance(data, list):
            data = [data]
        lines = "\\n".join([f"{x}" for x in data])
        if newline:
            lines = "\\n" + lines
        adg_command = f"adg {dg} {lines}"
        full_cmd, effective_cwd = self._build_command(adg_command.split(),
                                                      cwd=self.workdir)
        result = subprocess.run(full_cmd,
                                capture_output=True,
                                text=True,
                                cwd=effective_cwd,
                                check=True)
        # check that the command ran successfully
        if "abnormal" in result.stderr:
            raise RuntimeError(f"Call to adg ended abnormally: {result.stderr}")

    def cpc(self, dest):
        """Copy the control file and other files to a new directory"""
        full_cmd, effective_cwd = self._build_command(["cpc", dest],
                                                      cwd=self.workdir)
        subprocess.run(full_cmd,
                       cwd=effective_cwd,
                       check=False,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        file_list = [
            'ciss_a', 'exspectrum', 'statistics', 'dipl_a', 'excitationlog.1',
            'moments', 'vecsao', 'control', 'gradient', 'energy', 'moments'
        ]
        for f in file_list:
            if os.path.exists(os.path.join(os.path.abspath(self.workdir), f)) and not \
                    os.path.exists(os.path.join(dest,f)):
                shutil.copy(os.path.join(os.path.abspath(self.workdir), f),
                            dest)

    def use_weight_derivatives(self, use=True):
        """Check if weight derivatives are used in the control file"""
        sdg_dft = self.sdg("dft", show_body=True)
        if use:  # make sure weight derivatives turned on
            if "weight derivatives" not in sdg_dft:
                current_dft = sdg_dft.split("\n")
                self.adg("dft",
                         current_dft + [" weight derivatives"],
                         newline=True)
        else:  # remove weight derivatives
            if "weight derivatives" in sdg_dft:
                current_dft = sdg_dft.split("\n")
                self.adg(
                    "dft",
                    [x for x in current_dft if "weight derivatives" not in x],
                    newline=True)

    def run_single(self,
                   module: str,
                   outname: Optional[Union[str, Path]] = None) -> dict:
        """Run a single turbomole module, verify its output, and return parsed results.

        :param module: name of the turbomole module to run
        :param outname: optional path to write the module output
        :return: parsed output dictionary from turboparse
        """
        full_cmd, effective_cwd = self._build_command([module],
                                                      cwd=self.workdir)
        output = subprocess.run(full_cmd,
                                capture_output=True,
                                text=True,
                                cwd=effective_cwd,
                                check=False)
        if outname is not None:
            with open(outname, "w", encoding='utf-8') as f:
                f.write(output.stdout)
        else:
            print(output.stdout)
        if "abnormal" in output.stderr:
            raise RuntimeError(f"Call to {module} ended abnormally")
        data_dict = turboparse.parse_turbo(io.StringIO(output.stdout))
        verify_module_output(module, data_dict, output.stderr)
        return data_dict

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
        atomic_masses = np.array([masses[s] for s in symbols for _ in range(3)],
                                 dtype=np.float64)
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
        with open(os.path.join(self.workdir, "point_charges"),
                  "w",
                  encoding='utf-8') as f:
            print("$point_charges nocheck list pcgrad", file=f)
            for xyz, q in zip(coords, charges):
                print(
                    f"{xyz[0]:22.16g} {xyz[1]:22.16g} {xyz[2]:22.16g} {q:22.16f}",
                    file=f)
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
        pcgrad = self.sdg("point_charge_gradients",
                          show_body=True,
                          show_keyword=False)
        pcgrad_list = pcgrad.rstrip().split("\n")
        gradients = np.zeros((nq, 3))
        for i, p in enumerate(pcgrad_list):
            p_list = p.split()
            gradients[i, :] = [
                float(val.replace('D', 'e')) for val in p_list[:3]
            ]

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
            expert:
        bool = False,  # when False, update turbomole parameters for NAMD
            turbomole_modules: Dict = None,
            command_prefix: Optional[str] = None,
            keep_output: int = 0,
            exopt_all: bool = False):
        self.command_prefix = _resolve_command_prefix(command_prefix)
        self.keep_output = keep_output
        self.workdir_stem = workdir_stem
        self.run_turbomole_dir = run_turbomole_dir
        unique_workdir = find_unique_name(self.workdir_stem,
                                          self.run_turbomole_dir,
                                          always_enumerate=True)
        abs_run_dir = os.path.abspath(self.run_turbomole_dir)
        work = os.path.join(abs_run_dir, unique_workdir)
        if self.command_prefix:
            cmd_str = " ".join(shlex.quote(c) for c in ["cpc", work])
            shell_cmd = f"cd {shlex.quote(abs_run_dir)} && {self.command_prefix} {cmd_str}"
            subprocess.run(["sh", "-c", shell_cmd],
                           check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["cpc", work],
                           cwd=self.run_turbomole_dir,
                           check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        self.control = TurboControl(workdir=work,
                                    command_prefix=self.command_prefix)

        # read coordinates and elements from the control file
        elements, X = self.control.read_coords()
        nparts, ndims = X.shape
        self._position = X.flatten()
        self.states = states
        ElectronicModel_.__init__(self,
                                  nstates=len(self.states),
                                  ndims=ndims,
                                  nparticles=nparts,
                                  atom_types=elements,
                                  representation=representation,
                                  reference=reference)
        self.mass = self.control.get_masses(elements)

        self.expert = expert

        self.energies = np.zeros(self._nstates, dtype=np.float64)

        if not self.command_prefix:
            if not turbomole_is_installed():
                raise RuntimeError("Turbomole is not installed")

        if turbomole_modules is None:
            # always need energy and gradients
            mod = {"gs_energy": "ridft", "gs_grads": "rdgrad"}
            if any(s != 0 for s in self.states):
                mod["es_grads"] = "egrad"
            self.turbomole_modules = mod
        else:
            self.turbomole_modules = turbomole_modules
        if not self.command_prefix:
            if not all(
                    shutil.which(x) is not None
                    for x in self.turbomole_modules.values()):
                raise RuntimeError("Turbomole modules not found")

        if not self.expert:
            self.apply_suggested_parameters()

        self.exopt_all = exopt_all

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
            s = self.atom_types[i]
            lines[coordline] = f"{x:26.16e} {y:28.16e} {z:28.16e} {s:>7}\n"
            coordline += 1

        with open(coord_path, "w", encoding='utf-8') as coord_file:
            coord_file.write("".join(lines))

        # Now add results to model

    def call_turbomole(self,
                       outname: Union[str, Path] = "tm.current",
                       gradients: Any = None) -> None:
        """Call Turbomole to run the calculation.

        Parameters
        ----------
        outname : str or Path, optional
            Output file path, by default "tm.current"
        gradients : list of int or None, optional
            Which state forces to compute. None means all.
            When a list is provided, only the specified states are
            marked as available. rdgrad is skipped when the ground
            state gradient is not requested.
        """
        # which forces are actually found?
        self._force = np.zeros((self.nstates, self.ndof))
        self._forces_available = np.zeros(self.nstates, dtype=bool)

        gradients = range(self.nstates) if (gradients is None or
                                            self.exopt_all) else gradients

        need_gs_grad = 0 in gradients

        # determine which modules to run -- skip rdgrad when ground
        # state gradient is not requested
        modules_to_run = {}
        modules_to_run["gs_energy"] = self.turbomole_modules["gs_energy"]
        if need_gs_grad and "gs_grads" in self.turbomole_modules:
            modules_to_run["gs_grads"] = self.turbomole_modules["gs_grads"]
        if "es_grads" in self.turbomole_modules:
            modules_to_run["es_grads"] = self.turbomole_modules["es_grads"]

        outpath = Path(outname)
        data_dict = {}
        module_outfiles = []

        # update control file to ensure requested gradients are computed
        excited_grads = [s for s in gradients if s > 0]
        if not excited_grads:
            excited_grads = [1]  # turbomole wants at least one excited state

        grad_str = ",".join(str(s) for s in excited_grads)
        self.control.adg("exopt", grad_str)

        for turbomole_module in modules_to_run.values():
            module_outpath = outpath.parent / f"tm.{turbomole_module}.current"
            module_data = self.control.run_single(turbomole_module,
                                                  outname=module_outpath)
            data_dict.update(module_data)
            module_outfiles.append(module_outpath)

        # Combine individual output files into one
        with open(outpath, "w", encoding='utf-8') as combined:
            for mf in module_outfiles:
                with open(mf, "r", encoding='utf-8') as inf:
                    combined.write(inf.read())
                mf.unlink()

        # Now add results to model
        energy = data_dict[modules_to_run["gs_energy"]]["energy"]
        self.energies[:] = 0.0
        self.energies[0] = energy

        if "gs_grads" in modules_to_run:
            grad = data_dict[modules_to_run["gs_grads"]]["gradient"][0]
            dE0 = np.array(grad["d/dR"]).flatten()
            self._force[0, :] = -dE0.flatten()
            self._forces_available[0] = True

        # Check for presence of egrad turbomole module
        if "es_grads" in modules_to_run and "egrad" in data_dict:
            # egrad updates to energy
            excited_energies = [
                data_dict["egrad"]["excited_state"][i]["energy"] + energy
                for i in range(len(data_dict["egrad"]["excited_state"]))
            ]
            for i, e in enumerate(excited_energies):
                self.energies[i + 1] = e

            # egrad couplings
            parsed_nac_coupling = data_dict["egrad"]["coupling"]
            self._derivative_coupling = np.zeros(
                (self.nstates, self.nstates, self.ndof))
            self._derivative_couplings_available = np.zeros(
                (self.nstates, self.nstates), dtype=bool)
            # Set diagonal elements to True since they are always zero
            for i in range(self.nstates):
                self._derivative_couplings_available[i, i] = True

            for dct in parsed_nac_coupling:
                i = dct["bra_state"]
                j = dct["ket_state"]

                ddr = np.array(dct["d/dR"]).flatten()
                self._derivative_coupling[i, j, :] = ddr
                self._derivative_coupling[
                    j, i, :] = -(self._derivative_coupling[i, j, :])
                self._derivative_couplings_available[i, j] = True
                self._derivative_couplings_available[j, i] = True

            # egrad updates to gradients
            for g in data_dict["egrad"]["gradient"]:
                state_idx = g["index"]
                dE = np.array(g["d/dR"]).flatten()
                self._force[state_idx, :] = -dE
                self._forces_available[state_idx] = True

        self._manage_output(outpath)

    def _manage_output(self, outpath: Path) -> None:
        """Rename the output file according to the keep_output policy.

        keep_output == 0:  rename to tm.last (overwriting previous)
        keep_output < 0:   rename to tm.{N} keeping all
        keep_output > 0:   rename to tm.{N} keeping only the last N
        """
        parent = outpath.parent

        # find existing numbered tm.N files
        existing = glob.glob(str(parent / "tm.[0-9]*"))
        numbers = []
        for f in existing:
            base = os.path.basename(f)
            parts = base.split(".", 1)
            if len(parts) == 2 and parts[1].isdigit():
                numbers.append(int(parts[1]))

        next_number = max(numbers) + 1 if numbers else 1

        if self.keep_output == 0:
            outpath.rename(parent / "tm.last")
        else:
            outpath.rename(parent / f"tm.{next_number}")

            if self.keep_output > 0:
                numbers.append(next_number)
                for n in sorted(numbers)[:-self.keep_output]:
                    (parent / f"tm.{n}").unlink(missing_ok=True)

    def compute(self,
                X,
                couplings: Any = None,
                gradients: Any = None,
                reference: Any = None) -> None:
        """
        Calls Turbomole/Turboparse to generate electronic properties including
        gradients, couplings, energies. For this to work, this class
        needs to know where the "original" files/data sit, so that they
        can get properly passed to Turbomole. (__init__() can get these
        file locations.)

        Parameters
        ----------
        X : ArrayLike
            Position at which to compute properties
        couplings : list of tuple(int, int) or None, optional
            Which coupling pairs to compute. None means all.
        gradients : list of int or None, optional
            Which state forces to compute. None means all.
        reference : Any, optional
            Reference state information.
        """
        self._position = X
        self.update_coords(X)
        self.call_turbomole(outname=Path(self.control.workdir) / "tm.current",
                            gradients=gradients)

        self._hamiltonian = np.zeros([self.nstates, self.nstates])
        self._hamiltonian = np.diag(self.energies)

    def compute_additional(self,
                           couplings: Any = None,
                           gradients: Any = None) -> None:
        """Compute additional gradients at the current geometry.

        Parameters
        ----------
        couplings : list of tuple(int, int) or None, optional
            Coupling pairs to compute. None means all.
        gradients : list of int or None, optional
            State indices whose forces are needed. None means all.
        """
        needed_g = self._needed_gradients(gradients)
        needed_c = self._needed_couplings(couplings)
        if not needed_g and not needed_c:
            return

        # If ground state gradient is needed but was not computed, run rdgrad
        if 0 in needed_g and "gs_grads" in self.turbomole_modules:
            outpath = Path(self.control.workdir) / "tm.additional.current"
            gs_module = self.turbomole_modules["gs_grads"]
            module_outpath = outpath.parent / f"tm.{gs_module}.current"
            module_data = self.control.run_single(gs_module,
                                                  outname=module_outpath)
            module_outpath.rename(outpath)

            dE0 = np.array(module_data[gs_module]["gradient"][0]["d/dR"])
            self._force[0, :] = -dE0.flatten()
            self._forces_available[0] = True

            self._manage_output(outpath)

        needed_excited_g = [s for s in needed_g if s != 0]
        # if any excited states needed but not computed, go back to egrad
        if needed_excited_g:
            # add additional excited states to exopt
            # keep the already computed ones just for consistency (for now)
            all_excited = set(needed_excited_g) | set(
                s for s in range(1, self.nstates) if self._forces_available[s])
            self.control.adg("exopt",
                             ",".join(str(s) for s in sorted(all_excited)))

            outpath = Path(self.control.workdir) / "tm.additional.current"
            es_module = self.turbomole_modules["es_grads"]
            module_outpath = outpath.parent / f"tm.{es_module}.current"
            module_data = self.control.run_single(es_module,
                                                  outname=module_outpath)
            module_outpath.rename(outpath)

            # egrad updates to gradients
            for g in module_data[es_module]["gradient"]:
                state_idx = g["index"]
                if state_idx in needed_excited_g:
                    dE = np.array(g["d/dR"]).flatten()
                    self._force[state_idx, :] = -dE
                    self._forces_available[state_idx] = True

        # all couplings computed for now
        for (i, j) in needed_c:
            self._derivative_couplings_available[i, j] = True

    def clone(self):
        model_clone = cp.deepcopy(self)
        unique_workdir = find_unique_name(self.workdir_stem,
                                          self.run_turbomole_dir,
                                          always_enumerate=True)
        workdir = os.path.join(os.path.abspath(self.run_turbomole_dir),
                               unique_workdir)
        model_clone.control.workdir = workdir
        os.makedirs(model_clone.control.workdir, exist_ok=True)
        self.control.cpc(model_clone.control.workdir)

        # necessary?
        # model_clone.turbomole_init()
        return model_clone
