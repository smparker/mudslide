#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for Turbomole_class"""

import numpy as np
import os
import shutil
import unittest
import sys
from pathlib import Path
import mudslide
import yaml

from mudslide import TMModel
from mudslide.tracer import YAMLTrace

testdir = os.path.dirname(__file__)
_refdir = os.path.join(testdir, "ref")
_checkdir = os.path.join(testdir, "checks")

def clean_directory(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

@unittest.skipUnless(mudslide.turbomole_model.turbomole_is_installed(),
        "Turbomole must be installed")
class TestTMModel(unittest.TestCase):
    """Test Suite for TMModel class"""

    def setUp(self):
        self.refdir = os.path.join(_refdir, "tm-c2h4")

        self.rundir = os.path.join(_checkdir, "tm-c2h4")

        clean_directory(self.rundir)
        os.makedirs(self.rundir, exist_ok=True)

        self.origin = os.getcwd()

        os.chdir(self.rundir)
        with os.scandir(self.refdir) as it:
            for fil in it:
                if fil.name.endswith(".input") and fil.is_file():
                    filename = fil.name[:-6]
                    shutil.copy(os.path.join(self.refdir, fil.name), filename)

    def test_get_gs_ex_properties(self):
        """test for gs_ex_properties function"""
        model = TMModel(states = [0,1,2,3], turbomole_dir = ".", expert=True)
        positions = model.X

        mom = [5.583286976987380000, -2.713959745507320000, 0.392059702162967000,
                -0.832994241764031000, -0.600752326053757000, -0.384006560250834000,
                -1.656414687719690000, 1.062437820195600000, -1.786171104341720000,
                -2.969087779972610000, 1.161804203506510000, -0.785009852486148000,
                2.145175145340160000, 0.594918215579156000, 1.075977514428970000,
                -2.269965412856570000,  0.495551832268249000,   1.487150300486560000]

        log = mudslide.YAMLTrace(base_name="TMtrace", location=self.rundir, log_pitch=8)
        traj = mudslide.TrajectorySH(model, positions, mom, 3,tracer=log,  dt = 20, max_time =81, t0 = 1)
        results = traj.simulate()

        main_log = results.main_log

        assert main_log == "TMtrace-0.yaml"

        refs = mudslide.load_log(os.path.join(self.refdir, "traj-0.yaml"))

        gs_e_from_ridft_t1 = results[0]["electronics"]["hamiltonian"][0][0]
        ex_e_1_from_egrad_t1 = results[0]["electronics"]["hamiltonian"][1][1]
        ex_e_2_from_egrad_t1 = results[0]["electronics"]["hamiltonian"][2][2]
        ex_e_3_from_egrad_t1 = results[0]["electronics"]["hamiltonian"][3][3]

        gs_e_from_ridft_t21 = results[1]["electronics"]["hamiltonian"][0][0]
        ex_e_1_from_egrad_t21 = results[1]["electronics"]["hamiltonian"][1][1]
        ex_e_2_from_egrad_t21 = results[1]["electronics"]["hamiltonian"][2][2]
        ex_e_3_from_egrad_t21 = results[1]["electronics"]["hamiltonian"][3][3]

        gs_e_from_ridft_t41   = results[2]["electronics"]["hamiltonian"][0][0]
        ex_e_1_from_egrad_t41 = results[2]["electronics"]["hamiltonian"][1][1]
        ex_e_2_from_egrad_t41 = results[2]["electronics"]["hamiltonian"][2][2]
        ex_e_3_from_egrad_t41 = results[2]["electronics"]["hamiltonian"][3][3]

        gs_e_from_ridft_t61   = results[3]["electronics"]["hamiltonian"][0][0]
        ex_e_1_from_egrad_t61 = results[3]["electronics"]["hamiltonian"][1][1]
        ex_e_2_from_egrad_t61 = results[3]["electronics"]["hamiltonian"][2][2]
        ex_e_3_from_egrad_t61 = results[3]["electronics"]["hamiltonian"][3][3]

        dm_from_mudslide_t1 = results[0]["density_matrix"]
        dm_from_mudslide_t21 = results[1]["density_matrix"]
        dm_from_mudslide_t41 = results[2]["density_matrix"]
        dm_from_mudslide_t61 = results[3]["density_matrix"]

        gs_grad_from_rdgrad_t1 =  results[0]["electronics"]["force"][0]
        ex_st_1_gradients_from_egrad_t1 = results[0]["electronics"]["force"][1]
        ex_st_2_gradients_from_egrad_t1 = results[0]["electronics"]["force"][2]
        ex_st_3_gradients_from_egrad_t1 = results[0]["electronics"]["force"][3]

        gs_grad_from_rdgrad_t21 = results[1]["electronics"]["force"][0]
        ex_st_1_gradients_from_egrad_t21 = results[1]["electronics"]["force"][1]
        ex_st_2_gradients_from_egrad_t21 = results[1]["electronics"]["force"][2]
        ex_st_3_gradients_from_egrad_t21 = results[1]["electronics"]["force"][3]

        gs_grad_from_rdgrad_t41 = results[2]["electronics"]["force"][0]
        ex_st_1_gradients_from_egrad_t41 = results[2]["electronics"]["force"][1]
        ex_st_2_gradients_from_egrad_t41 = results[2]["electronics"]["force"][2]
        ex_st_3_gradients_from_egrad_t41 = results[2]["electronics"]["force"][3]

        gs_grad_from_rdgrad_t61 = results[3]["electronics"]["force"][0]
        ex_st_1_gradients_from_egrad_t61 = results[3]["electronics"]["force"][1]
        ex_st_2_gradients_from_egrad_t61 = results[3]["electronics"]["force"][2]
        ex_st_3_gradients_from_egrad_t61 = results[3]["electronics"]["force"][3]

        derivative_coupling02_from_egrad_t61 = results[3]["electronics"]["derivative_coupling"][1][0]

        coord_t1 = results[0]["position"]
        coord_t21 = results[1]["position"]
        coord_t41 = results[2]["position"]

        mom_t1 = results[0]["momentum"]
        mom_t21 = results[1]["momentum"]
        mom_t41 = results[2]["momentum"]
        mom_t61 = results[3]["momentum"]

        gs_energy_ref_t1    = refs[0]["electronics"]["hamiltonian"][0][0]
        excited_1_energy_ref_t1  = refs[0]["electronics"]["hamiltonian"][1][1]
        excited_2_energy_ref_t1  = refs[0]["electronics"]["hamiltonian"][2][2]
        excited_3_energy_ref_t1  = refs[0]["electronics"]["hamiltonian"][3][3]

        gs_energy_ref_t21    = refs[1]["electronics"]["hamiltonian"][0][0]
        excited_1_energy_ref_t21  = refs[1]["electronics"]["hamiltonian"][1][1]
        excited_2_energy_ref_t21  = refs[1]["electronics"]["hamiltonian"][2][2]
        excited_3_energy_ref_t21  = refs[1]["electronics"]["hamiltonian"][3][3]

        gs_energy_ref_t41    = refs[2]["electronics"]["hamiltonian"][0][0]
        excited_1_energy_ref_t41  = refs[2]["electronics"]["hamiltonian"][1][1]
        excited_2_energy_ref_t41  = refs[2]["electronics"]["hamiltonian"][2][2]
        excited_3_energy_ref_t41  = refs[2]["electronics"]["hamiltonian"][3][3]

        gs_energy_ref_t61    = refs[3]["electronics"]["hamiltonian"][0][0]
        excited_1_energy_ref_t61  = refs[3]["electronics"]["hamiltonian"][1][1]
        excited_2_energy_ref_t61  = refs[3]["electronics"]["hamiltonian"][2][2]
        excited_3_energy_ref_t61  = refs[3]["electronics"]["hamiltonian"][3][3]

        dm_t_21_ref = refs[1]["density_matrix"]
        dm_t_41_ref = refs[2]["density_matrix"]
        dm_t_61_ref = refs[3]["density_matrix"]

        gs_gradients_t1_ref      = refs[0]["electronics"]["force"][0]
        ex_st_1_gradients_t1_ref = refs[0]["electronics"]["force"][1]
        ex_st_2_gradients_t1_ref = refs[0]["electronics"]["force"][2]
        ex_st_3_gradients_t1_ref = refs[0]["electronics"]["force"][3]

        gs_gradients_t21_ref      = refs[1]["electronics"]["force"][0]
        ex_st_1_gradients_t21_ref = refs[1]["electronics"]["force"][1]
        ex_st_2_gradients_t21_ref = refs[1]["electronics"]["force"][2]
        ex_st_3_gradients_t21_ref = refs[1]["electronics"]["force"][3]

        gs_gradients_t41_ref      = refs[2]["electronics"]["force"][0]
        ex_st_1_gradients_t41_ref = refs[2]["electronics"]["force"][1]
        ex_st_2_gradients_t41_ref = refs[2]["electronics"]["force"][2]
        ex_st_3_gradients_t41_ref = refs[2]["electronics"]["force"][3]

        gs_gradients_t61_ref      = refs[3]["electronics"]["force"][0]
        ex_st_1_gradients_t61_ref = refs[3]["electronics"]["force"][1]
        ex_st_2_gradients_t61_ref = refs[3]["electronics"]["force"][2]
        ex_st_3_gradients_t61_ref = refs[3]["electronics"]["force"][3]

        derivative_coupling02_t61_ref = refs[3]["electronics"]["derivative_coupling"][1][0]

        coord_t1_ref = refs[0]["position"]
        coord_t21_ref = refs[1]["position"]
        coord_t41_ref = refs[2]["position"]
        coord_t61_ref = refs[3]["position"]

        mom_t1_ref = refs[0]["momentum"]
        mom_t21_ref = refs[1]["momentum"]
        mom_t41_ref = refs[2]["momentum"]
        mom_t61_ref = refs[3]["momentum"]

        ref_times = [ 0, 1, 2, 3]
        states = [ 0, 1, 2, 3 ]

        for t in ref_times:
            for s in states:
                self.assertAlmostEqual(refs[t]["electronics"]["hamiltonian"][s][s],results[t]["electronics"]["hamiltonian"][s][s], places = 8)
                np.testing.assert_almost_equal(refs[t]["electronics"]["force"][s],results[t]["electronics"]["force"][s], decimal = 8)

        for t in ref_times[1:]:
                np.testing.assert_almost_equal(refs[t]["density_matrix"],results[t]["density_matrix"], decimal = 8)
                np.testing.assert_almost_equal(refs[t]["position"],results[t]["position"], decimal = 8)
                np.testing.assert_almost_equal(refs[t]["momentum"],results[t]["momentum"], decimal = 8)

        for t in ref_times:
            for s1 in states:
                for s2 in range(s1,3):
                    np.testing.assert_almost_equal(refs[t]["electronics"]["derivative_coupling"][s1][s2], results[t]["electronics"]["derivative_coupling"][s1][s2], decimal = 8)

    def tearDown(self):
        os.chdir(self.origin)

if __name__ == '__main__':
    unittest.main()
