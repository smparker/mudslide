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


class TestTMModel(unittest.TestCase):
    """Test Suite for TMModel class"""
    
    def setUp(self):
        self.turbomole_files = ["control"]
        for fl in self.turbomole_files:
            shutil.copy("turbomole_files/"+fl,".")

    def test_get_gs_ex_properties(self):
        """test for gs_ex_properties function"""
        model = TMModel(states = [0,1,2,3], turbomole_dir = ".") 
        positions = model.X 

        mom = [5.583286976987380000, -2.713959745507320000, 0.392059702162967000, 
                -0.832994241764031000, -0.600752326053757000, -0.384006560250834000, 
                -1.656414687719690000, 1.062437820195600000, -1.786171104341720000,
                -2.969087779972610000, 1.161804203506510000, -0.785009852486148000,
                2.145175145340160000, 0.594918215579156000, 1.075977514428970000,
                -2.269965412856570000,  0.495551832268249000,   1.487150300486560000]

        traj = mudslide.TrajectorySH(model, positions, mom, 3, tracer = YAMLTrace(name = "TMtrace"), dt = 20, max_time = 21, t0 = 1) 
        results = traj.simulate()

        with open("TMtrace-0-log_0.yaml", "r") as f:
            data = yaml.safe_load(f) 
            
            gs_e_from_ridft = data[0]["electronics"]["hamiltonian"][0][0]
            ex_e_1_from_egrad = data[0]["electronics"]["hamiltonian"][1][1] 
            ex_e_2_from_egrad = data[0]["electronics"]["hamiltonian"][2][2] 


            dm_from_mudslide_t1 = data[0]["density_matrix"]
            
            dm_from_mudslide_t2 = data[1]["density_matrix"]

            gs_grad_from_rdgrad = data[0]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad = data[0]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad = data[0]["electronics"]["force"][2]


            derivative_coupling01_from_egrad = data[0]["electronics"]["derivative_coupling"][1][0]
            derivative_coupling02_from_egrad = data[1]["electronics"]["derivative_coupling"][1][0]

        gs_energy_ref =  -78.40037178008 
        excited_1_energy_ref = -78.10536751497386 
        excited_2_energy_ref = -78.08798681828038  


        dm_t_1_ref = np.array([ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] ])
        

        dm_t_2_ref = np.array(
                        [ [1.750696427558272e-10, 0.0, 1.9635196878882362e-08, 1.0186172011065542e-07, 
                          -3.2512605768905584e-08, -1.1467681121179046e-06, -1.742901089655016e-06, 1.3065414000075634e-05], 
                        [1.9635196878882362e-08, -1.0186172011065542e-07, 6.146897206720771e-05, 0.0, -0.0006708768124209091, 
                        -0.00010970050206673946, 0.007406443044944121, 0.0024794525900529852], 
                        [-3.2512605768905584e-08, 1.1467681121179046e-06, -0.0006708768124209091, 0.00010970050206673946, 
                          0.007517774936800343, 0.0, -0.08525940680476202, -0.013843028457903528], 
                        [-1.742901089655016e-06, -1.3065414000075634e-05, 0.007406443044944121, -0.0024794525900529852, 
                          -0.08525940680476202, 0.013843028457903528, 0.9924207559160625, 0.0]])


        gs_gradients_ref = np.array([-2.929593e-10, -0.0, 0.01142713, -2.836152e-10, -0.0, 
                                -0.01142713, 0.01198766, -5.375224e-14, 0.007575883, 
                                -0.01198766, 5.362021e-14, 0.007575883, 0.01198766, 
                                -5.355601e-14, -0.007575883, -0.01198766, 5.358574e-14, -0.007575883])

        ex_st_1_gradients_ref = np.array([-2.55777e-10, -0.0, 0.01232717, -2.503296e-10, -0.0, -0.01232717, 
                                    0.02187082, -1.271217e-13, 0.03277518, -0.02187082, 1.2681e-13, 
                                    0.03277518, 0.02187082, -1.266736e-13, -0.03277518, -0.02187082, 
                                    1.268431e-13, -0.03277518])

        ex_st_2_gradients_ref = np.array([-2.975308e-10, -0.0, 0.08614947, -2.841102e-10, -0.0, -0.08614947, 
                                     0.03613404, -1.670112e-14, 0.009036918, -0.03613404, 1.675635e-14, 
                                     0.009036918, 0.03613404, -1.637944e-14, -0.009036918, -0.03613404, 
                                     1.656479e-14, -0.009036918])

        derivative_coupling01_ref = np.array([0.0, 6.959001e-11, 0.0, 0.0, 8.935593e-11, 0.0, 
                                                0.0, -0.1983295, 0.0, 0.0, 0.1983295, 0.0, 0.0, 
                                               -0.1983295, 0.0, 0.0, 0.1983295, 0.0])

        derivative_coupling02_ref = np.array([-0.001653841, 0.01234554, -0.0001328363, -0.0007634083, 
                                               -0.009828649, -0.0001557527, 0.0005532373, -0.1940909, 
                                                0.001014699, 0.0007221077, 0.1989962, -0.0009025696, 
                                                0.0004040651, -0.1972493, -0.0003609095, 0.0002632777, 0.1933652, 0.0004283189])

        self.assertAlmostEqual(gs_energy_ref, gs_e_from_ridft, places=6)

        self.assertAlmostEqual(excited_1_energy_ref, ex_e_1_from_egrad, places=6)
        self.assertAlmostEqual(excited_2_energy_ref, ex_e_2_from_egrad, places=6)
        np.testing.assert_almost_equal(dm_t_1_ref, dm_from_mudslide_t1,  decimal = 6)
#        np.testing.assert_almost_equal(dm_t_2_ref, dm_from_mudslide_t2,  decimal = 4)

        np.testing.assert_almost_equal(gs_gradients_ref, gs_grad_from_rdgrad,decimal = 6)
        np.testing.assert_almost_equal(ex_st_1_gradients_ref, ex_st_1_gradients_from_egrad,decimal = 6)
        np.testing.assert_almost_equal(ex_st_2_gradients_ref, ex_st_2_gradients_from_egrad,decimal = 6)

        np.testing.assert_almost_equal(derivative_coupling01_ref, derivative_coupling01_from_egrad, decimal = 6)
#        np.testing.assert_almost_equal(derivative_coupling02_ref, derivative_coupling02_from_egrad, decimal = 6)

    def tearDown(self):
        turbomole_files = ["TMtrace-0.yaml", "dipl_a", "ciss_a", "TMtrace-0-log_0.yaml", "TMtrace-0-events.yaml", "egradmonlog.1",  "excitationlog.1" ]
        for f in turbomole_files:
            os.remove(f)



if __name__ == '__main__':
    unittest.main()

