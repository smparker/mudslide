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

@unittest.skipUnless(mudslide.turbomole_model.turbomole_is_installed(),
        "Turbomole must be installed")
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

            gs_grad_from_rdgrad_t2 = data[1]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad_t2 = data[1]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad_t2 = data[1]["electronics"]["force"][2]

            derivative_coupling01_from_egrad = data[0]["electronics"]["derivative_coupling"][1][0]
            derivative_coupling02_from_egrad = data[1]["electronics"]["derivative_coupling"][1][0]

            coord_t1 = data[0]["position"]
            coord_t2 = data[1]["position"]

        gs_energy_ref =  -78.40037178008 
        excited_1_energy_ref = -78.10536751497386 
        excited_2_energy_ref = -78.08798681828038  

        dm_t_1_ref = np.array([ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] ])
        
        dm_t_2_ref = np.array(
                             [[1.7520369214582154e-10, 0.0, 1.9643997327888243e-08, 1.019123268497889e-07, 
                              -3.2523972663271146e-08, -1.147451155392712e-06, -1.7436744035150584e-06, 1.3070379402215022e-05], 
                            [1.9643997327888243e-08, -1.019123268497889e-07, 6.148277392459728e-05, 0.0, -0.000671094978562321, -0.00010973475194747686, 
                            0.007407264233254802, 0.0024797217938799946], [-3.2523972663271146e-08, 1.147451155392712e-06, 
                            -0.000671094978562321, 0.00010973475194747686, 0.007520971428576696, 0.0, -0.08527737369497834, -0.013846065922510479], 
                            [-1.7436744035150584e-06, -1.3070379402215022e-05, 0.007407264233254802, -0.0024797217938799946, -0.08527737369497834, 
                            0.013846065922510479, 0.9924175456222948, 0.0]] )

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

        gs_gradients_t2_ref  = np.array([-0.01882071, 0.0008168962, 0.01070758, 0.001060233, 0.0001804384, 
                                        -0.01568363, 0.01964265, -0.0002368164, 0.0134072, -0.001273064, 
                                        -0.0004099903, 0.002688204, 0.004777519, -0.0001934963, -0.00541991, 
                                        -0.005386627, -0.0001570316, -0.005699438])

        ex_st_1_gradients_t2_ref = np.array([-0.008393041, -0.001267764, 0.01299881, -0.004781268, -0.0002859212, 
                                            -0.01815933, 0.02713894, 0.0005779374, 0.03850215, -0.0147826, 0.0005513749, 
                                             0.0265579, 0.01647766, 0.0002578082, -0.02925562, -0.01565969, 0.0001665643, -0.03064392])

        ex_st_2_gradients_t2_ref = np.array([ -0.00830215, 0.0006506364, 0.08606266, 0.01015423, 5.456523e-05, -0.09005891, 
                                               0.03879733, -0.0002625103, 0.0129691, -0.03045149, -0.0002571386, 0.005181622, 
                                               0.02646374, -8.047436e-05, -0.00548443, -0.03666166, -0.0001050784, -0.008670045])

        derivative_coupling01_ref = np.array([0.0, 6.959001e-11, 0.0, 0.0, 8.935593e-11, 0.0, 
                                                0.0, -0.1983295, 0.0, 0.0, 0.1983295, 0.0, 0.0, 
                                               -0.1983295, 0.0, 0.0, 0.1983295, 0.0])

        derivative_coupling02_ref = np.array([-0.001653841, 0.01234554, -0.0001328363, -0.0007634083, 
                                               -0.009828649, -0.0001557527, 0.0005532373, -0.1940909, 
                                                0.001014699, 0.0007221077, 0.1989962, -0.0009025696, 
                                                0.0004040651, -0.1972493, -0.0003609095, 0.0002632777, 0.1933652, 0.0004283189])

        coord_t1_ref = np.array([     
                                                0.00000000000000,      0.00000000000000,      1.24876020687021, 
                                                0.00000000000000,      0.00000000000000,     -1.24876020687021, 
                                                1.74402803906479,      0.00000000000000,      2.32753092373037,      
                                               -1.74402803906479,      0.00000000000000,      2.32753092373037, 
                                                1.74402803906479,      0.00000000000000,     -2.32753092373037, 
                                               -1.74402803906479,      0.00000000000000,     -2.32753092373037, 
                                                ])

        coord_t2_ref = np.array([               0.0051047984341238434, -0.002481372984005959, 1.24930724947749, 
                                               -0.000761606511636359, -0.0005492677606645478, -1.2492998864910603, 
                                                1.7302893093023415, 0.011566135473005156, 2.3109143719534857, 
                                               -1.780644433296515, 0.012647878826818514, 2.3218134254784752, 
                                                1.771674981184649, 0.006476524598380916, -2.318645831731036, 
                                               -1.7730334994079007, 0.00539478124456759, -2.3141696355189327
                                                ])

        self.assertAlmostEqual(gs_energy_ref, gs_e_from_ridft, places=8)

        self.assertAlmostEqual(excited_1_energy_ref, ex_e_1_from_egrad, places=8)
        self.assertAlmostEqual(excited_2_energy_ref, ex_e_2_from_egrad, places=8)
        np.testing.assert_almost_equal(dm_t_1_ref, dm_from_mudslide_t1,  decimal = 8)
        np.testing.assert_almost_equal(dm_t_2_ref, dm_from_mudslide_t2,  decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_ref, gs_grad_from_rdgrad,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_ref, ex_st_1_gradients_from_egrad,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_ref, ex_st_2_gradients_from_egrad,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t2_ref, gs_grad_from_rdgrad_t2,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t2_ref, ex_st_1_gradients_from_egrad_t2,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t2_ref, ex_st_2_gradients_from_egrad_t2,decimal = 8)

        np.testing.assert_almost_equal(derivative_coupling01_ref, derivative_coupling01_from_egrad, decimal = 8)
        np.testing.assert_almost_equal(derivative_coupling02_ref, derivative_coupling02_from_egrad, decimal = 8)

        np.testing.assert_almost_equal(coord_t1_ref, coord_t1, decimal = 8)
        np.testing.assert_almost_equal(coord_t2_ref, coord_t2, decimal = 8)

    def tearDown(self):
        turbomole_files = ["TMtrace-0.yaml", "dipl_a", "ciss_a", "TMtrace-0-log_0.yaml", "TMtrace-0-events.yaml", "egradmonlog.1",  "excitationlog.1" ]
        for f in turbomole_files:
            os.remove(f)

if __name__ == '__main__':
    unittest.main()
