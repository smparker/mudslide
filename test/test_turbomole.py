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

from mudslide.models import TMModel
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

        traj = mudslide.TrajectorySH(model, positions, mom, 3, tracer = YAMLTrace(), dt = 20, max_time = 41, t0 = 1) 
        results = traj.simulate()



        gs_energy =  -78.40037178008 
        excited_1_energy = -78.10536751497386 
        excited_2_energy = -78.08798681828038  


        gs_gradients = np.array([2.929593e-10, 0.0, -0.01142713, 2.836152e-10, 0.0, 
                                0.01142713, -0.01198766, 5.375224e-14, -0.007575883, 
                                0.01198766, -5.362021e-14, -0.007575883, -0.01198766, 
                                5.355601e-14, 0.007575883, 0.01198766, -5.358574e-14, 0.007575883])







        ex_st_1_gradients = np.array([2.55777e-10, 0.0, -0.01232717, 2.503296e-10, 0.0, 0.01232717, 
                                    -0.02187082, 1.271217e-13, -0.03277518, 0.02187082, -1.2681e-13, 
                                    -0.03277518, -0.02187082, 1.266736e-13, 0.03277518, 0.02187082, 
                                    -1.268431e-13, 0.03277518])






        ex_st_2_gradients = np.array([2.975308e-10, 0.0, -0.08614947, 2.841102e-10, 0.0, 0.08614947, 
                                     -0.03613404, 1.670112e-14, -0.009036918, 0.03613404, -1.675635e-14, 
                                     -0.009036918, -0.03613404, 1.637944e-14, 0.009036918, 0.03613404, 
                                     -1.656479e-14, 0.009036918])







        derivative_coupling01 = np.array([0.0, 6.959001e-11, 0.0, 0.0, 8.935593e-11, 0.0, 
                                                0.0, -0.1983295, 0.0, 0.0, 0.1983295, 0.0, 0.0, 
                                               -0.1983295, 0.0, 0.0, 0.1983295, 0.0])


        derivative_coupling02 = np.array([0.0, 0.008271754, 0.0, 0.0, 0.008271754, 0.0, 0.0, 
                                               -0.02639676, 0.0, 0.0, -0.02639676, 0.0, 0.0, -0.02639676, 
                                                0.0, 0.0, -0.02639676, 0.0])





        derivative_coupling03 = np.array([0.0, -2.847169e-12, 0.0, 0.0, 2.998042e-12, 0.0, 0.0, 0.008101865, 
                                         0.0, 0.0, -0.008101865, 0.0, 0.0, 0.008101865, 0.0, 0.0, -0.008101865, 0.0])





        derivative_coupling12 = np.array([0.6964976, 0.0, -2.620811e-10, 0.6964976, 0.0, 2.889253e-10, 
                                             -0.3479412, 0.0, -0.1161213, -0.3479412, 0.0, 0.1161213, -0.3479412, 0.0, 0.1161213, -0.3479412, 0.0, -0.1161213])


        derivative_coupling13 = np.array([-1.939935e-10, 0.0, -0.008676887, -1.562714e-10, 0.0, 0.008676887, 
                                              0.08383228, 4.645253e-13, -0.008070732, -0.08383228, -4.639483e-13, 
                                             -0.008070731, 0.08383228, 4.341767e-13, 0.008070732, -0.08383228, -4.308535e-13, 0.008070731]) 


        derivative_coupling23 = np.array([-3.254183, 0.0, -4.535725e-10, -3.254183, 0.0, 9.549593e-11, 1.640597, 
                                               0.0, 0.5104488, 1.640597, 0.0, -0.5104488, 1.640597, 0.0, -0.5104488, 1.640597, 0.0, 0.5104488])




        gs_e_from_ridft = model.energies[0] 

        ex_e_1_from_egrad = model.energies[1]
        ex_e_2_from_egrad = model.energies[2]


        gs_grad_from_rdgrad = model.gradients[0] 


        ex_st_1_gradients_from_egrad = model.gradients[1]


        ex_st_2_gradients_from_egrad = model.gradients[2]

        

        derivative_coupling01_from_egrad = model.derivative_coupling[0][1]
        derivative_coupling02_from_egrad = model.derivative_coupling[0][2] 
        derivative_coupling03_from_egrad = model.derivative_coupling[0][3] 
        derivative_coupling12_from_egrad = model.derivative_coupling[1][2]
        derivative_coupling13_from_egrad = model.derivative_coupling[1][3]
        derivative_coupling23_from_egrad = model.derivative_coupling[2][3]

        self.assertAlmostEqual(gs_energy, gs_e_from_ridft, places=10)

        self.assertAlmostEqual(excited_1_energy, ex_e_1_from_egrad, places=10)
        self.assertAlmostEqual(excited_2_energy, ex_e_2_from_egrad, places=10)

        np.testing.assert_almost_equal(gs_gradients, gs_grad_from_rdgrad,decimal = 10)
        np.testing.assert_almost_equal(ex_st_1_gradients, ex_st_1_gradients_from_egrad,decimal = 10)
        np.testing.assert_almost_equal(ex_st_2_gradients, ex_st_2_gradients_from_egrad,decimal = 10)




        np.testing.assert_almost_equal(derivative_coupling01, derivative_coupling01_from_egrad, decimal = 10)
        np.testing.assert_almost_equal(derivative_coupling02, derivative_coupling02_from_egrad, decimal = 10)
        np.testing.assert_almost_equal(derivative_coupling03, derivative_coupling03_from_egrad, decimal = 10)
        np.testing.assert_almost_equal(derivative_coupling12, derivative_coupling12_from_egrad, decimal = 10)
        np.testing.assert_almost_equal(derivative_coupling13, derivative_coupling13_from_egrad, decimal = 5)
        np.testing.assert_almost_equal(derivative_coupling23, derivative_coupling23_from_egrad, decimal = 10)



    def test_get_coord(self):
        """test for getting coord function"""

        coord = np.array([    
    0.00000000000000,      0.00000000000000,      1.24876020687021, 
    0.00000000000000,      0.00000000000000,     -1.24876020687021,      
    1.74402803906479,      0.00000000000000,      2.32753092373037,      
   -1.74402803906479,      0.00000000000000,      2.32753092373037,      
    1.74402803906479,      0.00000000000000,     -2.32753092373037,      
   -1.74402803906479,      0.00000000000000,     -2.32753092373037,      
    ])

        coord_from_control = coord


        np.testing.assert_almost_equal(coord, coord_from_control ,decimal = 10)




    def test_density_matrix(self):
        dm_t_1 = np.array([ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] ])
        
        dm_t_2 = np.array(
                        [ [1.750696427558272e-10, 0.0, 1.9635196878882362e-08, 1.0186172011065542e-07, 
                          -3.2512605768905584e-08, -1.1467681121179046e-06, -1.742901089655016e-06, 1.3065414000075634e-05], 
                        [1.9635196878882362e-08, -1.0186172011065542e-07, 6.146897206720771e-05, 0.0, -0.0006708768124209091, 
                        -0.00010970050206673946, 0.007406443044944121, 0.0024794525900529852], 
                        [-3.2512605768905584e-08, 1.1467681121179046e-06, -0.0006708768124209091, 0.00010970050206673946, 
                          0.007517774936800343, 0.0, -0.08525940680476202, -0.013843028457903528], 
                        [-1.742901089655016e-06, -1.3065414000075634e-05, 0.007406443044944121, -0.0024794525900529852, 
                          -0.08525940680476202, 0.013843028457903528, 0.9924207559160625, 0.0]])



        dm_t_3 = np.array(
                        [[4.755043490192427e-10, 1.454028420503369e-26, 7.649642005146595e-08, 3.5835856618061715e-07, 
                        -1.3427109547686181e-07, -3.08127223073118e-06, -6.891696394388752e-06, 2.045390744453282e-05], 
                         [7.649642005146596e-08, -3.583585661806172e-07, 0.0002823792558630928, -2.710505431213761e-20, 
                         -0.0023437673262842137, -0.00039450553499091015, 0.014306163248332794, 0.008484358013417057], 
                        [-1.3427109547686176e-07, 3.08127223073118e-06, -0.002343767326284214, 0.00039450553499091026, 
                          0.02000458525053515, -8.673617379884035e-19, -0.13059544360067088, -0.05043394730559981], 
                        [-6.891696394388754e-06, -2.0453907444532825e-05, 0.01430616324833279, -0.008484358013417057, 
                         -0.13059544360067088, 0.050433947305599816, 0.9797130350180959, -1.3877787807814457e-17]])


        with open("traj-0-log_0.yaml", "r") as f:
            data = yaml.safe_load(f) 
            dm_from_mudslide_t1 = data[0]["density_matrix"]
            dm_from_mudslide_t2 = data[1]["density_matrix"]
            dm_from_mudslide_t3 = data[2]["density_matrix"]
    
        np.testing.assert_almost_equal(dm_t_1, dm_from_mudslide_t1,  decimal = 10)
        np.testing.assert_almost_equal(dm_t_2, dm_from_mudslide_t2,  decimal = 10)
        np.testing.assert_almost_equal(dm_t_3, dm_from_mudslide_t3,  decimal = 10)


    def tearDown(self):
        turbomole_files = ["control"] 




if __name__ == '__main__':
    unittest.main()

