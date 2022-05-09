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

@unittest.skipUnless(mudslide.turbomole_model.turbomole_is_installed(),
        "Turbomole must be installed")
class TestTMModel(unittest.TestCase):
    """Test Suite for TMModel class"""
    def setUp(self):
        self.turbomole_files = ["control"]
        for fl in self.turbomole_files:
            shutil.copy(os.path.join(testdir, "turbomole_files", fl), ".")

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

        traj = mudslide.TrajectorySH(model, positions, mom, 3, tracer = YAMLTrace(base_name = "TMtrace"), dt = 20, max_time = 61, t0 = 1)
        results = traj.simulate()

        with open("TMtrace-0-log_0.yaml", "r") as f:
            data = yaml.safe_load(f) 
            
            gs_e_from_ridft_t1 = data[0]["electronics"]["hamiltonian"][0][0]
            ex_e_1_from_egrad_t1 = data[0]["electronics"]["hamiltonian"][1][1]
            ex_e_2_from_egrad_t1 = data[0]["electronics"]["hamiltonian"][2][2] 
            ex_e_3_from_egrad_t1 = data[0]["electronics"]["hamiltonian"][3][3] 

            gs_e_from_ridft_t21 = data[1]["electronics"]["hamiltonian"][0][0]
            ex_e_1_from_egrad_t21 = data[1]["electronics"]["hamiltonian"][1][1] 
            ex_e_2_from_egrad_t21 = data[1]["electronics"]["hamiltonian"][2][2] 
            ex_e_3_from_egrad_t21 = data[1]["electronics"]["hamiltonian"][3][3] 

            gs_e_from_ridft_t41   = data[2]["electronics"]["hamiltonian"][0][0]
            ex_e_1_from_egrad_t41 = data[2]["electronics"]["hamiltonian"][1][1] 
            ex_e_2_from_egrad_t41 = data[2]["electronics"]["hamiltonian"][2][2] 
            ex_e_3_from_egrad_t41 = data[2]["electronics"]["hamiltonian"][3][3] 

            gs_e_from_ridft_t61   = data[3]["electronics"]["hamiltonian"][0][0]
            ex_e_1_from_egrad_t61 = data[3]["electronics"]["hamiltonian"][1][1] 
            ex_e_2_from_egrad_t61 = data[3]["electronics"]["hamiltonian"][2][2] 
            ex_e_3_from_egrad_t61 = data[3]["electronics"]["hamiltonian"][3][3] 

            dm_from_mudslide_t1 = data[0]["density_matrix"]
            dm_from_mudslide_t21 = data[1]["density_matrix"]
            dm_from_mudslide_t41 = data[2]["density_matrix"]
            dm_from_mudslide_t61 = data[3]["density_matrix"]

            gs_grad_from_rdgrad_t1 =  data[0]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad_t1 = data[0]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad_t1 = data[0]["electronics"]["force"][2]
            ex_st_3_gradients_from_egrad_t1 = data[0]["electronics"]["force"][3]

            gs_grad_from_rdgrad_t21 = data[1]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad_t21 = data[1]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad_t21 = data[1]["electronics"]["force"][2]
            ex_st_3_gradients_from_egrad_t21 = data[1]["electronics"]["force"][3]

            gs_grad_from_rdgrad_t41 = data[2]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad_t41 = data[2]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad_t41 = data[2]["electronics"]["force"][2]
            ex_st_3_gradients_from_egrad_t41 = data[2]["electronics"]["force"][3]

            gs_grad_from_rdgrad_t61          = data[3]["electronics"]["force"][0] 
            ex_st_1_gradients_from_egrad_t61 = data[3]["electronics"]["force"][1]
            ex_st_2_gradients_from_egrad_t61 = data[3]["electronics"]["force"][2]
            ex_st_3_gradients_from_egrad_t61 = data[3]["electronics"]["force"][3]

            derivative_coupling02_from_egrad_t61 = data[3]["electronics"]["derivative_coupling"][1][0]

            coord_t1 = data[0]["position"]
            coord_t21 = data[1]["position"]
            coord_t41 = data[2]["position"]

            mom_t1 = data[0]["momentum"]
            mom_t21 = data[1]["momentum"]
            mom_t41 = data[2]["momentum"]
            mom_t61 = data[3]["momentum"]

        gs_energy_ref_t1    = -78.40037210973
        excited_1_energy_ref_t1 = -78.10524258057474
        excited_2_energy_ref_t1 = -78.08769019235675 
        excited_3_energy_ref_t1 = -78.07198337433016

        gs_energy_ref_t21 =        -78.40049099485
        excited_1_energy_ref_t21 = -78.10519738790673
        excited_2_energy_ref_t21 =  -78.0899939356086 
        excited_3_energy_ref_t21 =  -78.07351501250295 

        gs_energy_ref_t41 =       -78.39991638812
        excited_1_energy_ref_t41 = -78.10564877269677
        excited_2_energy_ref_t41 = -78.0928440234923 
        excited_3_energy_ref_t41 = -78.0757862108226  

        gs_energy_ref_t61 =  -78.39827192055
        excited_1_energy_ref_t61 =  -78.10606067527172
        excited_2_energy_ref_t61 = -78.09558657121471
        excited_3_energy_ref_t61 = -78.07846085927763  



        dm_t_21_ref = np.array(
                            [[1.731368149112295e-10, 0.0, -1.7257657961630577e-08, -1.2863146494508783e-07,
                             -3.488961834846973e-08, 1.1391211962251552e-06, -2.4980730290432937e-06, 1.2867819852528612e-05], 
                            [-1.7257657961630577e-08, 1.2863146494508783e-07, 9.728653343239662e-05, 0.0,
                             -0.0008428289222912595, -0.0001394646581272517, -0.009311108265497362, -0.003138553907619713], 
                            [-3.488961834846973e-08, -1.1391211962251552e-06, -0.0008428289222912595, 
                            0.0001394646581272517, 0.007501665003043152, 0.0, 0.0851648054480628, 
                            0.013842547667859188], [-2.4980730290432937e-06, -1.2867819852528612e-05,
                           -0.009311108265497362, 0.003138553907619713, 0.0851648054480628, -0.013842547667859188, 
                            0.9924010482903879, 0.0]])

        dm_t_41_ref = np.array(
                            [[4.75924465066229e-10, 0.0, -8.746908611550243e-08, -5.407524492814686e-07, 
                            -1.4619549093151148e-07, 3.060290568717866e-06, -8.726638697986608e-06, 1.9750529773541383e-05], 
                            [-8.746908611550247e-08, 5.407524492814686e-07, 0.000630486714710142, 5.421010862427522e-20, 
                            -0.0034502786788501607, -0.000728553404037578, -0.020836996132350523, -0.013545241971585348], 
                            [-1.4619549093151148e-07, -3.060290568717866e-06, -0.0034502786788501603, 0.0007285534040375782, 
                            0.019723195959774786, -8.673617379884035e-19, 0.12968056856080215, 0.05004704203290305], 
                            [-8.726638697986608e-06, -1.975052977354139e-05, -0.02083699613235053, 0.01354524197158535, 
                            0.12968056856080215, -0.05004704203290305, 0.979646316849589, 0.0]])

        dm_t_61_ref = np.array(
                            [[6.304958076851207e-10, -2.5849394142282115e-26, -1.0719227641746169e-07, -1.105303011046594e-06, 
                            -2.3542567557879155e-07, 3.997746088888918e-06, -1.588013983373362e-05, 1.9001220544437683e-05], 
                            [-1.0719227641746182e-07, 1.105303011046594e-06, 0.0019558971135428963, 0.0, -0.006968301647376584, 
                            -0.0010923850777543832, -0.030610636434811177, -0.031069438083747524], [-2.354256755787905e-07,
                             -3.997746088888918e-06, -0.006968301647376584, 0.001092385077754384, 0.025436170779358606,
                              1.734723475976807e-18, 0.12640948090821827, 0.09359521666683793], [-1.5880139833733623e-05,
                             -1.9001220544437693e-05, -0.030610636434811177, 0.031069438083747535, 0.12640948090821827,
                             -0.09359521666683793, 0.9726079314766009, 2.7755575615628914e-17]])

        gs_gradients_t1_ref = np.array(
                                    [-2.929282e-10, -0.0, 0.01145157, -2.83588e-10, -0.0, -0.01145157, 0.01200561, -0.0, 0.007576407,
                                     -0.01200561, -0.0, 0.007576407, 0.01200561, -0.0, -0.007576407, -0.01200561, -0.0, -0.007576407])



        ex_st_1_gradients_t1_ref = np.array(
                                     [-2.557119e-10, -0.0, 0.01241548, -2.502851e-10, -0.0, -0.01241548, 0.02188935, -0.0, 0.03277569,
                                     -0.02188935, -0.0, 0.03277569, 0.02188935, -0.0, -0.03277569, -0.02188935, -0.0, -0.03277569])



        ex_st_2_gradients_t1_ref = np.array(
                                    [-2.974144e-10, -0.0, 0.08607073, -2.840021e-10, -0.0, -0.08607073, 0.03626497, -0.0, 0.009054764, 
                                    -0.03626497, -0.0, 0.009054764, 0.03626497, -0.0, -0.009054764, -0.03626497, -0.0, -0.009054764]) 


        ex_st_3_gradients_t1_ref = np.array(
                                    [-3.025873e-10, -0.0, 0.02036686, -2.882881e-10, -0.0, -0.02036686, 0.03961653, -0.0, 0.02608287, 
                                    -0.03961653, -0.0, 0.02608287, 0.03961653, -0.0, -0.02608287, -0.03961653, -0.0, -0.02608287])


        gs_gradients_t61_ref = np.array(
                                    [-0.04621397, 0.002699175, 0.01454934, 0.007016793, 0.000808795, -0.02767231, 0.02275779, -0.0005581619, 
                                    0.01762088, 0.02238164, -0.001635568, -0.007693726, -0.01742144, -0.0007418236, 0.003344876, 
                                    0.01147919, -0.0005724168, -0.0001490589])




        ex_st_1_gradients_t61_ref = np.array(
                                    [-0.01642949, -0.003212607, 0.02238243, -0.009789118, -0.0005193657, -0.03680248, 
                                    0.02627548, 0.001782334, 0.0414943, 0.001772682, 0.001057347, 0.0130657, -0.002551177, 
                                    0.0004453547, -0.0166409, 0.0007216291, 0.0004469366, -0.02349905])


        ex_st_2_gradients_t61_ref = np.array(
                                    [-0.02344647, 0.002122081, 0.08910204, 0.02498179, 0.000469164, -0.100891, 0.0381589,
                                     -0.0007821404, 0.01570241, -0.0139162, -0.0009772865, -0.003941857, 0.003766447,
                                     -0.0004093713, 0.004798628, -0.02954447, -0.0004224467, -0.004770215])

        ex_st_3_gradients_t61_ref = np.array(
                                    [-0.0487975, 0.001524823, 0.03065595, -0.02884981, -0.001117058, -0.03907545, 
                                    0.04399117, -0.0008684287, 0.02685064, -0.005594196, 0.0001086628, 0.01680979,
                                     0.03666939, 0.001301242, -0.02936503, 0.002580944, -0.0009492408, -0.005875903])


        derivative_coupling02_t61_ref = np.array(
                                    [0.004871155, -0.03437384, 0.0001187429, 0.002142345, 0.03142824, 0.0007249875, 
                                    -0.001489359, 0.1809575, -0.002930825, -0.002102945, -0.1976619, 0.002395972, 
                                    -0.001130147, 0.1903838, 0.0009421799, -0.0007880266, -0.1812818, -0.001228199])


        coord_t1_ref = np.array([
                0.0, 0.0, 1.24876020687021, 0.0, 0.0, -1.24876020687021, 1.74402803906479, 
                0.0, 2.32753092373037, -1.74402803906479, 0.0, 2.32753092373037, 1.74402803906479, 
                0.0, -2.32753092373037, -1.74402803906479, 0.0, -2.32753092373037])

        coord_t21_ref = np.array([ 
                                0.00510479909557585, -0.00248137330552885, 1.2493048812371685, 
                                -0.0007616066103219365, -0.0005492678318356935, -1.2492975182497847,
                                 1.7303084436590455, 0.011566136971682432, 2.3109254162437445, 
                                -1.7806635741779635, 0.012647880465662243, 2.3218244711809746,
                                 1.7716941209038846, 0.006476525437573944, -2.3186568770230958,
                                 -1.7730526393031656, 0.005394781943594164, -2.314180680230991])

        coord_t41_ref = np.array([
                0.009800081041650078, -0.0049548135246493576, 1.2502463328978897, -0.0017741867329623966, 
                -0.0011060367415167604, -1.2503016318775682, 1.72674830149487, 0.023069416033800624, 
                2.300577632363366, -1.8233816008380466, 0.025320202670492398, 2.3210908994626487, 
                1.8085133921652625, 0.013036372149542137, -2.3160582443433073, -1.8074430436795406, 
                0.010739514953014905, -2.304951854182684])


        mom_t1_ref = np.array([  
                        5.58328697698738, -2.7139597455073203, 0.392059702162967, -0.832994241764031, 
                        -0.600752326053757, -0.3840065602508341, -1.65641468771969, 1.0624378201956, 
                        -1.78617110434172, -2.96908777997261, 1.16180420350651, -0.785009852486148, 
                            2.14517514534016, 0.594918215579156, 1.07597751442897, -2.26996541285657, 
                            0.49555183226824895, 1.48715030048656])

        mom_t21_ref = np.array([
                        5.359335873960455, -2.7096214075073197, 0.8127125021629669, -0.9702432446477031,
                       -0.6048544140537568, -0.842953660250834, -0.7936377877196902, 1.0595508321953926,
                       -1.23793260434172, -3.64461467997261, 1.1629267835067172, -0.2957826524861479, 
                        2.96173554534016, 0.5987450615789732, 0.5269265144289701, -2.9125757128565697,
                         0.4932531442684315, 1.0370298004865601])

        mom_t41_ref = np.array([
                        4.741185573960455, -2.6959808795073195, 1.2862893021629669, -1.3445870446477028, 
                        -0.6169194420537569, -1.4219873602508342, 0.15477171228030984, 1.0510773161953926, 
                        -0.6606408043417199, -4.08351017997261, 1.1654938695067172, 0.12709814751385207, 
                        3.7917063453401605, 0.6110864085789732, -0.05949438557102994, -3.2595665128565696,
                         0.4852427272684315, 0.72873490048656])

        mom_t61_ref = np.array([   
                        3.859011473960455, -2.6714304595073197, 1.8494414021629668, -1.8701799446477028, 
                        -0.6360529620537568, -2.136496960250834, 1.0764813122803099, 1.0368065021953927, 
                        -0.10225240434171996, -4.298986079972609, 1.1680250035067172, 0.4896783475138521, 
                        4.5679759453401605, 0.6326133295789732, -0.6513432855710298, -3.3343029128565695,
                         0.47003859226843153, 0.55097275048656])

        self.assertAlmostEqual(gs_energy_ref_t1, gs_e_from_ridft_t1, places=8)
        self.assertAlmostEqual(excited_1_energy_ref_t1, ex_e_1_from_egrad_t1, places=8)
        self.assertAlmostEqual(excited_2_energy_ref_t1, ex_e_2_from_egrad_t1, places=8)
        self.assertAlmostEqual(excited_3_energy_ref_t1, ex_e_3_from_egrad_t1, places=8)

        self.assertAlmostEqual(gs_energy_ref_t21, gs_e_from_ridft_t21, places=8)
        self.assertAlmostEqual(excited_1_energy_ref_t21, ex_e_1_from_egrad_t21, places=8)
        self.assertAlmostEqual(excited_2_energy_ref_t21, ex_e_2_from_egrad_t21, places=8)
        self.assertAlmostEqual(excited_3_energy_ref_t21, ex_e_3_from_egrad_t21, places=8)

        self.assertAlmostEqual(gs_energy_ref_t41, gs_e_from_ridft_t41, places=8)
        self.assertAlmostEqual(excited_1_energy_ref_t41, ex_e_1_from_egrad_t41, places=8)
        self.assertAlmostEqual(excited_2_energy_ref_t41, ex_e_2_from_egrad_t41, places=8)
        self.assertAlmostEqual(excited_3_energy_ref_t41, ex_e_3_from_egrad_t41, places=8)

        self.assertAlmostEqual(gs_energy_ref_t61, gs_e_from_ridft_t61, places=8)
        self.assertAlmostEqual(excited_1_energy_ref_t61, ex_e_1_from_egrad_t61, places=8)
        self.assertAlmostEqual(excited_2_energy_ref_t61, ex_e_2_from_egrad_t61, places=8)
        self.assertAlmostEqual(excited_3_energy_ref_t61, ex_e_3_from_egrad_t61, places=8)

        np.testing.assert_almost_equal(dm_t_21_ref, dm_from_mudslide_t21,  decimal = 8)
        np.testing.assert_almost_equal(dm_t_41_ref, dm_from_mudslide_t41,  decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t1_ref,   gs_grad_from_rdgrad_t1,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t1_ref, ex_st_1_gradients_from_egrad_t1,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t1_ref, ex_st_2_gradients_from_egrad_t1,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t1_ref, ex_st_3_gradients_from_egrad_t1,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t61_ref, gs_grad_from_rdgrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t61_ref, ex_st_1_gradients_from_egrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t61_ref, ex_st_2_gradients_from_egrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t61_ref, ex_st_3_gradients_from_egrad_t61,decimal = 8)

        np.testing.assert_almost_equal(derivative_coupling02_t61_ref, derivative_coupling02_from_egrad_t61, decimal = 8)

        np.testing.assert_almost_equal(coord_t1_ref, coord_t1, decimal = 8)
        np.testing.assert_almost_equal(coord_t21_ref, coord_t21, decimal = 8)
        np.testing.assert_almost_equal(coord_t41_ref, coord_t41, decimal = 8)

        np.testing.assert_almost_equal(mom_t1_ref, mom_t1, decimal = 8)
        np.testing.assert_almost_equal(mom_t21_ref, mom_t21, decimal = 8)
        np.testing.assert_almost_equal(mom_t41_ref, mom_t41, decimal = 8)
        np.testing.assert_almost_equal(mom_t61_ref, mom_t61, decimal = 8)

    def tearDown(self):
        turbomole_files = ["TMtrace-0.yaml", "control", "dipl_a", "ciss_a", "TMtrace-0-log_0.yaml", "TMtrace-0-events.yaml", "egradmonlog.1",  "excitationlog.1","energy",  "gradient",    "moments", "exspectrum"  ]
        for f in turbomole_files:
            os.remove(f)

if __name__ == '__main__':
    unittest.main()
