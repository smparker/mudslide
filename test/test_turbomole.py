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

#testdir = os.path.dirname(__file__)
testdir = str(Path(__file__).parent)

@unittest.skipUnless(mudslide.turbomole_model.turbomole_is_installed(),
        "Turbomole must be installed")
class TestTMModel(unittest.TestCase):
    """Test Suite for TMModel class"""
    
    def setUp(self):
        self.turbomole_files = ["control"]
        for fl in self.turbomole_files:
            shutil.copy(testdir+"/turbomole_files/"+fl,".")

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
            coord_t61 = data[3]["position"]

            mom_t1 = data[0]["momentum"]
            mom_t21 = data[1]["momentum"]
            mom_t41 = data[2]["momentum"]
            mom_t61 = data[3]["momentum"]

        gs_energy_ref_t1    = -78.40037210973
        excited_1_energy_ref_t1 = -78.10524258057474
        excited_2_energy_ref_t1 = -78.08769019235673 
        excited_3_energy_ref_t1 = -78.07198337433016

        gs_energy_ref_t21 =   -78.40049111742
        excited_1_energy_ref_t21 = -78.1051974577254
        excited_2_energy_ref_t21 =  -78.08999361109902 
        excited_3_energy_ref_t21 =  -78.07351492132182 

        gs_energy_ref_t41 =  -78.39991698417
        excited_1_energy_ref_t41 = -78.1056489573636
        excited_2_energy_ref_t41 = -78.09284343760369 
        excited_3_energy_ref_t41 = -78.0757861096102  

        gs_energy_ref_t61 =  -78.398273272
        excited_1_energy_ref_t61 = -78.10606102227224
        excited_2_energy_ref_t61 = -78.09558585965726 
        excited_3_energy_ref_t61 = -78.07846067880071  

        dm_t_21_ref = np.array(
                            [[1.7300492094814887e-10, 0.0, -1.724719563480251e-08, -1.2855826725235406e-07, -3.4876573312516e-08, 
                              1.138445575057539e-06, -2.497195029947404e-06, 1.2862924959123944e-05], [-1.724719563480251e-08, 
                              1.2855826725235406e-07, 9.72497992773012e-05, 0.0, -0.0008424908760323563, -0.00013941028528649746, 
                             -0.00930936371344589, -0.003137971141351122], [-3.4876573312516e-08, -1.138445575057539e-06,
                             -0.0008424908760323563, 0.00013941028528649746, 0.007498484410873602, 0.0, 0.08514690522568664,
                              0.013839524757024784], [-2.497195029947404e-06, -1.2862924959123944e-05, -0.00930936371344589, 
                              0.003137971141351122, 0.08514690522568664, -0.013839524757024784, 0.9924042656168435, 0.0]])

        dm_t_41_ref = np.array(
                           [[4.756149238260524e-10, 1.2924697071141057e-26, -8.742249667872185e-08, -5.40461428511047e-07, 
                            -1.46173804234819e-07, 3.0588116956717945e-06, -8.723795824714387e-06, 1.9744185802136324e-05], 
                            [-8.742249667872181e-08, 5.40461428511047e-07, 0.0006302183418099965, 0.0, -0.0034489894607970385, 
                           -0.0007283418601894852, -0.02083264076785357, -0.013542386599985745], [-1.4617380423481937e-07,
                           -3.0588116956717945e-06, -0.0034489894607970385, 0.0007283418601894852, 0.01971699225748593,
                            -8.673617379884035e-19, 0.12966148408478495, 0.05003704626765577], [-8.723795824714387e-06, 
                            -1.9744185802136324e-05, -0.020832640767853558, 0.013542386599985743, 0.12966148408478495, 
                            -0.050037046267655774, 0.9796527889250883, -1.3877787807814457e-17]])

        dm_t_61_ref = np.array(
                            [[6.301992921350178e-10, -2.5849394142282115e-26, -1.0718827117388776e-07, -1.1048000353560558e-06, 
                            -2.3540757401621068e-07, 3.996472180511676e-06, -1.5875574272628806e-05, 1.89975302129778e-05], 
                            [-1.0718827117388763e-07, 1.104800035356056e-06, 0.0019550520906266304, 2.168404344971009e-19, 
                            -0.006966160911694898, -0.0010924373424594036, -0.030604281743096214, -0.031062629999918785], 
                            [-2.354075740162112e-07, -3.996472180511676e-06, -0.006966160911694898, 0.0010924373424594036, 
                            0.025431965436219773, -1.734723475976807e-18, 0.12640498396999564, 0.09358012483745562], 
                            [-1.5875574272628806e-05, -1.89975302129778e-05, -0.030604281743096214, 0.031062629999918774, 
                            0.12640498396999564, -0.09358012483745563, 0.9726129818429532, -5.551115123125783e-17]])

        gs_gradients_t1_ref = np.array(
                            [-2.929284e-10, -0.0, 0.01145157, -2.835931e-10, -0.0, -0.01145157, 0.01200561, 
                            -0.0, 0.007576407, -0.01200561, -0.0, 0.007576407, 0.01200561, -0.0, -0.007576407, 
                            -0.01200561, -0.0, -0.007576407]) 

        ex_st_1_gradients_t1_ref = np.array([
                            -2.556938e-10, -0.0, 0.01241548, -2.502705e-10, -0.0, -0.01241548, 0.02188935, 
                            -0.0, 0.03277569, -0.02188935, -0.0, 0.03277569, 0.02188935, -0.0, -0.03277569, 
                            -0.02188935, -0.0, -0.03277569])

        ex_st_2_gradients_t1_ref = np.array([
                            -2.973635e-10, -0.0, 0.08607073, -2.839574e-10, -0.0, -0.08607073, 0.03626497, 
                            -0.0, 0.009054764, -0.03626497, -0.0, 0.009054764, 0.03626497, -0.0, -0.009054764, 
                            -0.03626497, -0.0, -0.009054764]) 

        ex_st_3_gradients_t1_ref = np.array([
                            -3.026515e-10, -0.0, 0.02036686, -2.8836e-10, -0.0, -0.02036686, 0.03961653, 
                            -0.0, 0.02608287, -0.03961653, -0.0, 0.02608287, 0.03961653, -0.0, -0.02608287, 
                            -0.03961653, -0.0, -0.02608287])

        gs_gradients_t21_ref = np.array([
                            -0.01881827, 0.0008163477, 0.01074523, 0.0010609, 0.0001810409, -0.0157203, 0.01964748, 
                            -0.0002370792, 0.01339876, -0.001284034, -0.0004099352, 0.002683613, 0.004787766, 
                            -0.0001935213, -0.005415929, -0.005393842, -0.0001568529, -0.005691383])

        ex_st_1_gradients_t21_ref = np.array([
                            -0.008396862, -0.001268216, 0.01310096, -0.004792639, -0.0002855033, -0.0182614, 
                             0.02714792, 0.0005775356, 0.03849681, -0.01478725, 0.000551728, 0.02655111, 
                             0.01649112, 0.0002580281, -0.02925166, -0.01566229, 0.0001664275, -0.0306358]) 

        ex_st_2_gradients_t21_ref = np.array([
                            -0.008237319, 0.0006551376, 0.08600792, 0.01021447, 6.254457e-05, -0.08999608, 
                             0.03888107, -0.0002653978, 0.01296135, -0.03060261, -0.0002597005, 0.005202866, 
                             0.02656546, -8.382195e-05, -0.005484945, -0.03682106, -0.000108762, -0.008691105])

        ex_st_3_gradients_t21_ref = np.array([
                            -0.02238895, 0.0004336583, 0.02169873, -0.01372232, -0.0004100361, -0.02552762, 
                             0.04665892, -0.0002886274, 0.02874021, -0.02793896, 0.0001122793, 0.02284062, 
                             0.04203839, 0.0003825594, -0.02882135, -0.02464708, -0.0002298335, -0.01893058])


        gs_gradients_t41_ref = np.array([-0.03491805, 0.001693956, 0.01190502, 0.003235778, 0.0004438601, -0.0215996, 
                                          0.02345295, -0.0003920273, 0.01689011, 0.01063322, -0.000962715, -0.002635223, 
                                         -0.005308933, -0.0004336244, -0.001613286, 0.002905034, -0.0003494496, -0.002947018])


        ex_st_1_gradients_t41_ref = np.array(
                                [-0.0143182, -0.002384921, 0.01662066, -0.00838363, -0.0004696215, -0.02697068, 
                                0.02876781, 0.001213274, 0.04156422, -0.00650161, 0.0009070366, 0.01968649, 
                                0.008085887, 0.0004178463, -0.02364768, -0.007650252, 0.0003163865, -0.02725301])

        ex_st_2_gradients_t41_ref = np.array( 
                                [-0.01664355, 0.001348416, 0.0867666, 0.01865195, 0.0002074271, -0.095196, 0.03986531, 
                                -0.0005118308, 0.01546492, -0.02292415, -0.0005815455, 0.0007473742, 0.015291, -0.000219348, 
                                -0.0007017491, -0.03424056, -0.0002431191, -0.007081152])

        ex_st_3_gradients_t41_ref = np.array([-0.03941001, 0.0009298604, 0.02566012, -0.02370625, -0.0007959546, 
                                              -0.03237542, 0.04817669, -0.0005585188, 0.0289871, -0.01595811, 
                                               0.000144512, 0.01944935, 0.04095625, 0.0008511503, -0.02981858, 
                                              -0.01005857, -0.0005710493, -0.01190259])

        gs_gradients_t61_ref = np.array(
                                        [-0.04620223, 0.002698354, 0.01455387, 0.007012671, 0.0008087318, -0.02767439, 
                                        0.0227531, -0.0005580918, 0.01761796, 0.02237594, -0.001635009, -0.007691842, 
                                       -0.01741707, -0.0007416761, 0.00334377, 0.01147759, -0.0005723089, -0.0001493724]) 


        ex_st_1_gradients_t61_ref = np.array(
                                    [-0.01642933, -0.003211615, 0.02238539, -0.009784752, -0.0005195021, -0.03680232, 
                                      0.02627313, 0.001781739, 0.04149127, 0.001770496, 0.001057174, 0.01306885, 
                                     -0.002548523, 0.0004453704, -0.01664371, 0.0007189856, 0.000446834, -0.02349949])

        ex_st_2_gradients_t61_ref = np.array(
                                    [-0.02343859, 0.002121562, 0.08910774, 0.02497655, 0.0004690853, -0.1008938, 
                                    0.03815515, -0.0007819639, 0.01569964, -0.01391983, -0.0009769827, -0.003940523, 
                                    0.003771302, -0.0004093375, 0.004797253, -0.02954458, -0.0004223631, -0.004770268])

        ex_st_3_gradients_t61_ref = np.array(
                                    [-0.04878687, 0.001524276, 0.03065829, -0.02884802, -0.001116576, -0.0390765, 0.04398856, 
                                    -0.0008682145, 0.02684964, -0.005599453, 0.0001087589, 0.01681068, 0.0366689, 0.00130078, 
                                    -0.0293638, 0.002576876, -0.000949025, -0.005878302])

        derivative_coupling02_t61_ref = np.array(
                                                [0.004869645, -0.0343601, 0.0001189559, 0.002142044, 0.03141415,
                                                 0.0007245714, -0.001488959, 0.1809637, -0.002930135, -0.002102287, -0.19766, 
                                                 0.002395519, -0.001129933,     0.1903839, 0.0009418662, -0.0007878443, -0.1812871, -0.001227806])

        coord_t1_ref = np.array([
                                0.0, 0.0, 1.24876020687021, 0.0, 0.0, -1.24876020687021, 1.74402803906479, 0.0,
                                2.32753092373037, -1.74402803906479, 0.0, 2.32753092373037, 1.74402803906479, 0.0, 
                               -2.32753092373037, -1.74402803906479, 0.0, -2.32753092373037])

        coord_t21_ref = np.array([ 
                                0.005100060289556217, -0.0024790698364724607, 1.249304375613711, -0.000760899607770374, 
                               -0.0005487579442458317, -1.2492970194614186, 1.7303104168320083, 0.011564473512368861, 
                                2.310927804472948, -1.7806583051998905, 0.012646061428363182, 2.3218252918916114, 
                                1.7716901419261482, 0.006475593973889475, -2.3186581533019157, -1.7730484649412581, 
                                0.005394006057895185, -2.314182600283204])

        coord_t41_ref = np.array([
                                0.009791096305947211, -0.004950217158894037, 1.2502449589843767, -0.0017724926140315473, 
                               -0.0011050068479950793, -1.2503001967645113, 1.7267502990517016, 0.02306611372082485, 
                                2.30058134119985, -1.823370797640219, 0.025316565717591354, 2.321091993303329, 
                                1.8085038739636614, 0.013034469961353979, -2.3160597027880963, -1.8074344848957828, 
                                0.010737978064432604, -2.304955406314345])

        coord_t61_ref = np.array([
                                0.013762149889232408, -0.00740437683900746, 1.2516543277126555, -0.003217175679260338, 
                               -0.0016757970657290163, -1.251894841404568, 1.7336780996197088, 0.03444616609146138,
                                2.296545281110416, -1.8695573219136479, 0.03801852982808063, 2.3245927588387496, 
                                1.854233656602682, 0.019778638770897854, -2.319952666115756, -1.8440102248344077, 
                                0.015957634381551303, -2.2983193698545104])

        mom_t1_ref = np.array([  
                                5.58328697698738, -2.71395974550732, 0.392059702162967, -0.832994241764031, -0.600752326053757,
                               -0.384006560250834, -1.65641468771969, 1.0624378201956, -1.78617110434172, -2.96908777997261, 
                                1.16180420350651, -0.785009852486148, 2.14517514534016, 0.594918215579156, 1.07597751442897, 
                               -2.26996541285657, 0.49555183226824895, 1.48715030048656])

        mom_t21_ref = np.array([  
                               5.359397473960865, -2.70962316250732, 0.8127156021629669, -0.9702174446476309, -0.6048526870537569, 
                              -0.8429513602508341, -0.7936601877196898, 1.0595515461956, -1.2379403043417199, -3.6446426799726104, 
                               1.1629269965065099, -0.29577495248614794, 2.9617243453401603, 0.598743809579156, 0.52693531442897, 
                              -2.9126015128565697, 0.49325349726824896, 1.03701580048656])

        mom_t41_ref = np.array([   
                              4.741407873960865, -2.6959879755073204, 1.2863041021629669, -1.344503144647631, -0.6169125940537569, 
                             -1.421981760250834, 0.15469591228031024, 1.0510800841956, -0.66066720434172, -4.0836133799726095, 
                              1.16549490950651, 0.127124747513852, 3.7916707453401606, 0.6110809065791559, -0.05946398557102999, 
                             -3.25965801285657, 0.485244669268249, 0.72868410048656])

        mom_t61_ref = np.array([   
                             3.8594390739608655, -2.6714466115073203, 1.8494882021629668, -1.870045844647631, -0.6360379000537569, 
                            -2.136500960250834, 1.0763484122803102, 1.0368127511956, -0.10229980434171995, -4.29918900997261, 
                             1.16802761850651, 0.48972504751385204, 4.56792224534016, 0.632600209579156, -0.6512877855710301, 
                             -3.33447495285657, 0.470043926268249, 0.5508751804865599])

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

        np.testing.assert_almost_equal(gs_gradients_t21_ref, gs_grad_from_rdgrad_t21,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t21_ref, ex_st_1_gradients_from_egrad_t21,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t21_ref, ex_st_2_gradients_from_egrad_t21,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t21_ref, ex_st_3_gradients_from_egrad_t21,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t41_ref, gs_grad_from_rdgrad_t41,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t41_ref, ex_st_1_gradients_from_egrad_t41,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t41_ref, ex_st_2_gradients_from_egrad_t41,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t41_ref, ex_st_3_gradients_from_egrad_t41,decimal = 8)

        np.testing.assert_almost_equal(gs_gradients_t61_ref, gs_grad_from_rdgrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_1_gradients_t61_ref, ex_st_1_gradients_from_egrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_2_gradients_t61_ref, ex_st_2_gradients_from_egrad_t61,decimal = 8)
        np.testing.assert_almost_equal(ex_st_3_gradients_t61_ref, ex_st_3_gradients_from_egrad_t61,decimal = 8)

        np.testing.assert_almost_equal(derivative_coupling02_t61_ref, derivative_coupling02_from_egrad_t61, decimal = 8)

        np.testing.assert_almost_equal(coord_t1_ref, coord_t1, decimal = 8)
        np.testing.assert_almost_equal(coord_t21_ref, coord_t21, decimal = 8)
        np.testing.assert_almost_equal(coord_t41_ref, coord_t41, decimal = 8)
        np.testing.assert_almost_equal(coord_t61_ref, coord_t61, decimal = 8)

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
