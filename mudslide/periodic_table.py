#!/usr/bin/env python

from __future__ import print_function

import sys

import math as m
import numpy as np

DEBUG = False

# masses of most common isotopes to 3 decimal points, from
# http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
# yapf: disable
masses = {
    'x' : 0.00000,
    'h' : 1.00782503223,
    'he': 4.00260325413,
    'li': 7.0160034366,
    'be': 9.012183065,
    'b' : 11.00930536,
    'c' : 12.00000,
    'n' : 14.00307400443,
    'o' : 15.99491461957,
    'f' : 18.99840316273,
    'ne': 19.9924401762,
    'na': 22.9897692820,
    'mg': 23.985041697,
    'al': 26.98153853,
    'si': 27.97692653465,
    'p' : 30.97376199842,
    's' : 31.9720711744,
    'cl': 34.968852682,
    'ar': 39.9623831237,
    'k' : 38.9637064864,
    'ca': 39.962590863,
    'sc': 44.95590828,
    'ti': 47.94794198,
    'v' : 50.94395704,
    'cr': 51.94050623,
    'mn': 54.93804391,
    'fe': 55.93493633,
    'co': 58.93319429,
    'ni': 57.93534241,
    'cu': 62.92959772,
    'zn': 63.92914201,
    'ga': 68.9255735,
    'ge': 73.921177761,
    'as': 74.92159457,
    'se': 79.9165218,
    'br': 78.9183376,
    'kr': 83.9114977282,
    'rb': 84.9117897379,
    'sr': 87.9056125,
    'y':  88.9058403,
    'zr': 89.9046977,
    'nb': 92.9063730,
    'mo': 97.90540482,
    'tc': 98.9063667,
    'ru': 101.9043441,
    'rh': 102.9054980,
    'pd': 105.9034804,
    'ag': 106.9050916,
    'cd': 113.90336509,
    'in': 114.903878776,
    'sn': 119.90220163,
    'sb': 120.9038120,
    'te': 129.906222748,
    'i' : 126.9044719,
    'xe': 131.9041550856,
    'cs': 132.9054519610,
    'ba': 137.90524700,
    'la': 138.9063563,
    'ce': 139.9054431,
    'pr': 140.9076576,
    'nd': 141.9077290,
    'pm': 145.91395045, # no info about isotope abundance (mean of 145Pm and 147Pm)
    'sm': 151.9197397,
    'eu': 152.921238,
    'gd': 157.9241123,
    'tb': 158.9253547,
    'dy': 163.9291819,
    'ho': 164.9303329,
    'er': 165.9302995,
    'tm': 168.9342179,
    'yb': 173.9389664,
    'lu': 174.9407752,
    'hf': 179.9465570,
    'ta': 180.9479958,
    'w' : 183.9509309,
    're': 186.9557501,
    'os': 191.961477,
    'ir': 192.9629216,
    'pt': 194.9647917,
    'au': 196.9665688,
    'hg': 201.9706434,
    'tl': 204.9744278,
    'pb': 207.9766525,
    'bi': 208.9803991,
    'po': 209.4826525, # no info about isotope abundance (mean of 209Po and 210Po)
    'at': 210.4873223, # no info about isotope abundance (mean of 210At and 211At)
    'rn': 217.6731911, # no info about isotope abundance (mean of 211Rn, 220Rn, and 222Rn)
    'fr': 223.019736,
    'ra': 225.2737988, # no info about isotope abundance (mean of 223Ra, 224Ra, 226Ra, and 228Ra)
    'ac': 227.0277523,
    'th': 231.0355949, # no info about isotope abundance (mean of 230Th and 232Th)
    'pa': 231.0358842,
    'u' : 238.0507884,
    'np': 236.5473718, # no info about isotope abundance (mean of 236Np and 237Np)
    'pu': 240.7225563, # no info about isotope abundance (mean of 238Pu, 239Pu, 240Pu, 241Pu, 242Pu, and 244Pu)
    'am': 242.0591053, # no info about isotope abundance (mean of 242Am and 243Am)
    'cm': 245.5665936, # no info about isotope abundance (mean of 243Cm, 244Cm, 245Cm, 246Cm, 247Cm, and 248Cm)
    'bk': 248.0726475, # no info about isotope abundance (mean of 247Bk and 249Bk)
    'cf': 250.578119, # no info about isotope abundance (mean of 249Cf, 250Cf, 251Cf, and 252Cf)
    'es': 252.082980,
    'fm': 257.0951061,
    'md': 259.10104,
    'no': 259.10103,
    'lr': 262.10961,
    'rf': 267.12179,
    'db': 268.12567,
    'sg': 271.13393,
    'bh': 272.13826,
    'hs': 270.13429,
    'mt': 276.15159,
    'ds': 281.16451,
    'rg': 280.16514,
    'cn': 285.17712,
    'nh': 284.17873,
    'fl': 289.19042,
    'mc': 288.19274,
    'lv': 293.20449,
    'ts': 292.20746,
    'og': 294.21392
}
# yapf: enable
