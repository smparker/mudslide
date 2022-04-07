#!/usr/bin/env python


from __future__ import print_function

import sys

import math as m
import numpy as np

DEBUG = False

# masses of most common isotopes to 5 decimal points, from
# http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
masses = {
    'x' : 0.00000,
    'h' : 1.00783,
    'he': 4.00260,
    'li': 7.01600,
    'be': 9.01218,
    'b' : 11.00931,
    'c' : 12.00000,
    'n' : 14.00307,
    'o' : 15.99491,
    'f' : 18.99840,
    'ne': 19.99244,
    'na': 22.98977,
    'mg': 23.98504,
    'al': 26.98154,
    'si': 27.97693,
    'p' : 30.97376,
    's' : 31.97207,
    'cl': 34.96885,
    'ar': 39.96238,
    'k' : 38.96371,
    'ca': 39.96259,
    'sc': 44.95591,
    'ti': 47.94794,
    'v' : 50.94396,
    'cr': 51.94051,
    'mn': 54.93804,
    'fe': 55.93594,
    'co': 58.93319,
    'ni': 57.93534,
    'cu': 62.92960,
    'zn': 63.92914,
    'ga': 68.92557,
    'ge': 73.92118,
    'as': 74.92159,
    'se': 79.91652,
    'br': 78.91834,
    'kr': 83.91150,
    'rb': 84.91179,
    'sr': 87.90561,
    'y':  88.90584,
    'zr': 89.90469,
    'nb': 92.90637,
    'mo': 97.90541,
    'tc': 98.90637,
    'ru': 101.90434,
    'rh': 102.90550,
    'pd': 105.90348,
    'ag': 106.90509,
    'cd': 113.90337,
    'in': 114.90388,
    'sn': 119.90220,
    'sb': 120.90381,
    'te': 129.90622,
    'i' : 126.90447,
    'xe': 131.90416,
    'cs': 132.90545,
    'ba': 137.90525,
    'la': 138.90636,
    'ce': 139.90544,
    'pr': 140.90766,
    'nd': 141.90773,
    'pm': 145.91395, # no info about isotope abundance (mean of 145Pm and 147Pm)
    'sm': 151.91974,
    'eu': 152.92124,
    'gd': 157.92411,
    'tb': 158.92535,
    'dy': 163.92918,
    'ho': 164.93033,
    'er': 165.93030,
    'tm': 168.93422,
    'yb': 173.93987,
    'lu': 174.94077,
    'hf': 179.94656,
    'ta': 180.94799,
    'w' : 183.95093,
    're': 186.95575,
    'os': 191.96148,
    'ir': 192.96292,
    'pt': 194.96489,
    'au': 196.96657,
    'hg': 201.97064,
    'tl': 204.97443,
    'pb': 207.97665,
    'bi': 208.98040,
    'po': 209.48265, # no info about isotope abundance (mean of 209Po and 210Po)
    'at': 210.48732, # no info about isotope abundance (mean of 210At and 211At)
    'rn': 217.67319, # no info about isotope abundance (mean of 211Rn, 220Rn, and 222Rn)
    'fr': 223.01974,
    'ra': 225.27380, # no info about isotope abundance (mean of 223Ra, 224Ra, 226Ra, and 228Ra)
    'ac': 227.02775,
    'th': 231.03559, # no info about isotope abundance (mean of 230Th and 232Th)
    'pa': 231.03588,
    'u' : 238.05079,
    'np': 236.54737, # no info about isotope abundance (mean of 236Np and 237Np)
    'pu': 240.72255, # no info about isotope abundance (mean of 238Pu, 239Pu, 240Pu, 241Pu, 242Pu, and 244Pu)
    'am': 242.05911, # no info about isotope abundance (mean of 242Am and 243Am)
    'cm': 245.56659, # no info about isotope abundance (mean of 243Cm, 244Cm, 245Cm, 246Cm, 247Cm, and 248Cm)
    'bk': 248.07265, # no info about isotope abundance (mean of 247Bk and 249Bk)
    'cf': 250.57812, # no info about isotope abundance (mean of 249Cf, 250Cf, 251Cf, and 252Cf)
    'es': 252.08298,
    'fm': 257.09511,
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
