#!/usr/bin/env python

import re

from .section_parser import ParseSection

FLT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def try_to_fill_with_float(dict_obj, key, value):
    """Try to fill a dict with a float, but do nothing if it fails."""
    try:
        dict_obj[key] = float(value)
    except:
        pass

class FreeHData(ParseSection):
    name = ''

    start1st_re = re.compile(r"T\s+p\s+ln\(qtrans\)\s+ln\(qrot\)\s+ln\(qvib\)\s+chem\.pot\.\s+energy\s+entropy")
    data1st_re = re.compile(rf"({FLT})\s+({FLT})\s+({FLT})\s+({FLT})\s+({FLT})\s+({FLT})\s+({FLT})\s+({FLT})")
    start2nd_re = re.compile(r"\(K\)\s+\(MPa\)\s+\(kJ/mol-K\)\s+\(kJ/mol-K\)(?:\s+\(kJ/mol\))?")
    data2nd_re = re.compile(rf"({FLT})\s+({FLT})\s+({FLT})\s+({FLT})\s*({FLT})?")
    end_re = re.compile(r"\*{50}")

    def __init__(self):
        super().__init__(
            r"T\s+p\s+ln\(qtrans\)\s+ln\(qrot\)\s+ln\(qvib\)\s+chem\.pot\.\s+energy\s+entropy",
            r"\*{50}",
            multi=False)

    def parse_driver(self, liter, out):
        """
        Driver to FreeH section

        return: advanced (whether next has been called on liter)
        """
        data = []
        done = False
        advanced = False
        while not done:
            m = self.start1st_re.search(liter.top())

            if m:
                next(liter)
                advanced = True
                done1st = False
                while not done1st:
                    m1st = self.data1st_re.search(liter.top())
                    if m1st:
                        new_data = {}
                        try_to_fill_with_float(new_data, 'T', m1st.group(1))
                        try_to_fill_with_float(new_data, 'P', m1st.group(2))
                        try_to_fill_with_float(new_data, 'qtrans', m1st.group(3))
                        try_to_fill_with_float(new_data, 'qrot', m1st.group(4))
                        try_to_fill_with_float(new_data, 'qvib', m1st.group(5))
                        try_to_fill_with_float(new_data, 'chem.pot.', m1st.group(6))
                        try_to_fill_with_float(new_data, 'energy', m1st.group(7))
                        try_to_fill_with_float(new_data, 'entropy', m1st.group(8))
                        data.append(new_data)

                    mend = self.start2nd_re.search(liter.top())
                    if mend:
                        done1st = True
                    else:
                        next(liter)

            m = self.start2nd_re.search(liter.top())

            if m:
                next(liter)
                advanced = True
                done2nd = False
                i = 0
                while not done2nd:
                    m2nd = self.data2nd_re.search(liter.top())
                    if m2nd:
                        try_to_fill_with_float(data[i], 'T', m2nd.group(1))
                        try_to_fill_with_float(data[i], 'P', m2nd.group(2))
                        try_to_fill_with_float(data[i], 'Cv', m2nd.group(3))
                        try_to_fill_with_float(data[i], 'Cp', m2nd.group(4))
                        try_to_fill_with_float(data[i], 'H', m2nd.group(5))
                        i += 1

                    if self.test_tail(liter.top()):
                        done2nd = True
                    else:
                        next(liter)
                    mend = self.start2nd_re.search(liter.top())

            if self.test_tail(liter.top()):
                done = True
            else:
                next(liter)
                advanced = True
        out['data'] = data
        out['units'] = {
            'T': 'K',
            'P': 'MPa',
            'qtrans': '',
            'qrot': '',
            'qvib': '',
            'chem.pot.': 'kJ/mol',
            'energy': 'kJ/mol',
            'entropy': 'kJ/mol/K',
            'enthalpy': 'kJ/mol',
            'Cv': 'kJ/mol-K',
            'Cp': 'kJ/mol-K'
        }

        return advanced


class FreeHParser(ParseSection):
    name = "freeh"

    def __init__(self):
        super().__init__(r"f r e e   e n t h a l p y", r"freeh\s*:\s*all done")
        self.parsers = [FreeHData()]
