#!/usr/bin/env python

from __future__ import print_function

import re

from .section_parser import ParseSection


class FreeHData(ParseSection):
    name = ''

    start1st_re = re.compile(r"T\s+p\s+ln\(qtrans\)\s+ln\(qrot\)\s+ln\(qvib\)\s+chem\.pot\.\s+energy\s+entropy")
    data1st_re = re.compile(r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")
    start2nd_re = re.compile(r"\(K\)\s+\(MPa\)\s+\(kJ/mol-K\)\s+\(kJ/mol-K\)\s+\(kJ/mol\)")
    data2nd_re = re.compile(r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")
    end_re = re.compile(r"\*{50}")

    def __init__(self):
        ParseSection.__init__(self,
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
            found = False
            m = self.start1st_re.search(liter.top())

            if m:
                next(liter)
                advanced = True
                done1st = False
                while not done1st:
                    m1st = self.data1st_re.search(liter.top())
                    if m1st:
                        data.append({
                            'T': float(m1st.group(1)),
                            'P': float(m1st.group(2)),
                            'qtrans': float(m1st.group(3)),
                            'qrot': float(m1st.group(4)),
                            'qvib': float(m1st.group(5)),
                            'chem.pot.': float(m1st.group(6)),
                            'energy': float(m1st.group(7)),
                            'entropy': float(m1st.group(8))
                        })

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
                        data[i].update({
                            'T': float(m2nd.group(1)),
                            'P': float(m2nd.group(2)),
                            'Cv': float(m2nd.group(3)),
                            'Cp': float(m2nd.group(4)),
                            'H': float(m2nd.group(5))
                        })
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
            'Cv': 'kJ/mol-K'
        }

        return advanced


class FreeHParser(ParseSection):
    name = "freeh"

    freehdata = FreeHData()

    def __init__(self):
        ParseSection.__init__(self, r"f r e e   e n t h a l p y", r"freeh\s*:\s*all done")

    parsers = [freehdata]
