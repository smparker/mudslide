# -*- coding: utf-8 -*-
"""Physical constants"""

# mudslide always uses atomic units. These are convenience definitions
# that let you write things like dt=0.5*fs without having to care what the base units are.

# Energy units
Hartree = 1.0
eV = 1.0 / 27.211386245988
kJmol = 1.0 / 2625.499639
kcalmol = 1.0 / 627.5094740631

# mass
amu = 1822.8884862
electron_mass = 1.0

# distance
bohr = 1.0
angstrom = 1.0 / 0.52917721092
nm = 10.0 * angstrom

# From NIST CODATA 2024 (https://physics.nist.gov/cgi-bin/cuu/Value?aut)
autime = 1.0
fs = 1e-15 / 2.4188843265857e-17
ps = 1e-12 / 2.4188843265857e-17
