#!/usr/bin/env python

import numpy as np
import math as m

class TullyModel:
    # parameters in Tully's model
    A = 0.0
    B = 0.0
    C = 0.0
    D = 0.0

    # default to the values reported in Tully's 1990 JCP
    def __init__(self, a = 0.01, b = 1.6, c = 0.005, d = 1.0):
        self.A = a
        self.B = b
        self.C = c
        self.D = d

    def hamiltonian(self, R):
        v11 = m.copysign(self.A, R) * ( 1.0 - m.exp(-self.B * abs(R)) )
        v22 = -v11
        v12 = self.C * m.exp(-self.D * R * R)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

    def Vgrad(self, R):
        v11 = self.A * self.B * m.exp(-self.B * abs(R))
        v22 = -v11
        v12 = -2.0 * self.C * self.D * m.exp(-self.D * R * R)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

model = TullyModel()

# print surfaces for debugging
nr = np.linspace(-10, 10, 200 + 1)

for r in nr:
    ham = model.hamiltonian(r)
    diabat1 = ham[0,0]
    diabat2 = ham[1,1]
    adiabats = np.linalg.eigvalsh(ham)
    print "%12.6f %12.6f %12.6f %12.6f %12.6f" % (r, ham[0,0], ham[1,1], adiabats[0], adiabats[1])
