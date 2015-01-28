## @package tullymodels
#  Implementations of the one-dimensional two-state models Tully demonstrated FSSH on in Tully, J.C. <I>J. Chem. Phys.</I> 1990 <B>93</B> 1061.

import numpy as np
import math as m

## Tunneling through a single barrier model used in Tully's 1990 JCP
#
# \f[
#     V_{11} = \left\{ \begin{array}{cr}
#                   A (1 - e^{Bx}) & x < 0 \\
#                  -A (1 - e^{-Bx}) & x > 0
#                   \end{array} \right.
# \f]
# \f[ V_{22} = -V_{11} \f]
# \f[ V_{12} = V_{21} = C e^{-D x^2} \f]
class TullySimpleAvoidedCrossing:
    ## Constructor that defaults to the values reported in Tully's 1990 JCP
    def __init__(self, a = 0.01, b = 1.6, c = 0.005, d = 1.0):
        self.A = a
        self.B = b
        self.C = c
        self.D = d

    ## \f$V(x)\f$
    def V(self, x):
        v11 = m.copysign(self.A, x) * ( 1.0 - m.exp(-self.B * abs(x)) )
        v22 = -v11
        v12 = self.C * m.exp(-self.D * x * x)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

    ## \f$\nabla V(x)\f$
    def dV(self, x):
        v11 = self.A * self.B * m.exp(-self.B * abs(x))
        v22 = -v11
        v12 = -2.0 * self.C * self.D * x * m.exp(-self.D * x * x)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out.reshape([2, 2, 1])

    def nstates(self):
        return 2

    def ndim(self):
        return 1

## Tunneling through a double avoided crossing used in Tully's 1990 JCP
#
# \f[ V_{11} = 0 \f]
# \f[ V_{22} = -A e^{-Bx^2} + E_0 \f]
# \f[ V_{12} = V_{21} = C e^{-D x^2} \f]
class TullyDualAvoidedCrossing:
    ## Constructor that defaults to the values reported in Tully's 1990 JCP
    def __init__(self, a = 0.1, b = 0.28, c = 0.015, d = 0.06, e = 0.05):
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.E0 = e

    ## \f$V(x)\f$
    def V(self, x):
        v11 = 0.0
        v22 = - self.A * m.exp(-self.B * x * x) + self.E0
        v12 = self.C * m.exp(-self.D * x * x)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

    ## \f$\nabla V(x)\f$
    def dV(self, x):
        v11 = 0.0
        v22 = 2.0 * self.A * self.B * x * m.exp(-self.B * x * x)
        v12 = -2.0 * self.C * self.D * x * m.exp(-self.D * x * x)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out.reshape([2, 2, 1])

    def nstates(self):
        return 2

    def ndim(self):
        return 1

## Model with extended coupling and the possibility of reflection. The most challenging of the
#  models used in Tully's 1990 JCP
# \f[ V_{11} = A \f]
# \f[ V_{22} = -A \f]
# \f[
#     V_{12} = \left\{ \begin{array}{cr}
#                   B e^{Cx} & x < 0 \\
#                   B \left( 2 - e^{-Cx} \right) & x > 0
#                   \end{array} \right.
# \f]
class TullyExtendedCouplingReflection:
    ## Constructor that defaults to the values reported in Tully's 1990 JCP
    def __init__(self, a = 0.0006, b = 0.10, c = 0.90):
        self.A = a
        self.B = b
        self.C = c

    ## \f$V(x)\f$
    def V(self, x):
        v11 = self.A
        v22 = -self.A
        v12 = m.exp(-abs(x)*self.C)
        if x < 0:
            v12 = self.B * v12
        else:
            v12 = self.B * (2.0 - v12)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

    ## \f$\nabla V(x)\f$
    def dV(self, x):
        v11 = 0.0
        v22 = 0.0
        v12 = self.B * self.C * m.exp(-self.C * abs(x))
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out.reshape([2, 2, 1])

    def nstates(self):
        return 2

    def ndim(self):
        return 1
