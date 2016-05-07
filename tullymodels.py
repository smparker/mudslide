## @package tullymodels
#  Implementations of the one-dimensional two-state models Tully demonstrated FSSH on in Tully, J.C. <I>J. Chem. Phys.</I> 1990 <B>93</B> 1061.

import numpy as np
import math as m

# Here are some helper functions that pad the model problems with fake electronic states.
# Useful for debugging, so keeping it around
'''
def pad_model(nstates, diags):
    def V_decorator(func):
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            oldnstates = out.shape[0]
            out = np.pad(out, (0,nstates), 'constant')
            if nstates > 1:
                for i in range(nstates):
                    out[oldnstates+i,oldnstates+i] = diags[i]
            else:
                out[-1,-1] = diags
            return out
        return wrapper

    def dV_decorator(func):
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            nout = np.zeros([out.shape[0], out.shape[1]+nstates, out.shape[2]+nstates])
            nout[:,0:out.shape[1],0:out.shape[2]] += out[:,:,:]
            return nout
        return wrapper

    def nstates_decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs) + nstates
        return wrapper

    def class_decorator(cls):
        class padded_model(cls):
            def __init__(self, *args, **kwargs):
                cls.__init__(self, *args, **kwargs)

                self.V = V_decorator(self.V)
                self.dV = dV_decorator(self.dV)
                self.nstates = nstates_decorator(self.nstates)

        return padded_model
    return class_decorator
'''

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
class TullySimpleAvoidedCrossing(object):
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
        return out.reshape([1, 2, 2])

    def nstates(self):
        return 2

    def ndim(self):
        return 1

## Tunneling through a double avoided crossing used in Tully's 1990 JCP
#
# \f[ V_{11} = 0 \f]
# \f[ V_{22} = -A e^{-Bx^2} + E_0 \f]
# \f[ V_{12} = V_{21} = C e^{-D x^2} \f]
class TullyDualAvoidedCrossing(object):
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
        return out.reshape([1, 2, 2])

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
class TullyExtendedCouplingReflection(object):
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
        return out.reshape([1, 2, 2])

    def nstates(self):
        return 2

    def ndim(self):
        return 1

class SuperExchange(object):
    ## Constructor defaults to Prezhdo paper on GFSH
    def __init__(self, v11 = 0.0, v22 = 0.01, v33 = 0.005, v12 = 0.001, v23 = 0.01):
        self.v11 = v11
        self.v22 = v22
        self.v33 = v33
        self.v12 = v12
        self.v23 = v23

    ## \f$V(x)\f$
    def V(self, x):
        v12 = self.v12 * m.exp(-0.5*x*x)
        v23 = self.v23 * m.exp(-0.5*x*x)

        return np.array([ [self.v11, v12, 0.0],
                          [v12, self.v22, v23],
                          [0.0, v23, self.v33] ])

    ## \f$ \nabla V(x)\f$
    def dV(self, x):
        v12 = -x * self.v12 * m.exp(-0.5*x*x)
        v23 = -x * self.v23 * m.exp(-0.5*x*x)
        out = np.array([ [0.0, v12, 0.0],
                         [v12, 0.0, v23],
                         [0.0, v23, 0.0] ])

        return out.reshape([1, 3, 3])

    def nstates(self):
        return 3

    def ndim(self):
        return 1

modeldict = { "simple" : TullySimpleAvoidedCrossing,
              "dual"   : TullyDualAvoidedCrossing,
              "extended" : TullyExtendedCouplingReflection,
              "super"  : SuperExchange }
