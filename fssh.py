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

    def V(self, R):
        v11 = m.copysign(self.A, R) * ( 1.0 - m.exp(-self.B * abs(R)) )
        v22 = -v11
        v12 = self.C * m.exp(-self.D * R * R)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

    def Vgrad(self, R):
        v11 = self.A * self.B * m.exp(-self.B * abs(R))
        v22 = -v11
        v12 = -2.0 * self.C * self.D * R * m.exp(-self.D * R * R)
        out = np.array([ [v11, v12],
                         [v12, v22] ])
        return out

#class FSSH:
#    

class Trajectory:
    previous_position = 0.0
    position = 0.0
    velocity = 0.0
    time = 0.0
    mass = 1.0
    rho = np.zeros([2,2])
    step = 0
    state = 0
    dt = 0.1

    def __init__(self, p, v, m, rho, options):
        # input explicitly
        self.position = p
        self.velocity = v
        self.mass = m
        self.rho = rho

        # fixed initial parameters
        self.time = 0.0
        self.step

        # read out of options
        self.dt = options["dt"]

    def nuclear_step(self, phi, Vgrad):
        force = - (np.dot(phi.T, np.dot(Vgrad, phi)))
        x = self.position
        xlast = self.previous_position
        xnew = 0.0
        if self.step == 0:
            xnew = x + self.velocity * self.dt + 0.5 * force * self.dt * self.dt
        else:
            xnew = 2.0 * x - xlast + force * self.dt * self.dt
        self.step += 1
        self.position, self.previous_position = xnew, self.position

    def derivative_coupling(self, phi_i, E_i, phi_j, E_j, Vgrad):
        matel = np.dot(phi_i.T, np.dot(Vgrad, phi_j))
        return matel / (E_i - E_j)

model = TullyModel()

# compute barrier

diss_ham = model.V(-10000)
diss_energy = np.linalg.eigvalsh(diss_ham)[0]
int_ham = model.V(0)
int_energy = np.linalg.eigvalsh(int_ham)[0]

# print surfaces for debugging
nr = np.linspace(-10, 10, 200 + 1)

traj = Trajectory(-13, 0.10000, 1, np.zeros([2,2]), {"dt" : 0.1})
nsteps = 5000
for i in range(nsteps):
    ham = model.V(traj.position)
    dham = model.Vgrad(traj.position)
    energies, adiabats = np.linalg.eigh(ham)
    traj.nuclear_step(adiabats[:,0], dham)
    print "%6d %12.6f" % (i, traj.position)

#
#for r in nr:
#    ham = model.V(r)
#    dham = model.Vgrad(r)
#    energies, adiabats = np.linalg.eigh(ham)
#    force = traj.nuclear_step(adiabats[:,0], dham)
#    d = traj.derivative_coupling(adiabats[:,0], energies[0], adiabats[:,1], energies[1], dham)
#    #print "%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (r, ham[0,0], ham[1,1], energies[0], energies[1], gradient, d)
#    print "%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (r, ham[0,0], energies[0], dham[0,0], force, d)
