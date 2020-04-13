#!/usr/bin/env python

import fssh

simple_model = fssh.models.TullySimpleAvoidedCrossing()

# Generates trajectories always with starting position -5, starting momentum 10.0, on ground state
traj_gen = fssh.TrajGenConst(-5.0, 10.0, "ground")

simulator = fssh.BatchedTraj(simple_model, traj_gen, fssh.TrajectorySH, samples = 4)
results = simulator.compute()
outcomes = results.outcomes

print("Probability of reflection on the ground state:    %12.4f" % outcomes[0,0])
print("Probability of transmission on the ground state:  %12.4f" % outcomes[0,1])
print("Probability of reflection on the excited state:   %12.4f" % outcomes[1,0])
print("Probability of transmission on the excited state: %12.4f" % outcomes[1,1])
