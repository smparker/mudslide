#!/usr/bin/env python

import numpy as np

import mudslide

mudslide.print_header()

simple_model = mudslide.models.TullySimpleAvoidedCrossing()

# Generates trajectories always with starting position -5, starting momentum 10.0, on ground state
x0 = np.array([-5.0])
p0 = np.array([15.0])
v0 = p0 / simple_model.mass

traj = mudslide.SurfaceHoppingMD(simple_model, x0, v0, 0, dt=1, max_steps=1000,
                                 hopping_method='cumulative', trace_every=1,
                                 seed_sequence=7943)
log = traj.simulate()

print("#" + " ".join([f"{x:20s}" for x in ["time", "hopping", "prob_cum", "zeta"]]))
for snap in log:
    t = snap["time"]
    hop = snap["hopping"]
    prob_cum = snap["prob_cum"]
    zeta = snap["zeta"]
    print(f"#{t:20.12f} {hop:20.12f} {prob_cum:20.12f} {zeta:20.12f}")
