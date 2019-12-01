#!/usr/bin/env python
## @package trajectory
#  Module responsible for propagating surface hopping trajectories

# fssh: program to run surface hopping simulations for model problems
# Copyright (C) 2018, Shane Parker <smparker@uci.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import print_function, division

import numpy as np

from .fssh import StillInteracting, TraceManager

#####################################################################################
# Canned classes act as generator functions for initial conditions                  #
#####################################################################################

## Canned class whose call function acts as a generator for static initial conditions
class TrajGenConst(object):
    def __init__(self, position, momentum, initial_state):
        self.position = position
        self.momentum = momentum
        self.initial_state = initial_state

    ## generate nsamples initial conditions
    #  @param nsamples number of initial conditions requested
    def __call__(self, nsamples):
        for i in range(nsamples):
            yield (self.position, self.momentum, self.initial_state, {})

## Canned class whose call function acts as a generator for normally distributed initial conditions
class TrajGenNormal(object):
    ## Constructor
    # @param position center of normal distribution for position
    # @param momentum center of normal distribution for momentum
    # @param initial_state initial state designation
    # @param sigma standard deviation of distribution
    # @param seed initial seed to give to trajectory
    def __init__(self, position, momentum, initial_state, sigma, seed = None):
        self.position = position
        self.position_deviation = 0.5 * sigma
        self.momentum = momentum
        self.momentum_deviation = 1.0 / sigma
        self.initial_state = initial_state

        self.random_state = np.random.RandomState(seed)

    ## Whether to skip given momentum
    #  @param ktest momentum
    def kskip(self, ktest):
        return ktest < 0.0

    ## generate nsamples initial conditions
    #  @param nsamples number of initial conditions requested
    def __call__(self, nsamples):
        for i in range(nsamples):
            x = self.random_state.normal(self.position, self.position_deviation)
            k = self.random_state.normal(self.momentum, self.momentum_deviation)

            if (self.kskip(k)): continue
            yield (x, k, self.initial_state, {})

## Class to manage many TrajectorySH trajectories
#
# Requires a model object which is a class that has functions V(x), dV(x), nstates(), and ndim()
# that return the Hamiltonian at position x, gradient of the Hamiltonian at position x
# number of electronic states, and dimension of nuclear space, respectively.
class BatchedTraj(object):
    ## Constructor requires model and options input as kwargs
    # @param model object used to describe the model system
    # @param traj_gen generator object to generate initial conditions
    # @param trajectory_type surface hopping trajectory class
    # @param tracemanager object to collect results
    # @param inp input options
    #
    # Accepted keyword arguments and their defaults:
    # | key                |   default                  |
    # ---------------------|----------------------------|
    # | initial_time       | 0.0                        |
    # | samples            | 2000                       |
    # | dt                 | 20.0  ~ 0.5 fs             |
    # | seed               | None (date)                |
    # | nprocs             | MultiProcessing.cpu_count  |
    # | outcome_type       | "state"                    |
    def __init__(self, model, traj_gen, trajectory_type, tracemanager = TraceManager(), **inp):
        self.model = model
        self.tracemanager = tracemanager
        self.trajectory = trajectory_type
        self.traj_gen = traj_gen
        self.options = {}

        # time parameters
        self.options["initial_time"]  = inp.get("initial_time", 0.0)
        # statistical parameters
        self.options["samples"]       = inp.get("samples", 2000)
        self.options["dt"]            = inp.get("dt", 20.0) # default to roughly half a femtosecond

        # random seed
        self.options["seed"]          = inp.get("seed", None)

        self.options["nprocs"]        = inp.get("nprocs", 1)
        self.options["outcome_type"]  = inp.get("outcome_type", "state")

        # everything else just gets copied over
        for x in inp:
            if x not in self.options:
                self.options[x] = inp[x]

    ## runs a set of trajectories and collects the results
    # @param n number of trajectories to run
    def run_trajectories(self, n):
        outcomes = np.zeros([self.model.nstates(),2])
        traces = []
        try:
            for x0, p0, initial, params in self.traj_gen(n):
                traj_input = self.options
                traj_input.update(params)
                traj = self.trajectory(self.model, x0, p0, initial, self.tracemanager.spawn_tracer(), **traj_input)
                try:
                    trace = traj.simulate()
                    traces.append(trace)
                    outcomes += traj.outcome()
                except StillInteracting:
                    print("BEWARE: a simulation ended while still in interaction region", file=sys.stderr)
                    pass
            return (outcomes, traces)
        except KeyboardInterrupt:
            raise

    ## run many trajectories and returns averaged results
    def compute(self):
        # for now, define four possible outcomes of the simulation
        outcomes = np.zeros([self.model.nstates(),2])
        nsamples = int(self.options["samples"])
        energy_list = []
        nprocs = self.options["nprocs"]

        if nprocs > 1:
            pool = mp.Pool(nprocs)
            chunksize = min((nsamples - 1)//nprocs + 1, 5)
            nchunks = (nsamples -1)//chunksize + 1
            batches = [ min(chunksize, nsamples - chunksize*ip) for ip in range(nchunks) ]
            poolresult = [ pool.apply_async(unwrapped_run_trajectories, (self, b)) for b in batches ]
            try:
                for r in poolresult:
                    oc, tr = r.get()
                    outcomes += oc
                    self.tracemanager.add_batch(tr)
            except KeyboardInterrupt:
                return
                pool.terminate()
                pool.join()
                exit(" Aborting!")
            pool.close()
            pool.join()
        else:
            try:
                oc, tr = self.run_trajectories(nsamples)
                outcomes += oc
                self.tracemanager.add_batch(tr)
            except KeyboardInterrupt:
                exit(" Aborting!")

        outcomes /= np.sum(outcomes)
        self.tracemanager.outcomes = outcomes
        return self.tracemanager

## global version of BatchedTraj.run_trajectories that is necessary because of the stupid way threading pools work in python
def unwrapped_run_trajectories(fssh, n):
    try:
        return BatchedTraj.run_trajectories(fssh, n)
    except KeyboardInterrupt:
        pass
