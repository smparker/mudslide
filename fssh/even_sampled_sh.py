#!/usr/bin/env python
## @package fssh
#  Module responsible for propagating surface hopping trajectories

# fssh: program to run surface hopping simulations for model problems
# Copyright (C) 2018-2020, Shane Parker <shane.parker@case.edu>
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

import copy as cp

import numpy as np

from .fssh import TrajectoryCum

## Data structure to inform how new traces are spawned and weighted
class SpawnStack(object):
    def __init__(self, sample_stack, weight):
        self.sample_stack = sample_stack
        self.base_weight = weight
        self.marginal_weight = 1.0

        if sample_stack is not None:
            mw = np.zeros(len(sample_stack))
            mw[0] = 1.0
            for i in range(1, len(sample_stack)):
                mw[i] = 1.0 - sample_stack[i-1]["zeta"]
            self.marginal_weights = mw
        else:
            self.margin_weights = np.ones(1)

        self.izeta = None
        self.zeta = None
        self.last_dw = 0.0 # differential weight of last hop TODO I don't love this structure

    def zeta(self):
        return self.zeta

    def next_zeta(self, current_value, random_state = np.random.RandomState()):
        if self.sample_stack is not None:
            if self.izeta is None:
                self.izeta = 0
                self.marginal_weight = 1.0
            else:
                izeta = self.izeta
                while self.sample_stack[izeta]["zeta"] < current_value:
                    izeta += 1

                self.marginal_weight = self.marginal_weights[izeta]
                self.last_dw = self.marginal_weights[self.izeta] - self.marginal_weights[izeta]

                self.izeta = izeta

            if self.izeta < len(self.sample_stack):
                self.zeta = self.sample_stack[self.izeta]["zeta"]
            else:
                raise Exception("Should I be returning None?")
                self.zeta = None
        else:
            self.zeta = random_state.uniform()

        return self.zeta

    def weight(self):
        return self.base_weight * self.marginal_weight

    def spawn(self, reweight = 1.0):
        if self.sample_stack:
            samp = self.sample_stack[self.izeta]
            dw = self.last_dw
            if dw == 0:
                raise Exception("What happened? A hop with no differential weight?")
            weight = self.base_weight * dw * reweight
            next_stack = samp["children"]
        else:
            weight = self.base_weight * reweight
            next_stack = None
        return self.__class__(next_stack, weight)

    ## Test whether the stack indicates we should keep spawning trajectories
    #  versus just following one. An empty stack means we should behave like
    #  a normal cumulative surface hopping run.
    def do_spawn(self):
        return self.sample_stack is not None

    @classmethod
    def build_simple(cls, nsamples, sample_depth, include_first=False):
        samples = np.sort(np.linspace(1.0, 0.0, nsamples, endpoint=include_first, retstep=False))
        dw = samples[1] - samples[0]

        forest = [ { "zeta" : s, "dw" : dw, "children" : None } for s in samples ]

        for d in range(1, sample_depth):
            leaves = cp.copy(forest)
            forest = [ { "zeta" : s, "dw" : dw, "children" : cp.deepcopy(leaves) } for s in samples ]

        return cls(forest, 1.0)


## Trajectory surface hopping using an even sampling approach
#
#  Related to the cumulative trajectory picture, but instead of hopping
#  stochastically, new trajectories are spawned at even intervals of the
#  of the cumulative probability distribution. This is an *experimental*
#  in principle deterministic algorithm for FSSH simulations.
class EvenSamplingTrajectory(TrajectoryCum):
    ## Constructor (see TrajectoryCum constructor)
    def __init__(self, *args, **options):
        TrajectoryCum.__init__(self, *args, **options)

        self.spawn_stack = cp.deepcopy(options["spawn_stack"])

        self.zeta = self.spawn_stack.next_zeta(self.random_state)

    def clone(self, spawn_stack=None):
        if spawn_stack is None:
            spawn_stack = self.spawn_stack

        out = EvenSamplingTrajectory(
                self.model,
                self.position,
                self.velocity * self.mass,
                self.rho,
                tracer=cp.deepcopy(self.tracer),
                queue = self.queue,
                last_velocity = self.last_velocity,
                state0 = self.state,
                t0 = self.time,
                previous_steps = self.nsteps,
                trace_every = self.trace_every,
                dt = self.dt,
                outcome_type = self.outcome_type,
                seed = None,
                electronics = self.electronics,
                duration = self.duration,
                spawn_stack = spawn_stack)
        return out

    ## given a set of probabilities, determines whether and where to hop
    # @param probs [nstates] numpy array of individual hopping probabilities
    #  returns [ (target_state, hop_weight) ]
    def hopper(self, probs):
        accumulated = self.prob_cum
        probs[self.state] = 0.0 # ensure self-hopping is nonsense
        gkdt = np.sum(probs)

        accumulated = 1 - (1 - accumulated) * np.exp(-gkdt)
        if accumulated > self.zeta: # then hop
            zeta = self.zeta
            next_zeta = self.spawn_stack.next_zeta(accumulated, self.random_state)

            # where to hop
            hop_choice = probs / gkdt
            if self.spawn_stack.do_spawn():
                targets = [ { "target" : i,
                              "weight" : hop_choice[i],
                              "zeta" : zeta,
                              "prob" : accumulated,
                              "stack" : self.spawn_stack.spawn(hop_choice[i])} for i in range(self.model.nstates()) if i != self.state ]
            else:
                target = self.random_state.choice(list(range(self.model.nstates())), p=hop_choice)
                targets = [ {"target" : target, "weight" : 1.0, "zeta" : zeta, "prob" : accumulated, "stack" : self.spawn_stack.spawn() }]

            # reset probabilities and random
            self.zeta = next_zeta

            self.prob_cum = accumulated
            return targets

        self.prob_cum = accumulated
        return []

    ## hop_to_it for even sampling spawns new trajectories instead of enacting hop
    #
    #  hop_to_it must accomplish:
    #    - copy current trajectory
    #    - initiate hops on the copied trajectories
    #    - make no changes to current trajectory
    #    - set next threshold for spawning
    #
    # @param hop_to [nspawn] list of states and associated weights on which
    # @param electronics model class
    def hop_to_it(self, hop_to, electronics=None):
        if self.spawn_stack.do_spawn():
            for hop in hop_to:
                stack = hop["stack"]
                spawn = self.clone(stack)

                # trigger hop
                TrajectoryCum.hop_to_it(spawn, [ hop ], electronics=spawn.electronics)
                spawn.update_weight(stack.weight())

                # add to total trajectory queue
                self.queue.put(spawn)
            self.update_weight(self.spawn_stack.weight())
        else:
            self.prob_cum = 0.0
            TrajectoryCum.hop_to_it(self, hop_to, electronics=self.electronics)
