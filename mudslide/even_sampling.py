# -*- coding: utf-8 -*-
"""Propagate even sampling algorithm"""

from __future__ import print_function, division

import copy as cp

import numpy as np

from .cumulative_sh import TrajectoryCum
from .integration import quadrature

from typing import Optional, List, Any, Dict, Union
from .typing import ArrayLike, ElectronicT

class SpawnStack(object):
    """Data structure to inform how new traces are spawned and weighted"""
    def __init__(self, sample_stack: List, weight: float = 1.0):
        self.sample_stack = sample_stack
        self.base_weight: float = weight
        self.marginal_weight: float = 1.0
        self.last_stack: Dict = {}
        self.last_dw: float

        if sample_stack:
            weights = np.array( [ s["dw"] for s in sample_stack ] )
            mw = np.ones(len(sample_stack))
            mw[1:] -= np.cumsum(weights[:len(weights)-1])
            self.marginal_weights = mw
            self.last_dw = weights[0]
        else:
            self.margin_weights = np.ones(1)
            self.last_dw = 0.0

        self.izeta: int = 0
        self.zeta_: float = -1.0

    def zeta(self) -> float:
        return self.zeta_

    def next_zeta(self, current_value: float, random_state: Any = np.random.RandomState()) -> float:
        if self.sample_stack:
            izeta = self.izeta
            while izeta < len(self.sample_stack) and self.sample_stack[izeta]["zeta"] < current_value:
                izeta += 1

            if izeta != self.izeta: # it means there was a hop, so update last_dw
                weights = np.array( [ s["dw"] for s in self.sample_stack ] )
                self.last_dw = np.sum(weights[self.izeta:izeta])
                self.last_stack = self.sample_stack[self.izeta] # should actually probably be a merge operation

            self.marginal_weight = self.marginal_weights[izeta] if izeta != len(self.sample_stack) else 0.0
            self.izeta = izeta

            if self.izeta < len(self.sample_stack):
                self.zeta_ = self.sample_stack[self.izeta]["zeta"]
            else:
                self.zeta_ = 10.0 # should be impossible

        else:
            self.zeta_ = random_state.uniform()

        return self.zeta_

    def weight(self) -> float:
        return self.base_weight * self.marginal_weight

    def spawn(self, reweight: float = 1.0) -> 'SpawnStack':
        if self.sample_stack:
            samp = self.last_stack
            dw = self.last_dw
            if dw == 0:
                raise Exception("What happened? A hop with no differential weight?")
            weight = self.base_weight * dw * reweight
            next_stack = samp["children"]
        else:
            weight = self.base_weight * reweight
            next_stack = None
        return self.__class__(next_stack, weight)

    def do_spawn(self) -> bool:
        """Test whether the stack indicates we should keep spawning trajectories
        versus just following one. An empty stack means we should behave like
        a normal cumulative surface hopping run.
        """
        return bool(self.sample_stack)

    def spawn_size(self) -> int:
        if self.sample_stack:
            samp = self.last_stack
            if "spawn_size" in samp:
                return samp["spawn_size"]
            else:
                return 1
        else:
            return 1

    @classmethod
    def from_quadrature(cls, nsamples: Union[List[int], int], weight: float = 1.0, method: str = "gl",
            mcsamples: int = 1, random_state: Any = np.random.RandomState()) -> 'SpawnStack':
        if not isinstance(nsamples, list):
            nsamples = [ int(nsamples) ]

        forest: List[Dict] = []
        spawn_size = [ mcsamples ] + [ 1 ] * (len(nsamples) - 1)

        for ns in reversed(nsamples):
            leaves = cp.copy(forest)
            samples, weights = quadrature(ns, 0.0, 1.0, method=method)
            spawnsize = spawn_size.pop(0)
            forest = [ { "zeta" : s,
                "dw" : dw,
                "children" : cp.deepcopy(leaves),
                "spawn_size" : spawnsize }
                    for s, dw in zip(samples, weights) ] # type: ignore

        return cls(forest, weight)

class EvenSamplingTrajectory(TrajectoryCum):
    """Trajectory surface hopping using an even sampling approach

    Related to the cumulative trajectory picture, but instead of hopping
    stochastically, new trajectories are spawned at even intervals of the
    of the cumulative probability distribution. This is an *experimental*
    in principle deterministic algorithm for FSSH simulations.
    """
    def __init__(self, *args: Any, **options: Any):
        """Constructor (see TrajectoryCum constructor)"""
        TrajectoryCum.__init__(self, *args, **options)

        ss = options["spawn_stack"]
        if isinstance(ss, SpawnStack):
            self.spawn_stack = cp.deepcopy(options["spawn_stack"])
        elif isinstance(ss, list):
            quadrature = options.get("quadrature", "gl")
            mcsamples = options.get("mcsamples", 1)
            self.spawn_stack = SpawnStack.from_quadrature(ss, method=quadrature,
                    mcsamples=mcsamples, random_state=self.random_state)
        else:
            self.spawn_stack = SpawnStack(ss)

        self.zeta = self.spawn_stack.next_zeta(0.0, self.random_state)

    def clone(self, spawn_stack: Optional[Any] = None) -> 'EvenSamplingTrajectory':
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
                t0 = self.time+self.dt, # will restart at the next step
                previous_steps = self.nsteps+1, # will restart at the next step
                trace_every = self.trace_every,
                dt = self.dt,
                outcome_type = self.outcome_type,
                seed_sequence = self.seed_sequence.spawn(1)[0],
                electronics = self.electronics,
                restart = True,
                duration = self.duration,
                spawn_stack = spawn_stack)
        return out

    def hopper(self, probs: ArrayLike) -> List[Dict[str, Union[int, float]]]:
        """Given a set of probabilities, determines whether and where to hop
        :param probs: [nstates] numpy array of individual hopping probabilities
        :returns: [ (target_state, hop_weight) ]
        """
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
                nspawn = self.spawn_stack.spawn_size()
                spawn_weight = 1.0 / nspawn
                targets = [ { "target" : i,
                              "weight" : hop_choice[i],
                              "zeta" : zeta,
                              "prob" : accumulated,
                              "stack" : self.spawn_stack.spawn(spawn_weight * hop_choice[i])}
                                    for i in range(self.model.nstates()) if i != self.state for j in range(nspawn) ]
            else:
                target = self.random_state.choice(list(range(self.model.nstates())), p=hop_choice)
                targets = [ {"target" : target, "weight" : 1.0, "zeta" : zeta, "prob" : accumulated, "stack" : self.spawn_stack.spawn() }]

            # reset probabilities and random
            self.zeta = next_zeta

            self.prob_cum = accumulated
            return targets

        self.prob_cum = accumulated
        return []

    def hop_to_it(self, hop_to: List[Dict[str, Any]], electronics: ElectronicT = None) -> None:
        """hop_to_it for even sampling spawns new trajectories instead of enacting hop

         hop_to_it must accomplish:
           - copy current trajectory
           - initiate hops on the copied trajectories
           - make no changes to current trajectory
           - set next threshold for spawning

        :param hop_to: [nspawn] list of states and associated weights on which to hop
        :param electronics: model class
        """
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
