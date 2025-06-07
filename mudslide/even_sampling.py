# -*- coding: utf-8 -*-
"""Propagate even sampling algorithm.

This module implements trajectory surface hopping with even sampling of phase space.
"""

from itertools import count
from typing import Optional, List, Any, Dict, Union
import copy as cp
import numpy as np
from numpy.typing import ArrayLike

from .integration import quadrature
from .surface_hopping_md import SurfaceHoppingMD
from .typing import ElectronicT

class SpawnStack:
    """Data structure to inform how new traces are spawned and weighted.

    Parameters
    ----------
    sample_stack : list
        Stack of samples containing zeta values and differential weights.
    weight : float, optional
        Base weight for the stack, by default 1.0
    """

    def __init__(self, sample_stack: List, weight: float = 1.0):
        self.sample_stack = sample_stack
        self.base_weight: float = weight
        self.marginal_weight: float = 1.0
        self.last_stack: Dict = {}
        self.last_dw: float

        if sample_stack:
            weights = np.array([s["dw"] for s in sample_stack])
            mw = np.ones(len(sample_stack))
            mw[1:] -= np.cumsum(weights[: len(weights) - 1])
            self.marginal_weights = mw
            self.last_dw = weights[0]
        else:
            self.margin_weights = np.ones(1)
            self.last_dw = 0.0

        self.izeta: int = 0
        self.zeta_: float = -1.0

    def zeta(self) -> float:
        """Get the current zeta value.

        Returns
        -------
        float
            Current zeta value.
        """
        return self.zeta_

    def next_zeta(self, current_value: float, random_state: Any = np.random.RandomState()) -> float:
        """Calculate the next zeta value.

        Parameters
        ----------
        current_value : float
            Current accumulated probability value.
        random_state : Any, optional
            Random state for generating random numbers, by default np.random.RandomState()

        Returns
        -------
        float
            Next zeta value for determining hops.
        """
        if self.sample_stack:
            izeta = self.izeta
            while izeta < len(self.sample_stack) and self.sample_stack[izeta]["zeta"] < current_value:
                izeta += 1

            if izeta != self.izeta:  # it means there was a hop, so update last_dw

                weights = np.array([s["dw"] for s in self.sample_stack])
                self.last_dw = np.sum(weights[self.izeta : izeta])
                self.last_stack = self.sample_stack[self.izeta]  # should actually probably be a merge operation

            self.marginal_weight = self.marginal_weights[izeta] if izeta != len(self.sample_stack) else 0.0
            self.izeta = izeta

            if self.izeta < len(self.sample_stack):
                self.zeta_ = self.sample_stack[self.izeta]["zeta"]
            else:
                self.zeta_ = 10.0  # should be impossible

        else:
            self.zeta_ = random_state.uniform()

        return self.zeta_

    def weight(self) -> float:
        """Calculate the current weight.

        Returns
        -------
        float
            Current weight as product of base weight and marginal weight.
        """
        return self.base_weight * self.marginal_weight

    def spawn(self, reweight: float = 1.0) -> "SpawnStack":
        """Create a new SpawnStack for spawning trajectories.

        Parameters
        ----------
        reweight : float, optional
            Weight adjustment factor, by default 1.0

        Returns
        -------
        SpawnStack
            New SpawnStack instance for spawned trajectory.

        Raises
        ------
        Exception
            If a hop occurs with no differential weight.
        """
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
        """Test whether the stack indicates we should keep spawning trajectories.

        Returns
        -------
        bool
            True if stack indicates spawning should continue, False otherwise.
        """
        return bool(self.sample_stack)

    def spawn_size(self) -> int:
        """Get the number of trajectories to spawn.

        Returns
        -------
        int
            Number of trajectories to spawn.
        """
        if self.sample_stack:
            samp = self.last_stack
            if "spawn_size" in samp:
                return samp["spawn_size"]
        return 1

    def append_layer(self, zetas: list, dws: list, stack=None, node=None, nodes=None, adj_matrix=None):
        """Append a layer to all leaves in the sample stack tree.

        A depth-first traversal of a sample_stack tree to append a layer to all leaves.
        The method builds the adjacency matrix as it traverses the tree.

        Parameters
        ----------
        zetas : list
            List of zeta values to append.
        dws : list
            List of differential weights to append.
        stack : list, optional
            Current stack being processed, by default None
        node : int, optional
            Current node being processed, by default None
        nodes : iterator, optional
            Iterator for generating node numbers, by default None
        adj_matrix : dict, optional
            Adjacency matrix being built, by default None

        Raises
        ------
        ValueError
            If dimensions of dws and zetas do not match.
        """

        if stack is None:
            stack = self.sample_stack

        if len(zetas) != len(dws):
            raise ValueError("dimension of dws should be same as zetas")

        l = len(stack)

        # First time through
        if adj_matrix is None:
            adj_matrix = {}
            nodes = count(start=1, step=1)
            node = next(nodes)
            adj_matrix[1] = []
            for i in range(l):
                adj_matrix[1].append(next(nodes))
        else:
            adj_matrix[node] = []
            for i in range(l):
                adj_matrix[node].append(next(nodes))
        if l == 0:
            for i in range(len(zetas)):
                stack.append({"zeta": zetas[i], "dw": dws[i], "children": [], "spawn_size": 1})
        else:
            for i in range(l):
                self.append_layer(
                    zetas=zetas,
                    dws=dws,
                    stack=stack[i]["children"],
                    node=adj_matrix[node][i],
                    nodes=nodes,
                    adj_matrix=adj_matrix,
                )

    def unpack(self, zeta_list, dw_list, stack=None, depth=0):
        """Recursively unpack a sample stack.

        Fills in the zeta_list and dw_list with flattened lists of zeta values
        and differential weights.

        Parameters
        ----------
        zeta_list : list
            List to be filled with zeta values.
        dw_list : list
            List to be filled with differential weights.
        stack : list or dict, optional
            Current stack being processed, by default None
        depth : int, optional
            Current depth in the tree, by default 0
        """

        if stack is None:
            stack = self.sample_stack

        if isinstance(stack, list):
            for dct in stack:
                self.unpack(zeta_list, dw_list, dct, depth=depth)
        if isinstance(stack, dict):
            zeta_list.append((depth, stack["zeta"]))
            dw_list.append((depth, stack["dw"]))
            if stack["children"] != []:
                self.unpack(zeta_list, dw_list, stack["children"], depth=depth + 1)


    def unravel(self):
        """Unravel the sample stack into points and weights.

        Calls unpack to recursively unpack a sample_stack and then unravels the list
        of zetas and dws to create a list of tuples of points and weights.

        Returns
        -------
        list
            List of tuples containing (points, weights) where points is a tuple of
            coordinates and weights is the product of weights at those coordinates.
        """
        zetas = []
        dws = []
        self.unpack(zeta_list=zetas, dw_list=dws)

        dim_list = []
        for tpl in zetas:
            dim_list.append(tpl[0])
        dim = max(dim_list) + 1
        main_list = []
        coords = []
        weights = []
        last_depth = 0
        for i, tpl in enumerate(zetas):
            # When we recurse back up in depth, we need to remove num_to_pop items from coords/weights.
            if tpl[0] < last_depth:
                num_to_pop = last_depth - tpl[0] + 1
                for j in range(num_to_pop):
                    coords.pop()
                    weights.pop()
            # Take care of the fact that the above doesn't work for 1D.
            if (dim == 1) and (len(coords) != 0):
                coords.pop()
                weights.pop()
            if tpl[0] == 0:
                last_depth = tpl[0]
                coords.append(tpl[1])
                weights.append(dws[i][1])
            else:
                last_depth = tpl[0]
                if len(coords) > tpl[0]:
                    coords[tpl[0]] = tpl[1]
                    weights[tpl[0]] = dws[i][1]
                else:
                    coords.append(tpl[1])
                    weights.append(dws[i][1])
            if len(coords) == dim:
                main_list.append((tuple(coords), tuple(weights)))

        # Return product of weights
        points_weights = [(points, np.prod(weights)) for (points, weights) in main_list]

        return points_weights

    @classmethod
    def from_quadrature(
        cls,
        nsamples: Union[List[int], int],
        weight: float = 1.0,
        method: str = "gl",
        mcsamples: int = 1,
        random_state: Any = np.random.RandomState(),
    ) -> "SpawnStack":
        """Create a SpawnStack from quadrature points.

        Parameters
        ----------
        nsamples : Union[List[int], int]
            Number of samples per dimension or list of samples per dimension.
        weight : float, optional
            Base weight for the stack, by default 1.0
        method : str, optional
            Quadrature method to use, by default "gl"
        mcsamples : int, optional
            Number of Monte Carlo samples per quadrature point, by default 1
        random_state : Any, optional
            Random state for generating random numbers, by default np.random.RandomState()

        Returns
        -------
        SpawnStack
            New SpawnStack instance created from quadrature points.
        """
        if not isinstance(nsamples, list):
            nsamples = [int(nsamples)]

        forest: List[Dict] = []
        spawn_size = [mcsamples] + [1] * (len(nsamples) - 1)

        for ns in reversed(nsamples):
            leaves = cp.copy(forest)
            samples, weights = quadrature(ns, 0.0, 1.0, method=method)
            spawnsize = spawn_size.pop(0)
            forest = [
                {"zeta": s, "dw": dw, "children": cp.deepcopy(leaves), "spawn_size": spawnsize}
                for s, dw in zip(samples, weights)
            ]  # type: ignore

        return cls(forest, weight)


class EvenSamplingTrajectory(SurfaceHoppingMD):
    """Trajectory surface hopping with even sampling of phase space.

    This class extends SurfaceHoppingMD to implement even sampling of phase space
    during trajectory surface hopping simulations.

    Parameters
    ----------
    *args : tuple
        Arguments passed to SurfaceHoppingMD constructor
    **options : dict
        Options passed to SurfaceHoppingMD constructor
    spawn_stack : SpawnStack, optional
        Stack for managing trajectory spawning
    quadrature : dict, optional
        Quadrature parameters for sampling
    mcsamples : int, optional
        Number of Monte Carlo samples
    """

    recognized_options = SurfaceHoppingMD.recognized_options + [ "spawn_stack", "quadrature", "mcsamples" ]

    def __init__(self, *args, **options):
        """Initialize the EvenSamplingTrajectory.

        Parameters
        ----------
        *args : tuple
            Arguments passed to SurfaceHoppingMD constructor
        **options : dict
            Options passed to SurfaceHoppingMD constructor
        spawn_stack : SpawnStack, optional
            Stack for managing trajectory spawning
        quadrature : dict, optional
            Quadrature parameters for sampling
        mcsamples : int, optional
            Number of Monte Carlo samples
        """
        options['hopping_method'] = 'cumulative'  # Force cumulative hopping
        SurfaceHoppingMD.__init__(self, *args, **options)

        ss = options["spawn_stack"]
        if isinstance(ss, SpawnStack):
            self.spawn_stack = cp.deepcopy(options["spawn_stack"])
        elif isinstance(ss, list):
            quadrature = options.get("quadrature", "gl")
            mcsamples = options.get("mcsamples", 1)
            self.spawn_stack = SpawnStack.from_quadrature(
                ss, method=quadrature, mcsamples=mcsamples, random_state=self.random_state
            )
        else:
            self.spawn_stack = SpawnStack(ss)

        self.zeta = self.spawn_stack.next_zeta(0.0, self.random_state)

    def clone(self, spawn_stack: Optional[Any] = None) -> "EvenSamplingTrajectory":
        """Create a clone of the current trajectory.

        Parameters
        ----------
        spawn_stack : Any, optional
            New spawn stack to use for the clone, by default None

        Returns
        -------
        EvenSamplingTrajectory
            Cloned trajectory instance.
        """
        if spawn_stack is None:
            spawn_stack = self.spawn_stack

        out = EvenSamplingTrajectory(
            self.model,
            self.position,
            self.velocity,
            self.rho,
            tracer=self.tracer.clone(),
            queue=self.queue,
            last_velocity=self.last_velocity,
            state0=self.state,
            t0=self.time,
            previous_steps=self.nsteps,
            trace_every=self.trace_every,
            dt=self.dt,
            outcome_type=self.outcome_type,
            seed_sequence=self.seed_sequence.spawn(1)[0],
            electronics=self.electronics,
            duration=self.duration,
            max_steps=self.max_steps,
            max_time=self.max_time,
            hopping_method=self.hopping_method,
            spawn_stack=spawn_stack,
        )
        return out

    def hopper(self, probs: ArrayLike) -> List[Dict[str, Union[int, float]]]:
        """Determine whether and where to hop based on probabilities.

        Parameters
        ----------
        probs : ArrayLike
            Array of individual hopping probabilities.

        Returns
        -------
        List[Dict[str, Union[int, float]]]
            List of dictionaries containing target state and hop weight information.
        """
        accumulated = self.prob_cum
        probs[self.state] = 0.0  # ensure self-hopping is nonsense
        gkdt = np.sum(probs)

        accumulated = 1 - (1 - accumulated) * np.exp(-gkdt)
        if accumulated > self.zeta:  # then hop
            zeta = self.zeta
            next_zeta = self.spawn_stack.next_zeta(accumulated, self.random_state)

            # where to hop
            hop_choice = probs / gkdt
            if self.spawn_stack.do_spawn():
                nspawn = self.spawn_stack.spawn_size()
                spawn_weight = 1.0 / nspawn
                targets = [
                    {
                        "target": i,
                        "weight": hop_choice[i],
                        "zeta": zeta,
                        "prob": accumulated,
                        "stack": self.spawn_stack.spawn(spawn_weight * hop_choice[i]),
                    }
                    for i in range(self.model.nstates)
                    if i != self.state
                    for j in range(nspawn)
                ]
            else:
                target = self.random_state.choice(list(range(self.model.nstates)), p=hop_choice)
                targets = [
                    {
                        "target": target,
                        "weight": 1.0,
                        "zeta": zeta,
                        "prob": accumulated,
                        "stack": self.spawn_stack.spawn(),
                    }
                ]

            # reset probabilities and random
            self.zeta = next_zeta

            self.prob_cum = accumulated
            return targets

        self.prob_cum = accumulated
        return []

    def hop_to_it(self, hop_to: List[Dict[str, Any]], electronics: ElectronicT = None) -> None:
        """Handle hopping by spawning new trajectories.

        This method spawns new trajectories instead of enacting hops directly.
        It must:
        - Copy current trajectory
        - Initiate hops on the copied trajectories
        - Make no changes to current trajectory
        - Set next threshold for spawning

        Parameters
        ----------
        hop_to : List[Dict[str, Any]]
            List of states and associated weights on which to hop.
        electronics : ElectronicT, optional
            Model class for electronic structure calculations, by default None
        """
        if self.spawn_stack.do_spawn():
            for hop in hop_to:
                stack = hop["stack"]
                spawn = self.clone(stack)
                SurfaceHoppingMD.hop_to_it(spawn, [hop], electronics=spawn.electronics)
                spawn.time+= spawn.dt
                spawn.nsteps += 1

                # trigger hop
                spawn.update_weight(stack.weight())

                # add to total trajectory queue
                self.queue.put(spawn)
            self.update_weight(self.spawn_stack.weight())
        else:
            self.prob_cum = 0.0
            SurfaceHoppingMD.hop_to_it(self, hop_to, electronics=self.electronics)
