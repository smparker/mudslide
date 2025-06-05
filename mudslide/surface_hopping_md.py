# -*- coding: utf-8 -*-
"""Propagate FSSH trajectory"""

from typing import List, Dict, Union, Any
import copy as cp

import numpy as np

from .typing import ElectronicT, ArrayLike, DtypeLike

from .util import check_options, is_string
from .constants import boltzmann, fs_to_au
from .propagation import propagate_exponential, propagate_interpolated_rk4
from .tracer import Trace
from .math import poisson_prob_scale
from .surface_hopping_propagator import SHPropagator

class SurfaceHoppingMD(object):
    """Class to propagate a single FSSH trajectory"""
    recognized_options = [ "propagator", "last_velocity", "bounds",
        "duration", "dt", "t0", "previous_steps", "trace_every", "max_time",
        "seed_sequence",
        "outcome_type",
        "electronics", "electronic_integration", "max_electronic_dt", "starting_electronic_intervals",
        "weight",
        "restarting",
        "hopping_probability", "zeta_list",
        "state0",
        "hopping_method"
        ]

    def __init__(self,
                 model: Any,
                 x0: ArrayLike,
                 v0: ArrayLike,
                 rho0: ArrayLike,
                 tracer: Any = "default",
                 queue: Any = None,
                 strict_option_check: bool = True,
                 **options: Any):
        """Constructor
        :param model: Model object defining problem
        :param x0: Initial position
        :param v0: Initial velocity
        :param rho0: Initial density matrix
        :param tracer: spawn from TraceManager to collect results
        :param queue: Trajectory queue
        :param options: option dictionary
        """
        check_options(options, self.recognized_options, strict=strict_option_check)

        self.model = model
        self.mass = model.mass
        self.tracer = Trace(tracer)
        self.queue: Any = queue

        # initial conditions
        self.position = np.array(x0, dtype=np.float64).reshape(model.ndim())
        self.last_position = np.zeros_like(self.position, dtype=np.float64)
        self.velocity = np.array(v0, dtype=np.float64).reshape(model.ndim())
        self.last_velocity = np.zeros_like(self.velocity, dtype=np.float64)
        if "last_velocity" in options:
            self.last_velocity[:] = options["last_velocity"]
        if np.isscalar(rho0):
            try:
                state = int(rho0)
                self.rho = np.zeros([model.nstates(), model.nstates()], dtype=np.complex128)
                self.rho[state, state] = 1.0
                self.state = state
            except:
                raise Exception("Unrecognized initial state option")
        else:
            try:
                self.rho = np.copy(rho0)
                self.state = int(options["state0"])
            except:
                raise Exception("Unrecognized initial state option")

        # function duration_initialize should get us ready to for future continue_simulating calls
        # that decide whether the simulation has finished
        if "duration" in options:
            self.duration = options["duration"]
        else:
            self.duration_initialize(options)

        # fixed initial parameters
        self.time = float(options.get("t0", 0.0))
        self.nsteps = int(options.get("previous_steps", 0))
        self.trace_every = int(options.get("trace_every", 1))
        self.dt = float(options.get("dt", fs_to_au))
        self.propagator = SHPropagator(self.model, options.get("propagator", "vv"))

        self.outcome_type = options.get("outcome_type", "state")

        ss = options.get("seed_sequence", None)
        self.seed_sequence = ss if isinstance(ss, np.random.SeedSequence) else np.random.SeedSequence(ss)
        self.random_state = np.random.default_rng(self.seed_sequence)

        self.electronics = options.get("electronics", None)
        self.last_electronics = options.get("last_electronics", None)
        self.hopping = 0.0

        self.electronic_integration = options.get("electronic_integration", "exp").lower()
        self.max_electronic_dt = options.get("max_electronic_dt", 0.1)
        self.starting_electronic_intervals = options.get("starting_electronic_intervals", 4)

        self.weight = float(options.get("weight", 1.0))

        self.restarting = options.get("restarting", False)
        self.force_quit = False

        self.hopping_probability = options.get("hopping_probability", "tully")
        if self.hopping_probability not in ["tully", "poisson"]:
            raise ValueError("hopping_probability accepts only \"tully\" or \"poisson\" options")

        self.zeta_list = list(options.get("zeta_list", []))
        self.zeta = 0.0

        # Add hopping method option
        self.hopping_method = options.get("hopping_method", "cumulative")
        if self.hopping_method not in ["instantaneous", "cumulative"]:
            raise ValueError("hopping_method accepts only \"instantaneous\" or \"cumulative\" options")
        
        if self.hopping_method == "cumulative":
            self.prob_cum = np.longdouble(0.0)
            self.zeta = self.draw_new_zeta()

    @classmethod
    def restart(cls, model, log, **options) -> 'SurfaceHoppingMD':
        last_snap = log[-1]
        penultimate_snap = log[-2]

        x = last_snap["position"]
        v = np.array(last_snap["velocity"])
        last_velocity = np.array(penultimate_snap["velocity"])
        t0 = last_snap["time"]
        dt = t0 - penultimate_snap["time"]
        k = last_snap["active"]
        rho = last_snap["density_matrix"]
        weight = log.weight
        previous_steps = len(log)

        # use inferred data if available, but let kwargs override
        for key, val in [["dt", dt]]:
            if key not in options:
                options[key] = val

        return cls(model,
                   x,
                   v,
                   rho,
                   tracer=log,
                   state0=k,
                   t0=t0,
                   last_velocity=last_velocity,
                   weight=weight,
                   previous_steps=previous_steps,
                   restarting=True,
                   **options)

    def update_weight(self, weight: float) -> None:
        """Update weight held by trajectory and by trace"""
        self.weight = weight
        self.tracer.weight = weight

        if self.weight == 0.0:
            self.force_quit = True

    def __deepcopy__(self, memo: Any) -> 'SurfaceHoppingMD':
        """Override deepcopy"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        shallow_only = ["queue"]
        for k, v in self.__dict__.items():
            setattr(result, k, cp.deepcopy(v, memo) if v not in shallow_only else cp.copy(v))
        return result

    def clone(self) -> 'SurfaceHoppingMD':
        """Clone existing trajectory for spawning

        :return: copy of current object
        """
        return cp.deepcopy(self)

    def random(self) -> np.float64:
        """Get random number for hopping decisions

        :return: uniform random number between 0 and 1
        """
        return self.random_state.uniform()

    def currently_interacting(self) -> bool:
        """Determines whether trajectory is currently inside an interaction region

        :return: boolean
        """
        if self.duration["box_bounds"] is None:
            return False
        return np.all(self.duration["box_bounds"][0] < self.position) and np.all(
            self.position < self.duration["box_bounds"][1])

    def duration_initialize(self, options: Dict[str, Any]) -> None:
        """Initializes variables related to continue_simulating

        :param options: dictionary with options
        """

        duration = {}  # type: Dict[str, Any]
        duration['found_box'] = False

        bounds = options.get('bounds', None)
        if bounds:
            duration["box_bounds"] = (np.array(bounds[0], dtype=np.float64), np.array(bounds[1], dtype=np.float64))
        else:
            duration["box_bounds"] = None
        duration["max_steps"] = options.get('max_steps', 100000)  # < 0 interpreted as no limit
        duration["max_time"] = options.get('max_time', 1e25)  # set to an outrageous number by default

        self.duration = duration

    def continue_simulating(self) -> bool:
        """Decide whether trajectory should keep running

        :return: True if trajectory should keep running, False if it should finish
        """
        if self.force_quit:
            return False
        elif self.duration["max_steps"] >= 0 and self.nsteps >= self.duration["max_steps"]:
            return False
        elif self.time >= self.duration["max_time"] or np.isclose(
                self.time, self.duration["max_time"], atol=1e-8, rtol=0.0):
            return False
        elif self.duration["found_box"]:
            return self.currently_interacting()
        else:
            if self.currently_interacting():
                self.duration["found_box"] = True
            return True

    def trace(self, force: bool = False) -> None:
        """Add results from current time point to tracing function
        Only adds snapshot if nsteps%trace_every == 0, unless force=True

        :param force: force snapshot
        """
        if force or (self.nsteps % self.trace_every) == 0:
            self.tracer.collect(self.snapshot())
            #self.trouble_shooter()

    def snapshot(self) -> Dict[str, Any]:
        """Collect data from run for logging

        :return: dictionary with all data from current time step
        """
        out = {
            "time": float(self.time),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "potential": float(self.potential_energy()),
            "kinetic": float(self.kinetic_energy()),
            "temperature": float(2 * self.kinetic_energy() / ( boltzmann * self.model.ndim())),
            "energy": float(self.total_energy()),
            "density_matrix": self.rho.view(dtype=np.float64).tolist(),
            "active": int(self.state),
            "electronics": self.electronics.as_dict(),
            "hopping": float(self.hopping),
            "zeta": float(self.zeta)
        }
        if self.hopping_method == "cumulative":
            out["prob_cum"] = float(self.prob_cum)
        return out

    def trouble_shooter(self):
        log = self.snapshot()
        with open("snapout.dat", "a") as file:
            file.write("{}\t{}\t{}\t{}\t{}\n".format(log["time"], log["potential"], log["kinetic"], log["energy"],
                                                     log["active"]))

    def kinetic_energy(self) -> np.float64:
        """Kinetic energy

        :return: kinetic energy
        """
        return 0.5 * np.sum(self.mass * self.velocity**2)

    def potential_energy(self, electronics: ElectronicT = None) -> DtypeLike:
        """Potential energy

        :param electronics: ElectronicStates from current step
        :return: potential energy
        """
        if electronics is None:
            electronics = self.electronics
        return electronics.hamiltonian()[self.state, self.state]

    def total_energy(self, electronics: ElectronicT = None) -> DtypeLike:
        """
        Kinetic energy + Potential energy

        :param electronics: ElectronicStates from current step
        :return: total energy
        """
        potential = self.potential_energy(electronics)
        kinetic = self.kinetic_energy()
        return potential + kinetic

    def _force(self, electronics: ElectronicT = None) -> ArrayLike:
        """
        Compute force on active state

        :param electronics: ElectronicStates from current step

        :return: [ndim] force on active electronic state
        """
        if electronics is None:
            electronics = self.electronics
        return electronics.force(self.state)

    def NAC_matrix(self, electronics: ElectronicT = None, velocity: ArrayLike = None) -> ArrayLike:
        """
        Nonadiabatic coupling matrix

        :param electronics: ElectronicStates from current step
        :param velocity: velocity used to compute NAC (defaults to self.velocity)

        :return: [nstates, nstates] NAC matrix
        """
        velo = velocity if velocity is not None else self.velocity
        if electronics is None:
            electronics = self.electronics
        return electronics.NAC_matrix(velo)

    def mode_kinetic_energy(self, direction: ArrayLike) -> np.float64:
        """
        Kinetic energy along given momentum mode

        :param direction: [ndim] numpy array defining direction

        :return: kinetic energy along specified direction
        """
        u = direction / np.linalg.norm(direction)
        momentum = self.velocity * self.mass
        component = np.dot(u, momentum) * u
        return 0.5 * np.einsum('m,m,m', 1.0 / self.mass, component, component)

    def draw_new_zeta(self) -> float:
        """
        Returns a new zeta value for hopping. First it checks the input list of
        zeta values in self.zeta_list. If no value is found in zeta_list, then
        a random number is pulled.

        :returns: zeta
        """
        if self.zeta_list:
            return self.zeta_list.pop(0)
        else:
            return self.random()

    def hop_allowed(self, direction: ArrayLike, dE: float):
        """
        Returns whether a hop with the given rescale direction and requested energy change
        is allowed

        :param direction: momentum unit vector
        :param dE: change in energy such that Enew = Eold + dE

        :return: whether to hop
        """
        if dE > 0.0:
            return True
        u = direction / np.linalg.norm(direction)
        a = np.sum(u**2 / self.mass)
        b = 2.0 * np.dot(self.velocity, u)
        c = -2.0 * dE
        return (b * b > 4.0 * a * c)

    def direction_of_rescale(self, source: int, target: int, electronics: ElectronicT = None) -> np.ndarray:
        """
        Return direction in which to rescale momentum

        :param source: active state before hop
        :param target: active state after hop
        :param electronics: electronic model information (used to pull derivative coupling)

        :return: unit vector pointing in direction of rescale
        """
        elec_states = self.electronics if electronics is None else electronics
        out = elec_states.derivative_coupling(source, target)
        return np.copy(out)

    def rescale_component(self, direction: ArrayLike, reduction: DtypeLike) -> None:
        """
        Updates velocity by rescaling the *momentum* in the specified direction and amount

        :param direction: the direction of the *momentum* to rescale
        :param reduction: how much kinetic energy should be damped
        """
        # normalize
        u = direction / np.linalg.norm(direction)
        M_inv = 1.0 / self.mass
        a = np.einsum('m,m,m', M_inv, u, u)
        b = 2.0 * np.dot(self.velocity, u)
        c = -2.0 * reduction
        roots = np.roots([a, b, c])
        scal = min(roots, key=lambda x: abs(x))
        self.velocity += scal * M_inv * u

    def hamiltonian_propagator(self,
                               last_electronics: ElectronicT,
                               this_electronics: ElectronicT,
                               velo: ArrayLike = None) -> np.ndarray:
        """Compute the Hamiltonian used to propagate the electronic wavefunction

        :param elec_states: ElectronicStates at current time step
        :return: nonadiabatic coupling H - i W at midpoint between current and previous time steps
        """
        if velo is None:
            velo = 0.5 * (self.velocity + self.last_velocity)
        if last_electronics is None:
            last_electronics = this_electronics

        H = 0.5 * (this_electronics.hamiltonian() + last_electronics.hamiltonian())  # type: ignore
        TV = 0.5 * np.einsum(
            "ijx,x->ij",
            this_electronics._derivative_coupling + last_electronics._derivative_coupling,  #type: ignore
            velo)
        return H - 1j * TV

    def propagate_electronics(self, last_electronics: ElectronicT, this_electronics: ElectronicT,
                              dt: DtypeLike) -> None:
        """Propagates density matrix from t to t+dt

        The propagation assumes the electronic energies and couplings are static throughout.
        This will only be true for fairly small time steps

        :param elec_states: ElectronicStates at t
        :param dt: time step
        """
        if self.electronic_integration == "exp":
            # Use midpoint propagator
            W = self.hamiltonian_propagator(last_electronics, this_electronics)

            propagate_exponential(self.rho, W, self.dt)
        elif self.electronic_integration == "linear-rk4":
            nsteps = self.starting_electronic_intervals
            while (dt / nsteps > self.max_electronic_dt):
                nsteps *= 2

            propagate_interpolated_rk4(self.rho,
                    last_electronics.hamiltonian(), last_electronics.derivative_coupling_tensor(), self.last_velocity,
                    this_electronics.hamiltonian(), this_electronics.derivative_coupling_tensor(), self.velocity,
                    self.dt, nsteps)
        else:
            raise Exception("Unrecognized electronic integration option")

    def surface_hopping(self, last_electronics: ElectronicT, this_electronics: ElectronicT):
        """Compute probability of hopping, generate random number, and perform hops

        :param elec_states: ElectronicStates from current step
        """
        H = self.hamiltonian_propagator(last_electronics, this_electronics)

        gkndt = 2.0 * np.imag(self.rho[self.state, :] * H[:, self.state]) * self.dt / np.real(self.rho[self.state,
                                                                                                       self.state])

        # zero out 'self-hop' for good measure (numerical safety)
        gkndt[self.state] = 0.0

        # clip probabilities to make sure they are between zero and one
        gkndt = np.maximum(gkndt, 0.0)

        hop_targets = self.hopper(gkndt)
        if hop_targets:
            self.hop_to_it(hop_targets, this_electronics)

    def hopper(self, gkndt: np.ndarray) -> List[Dict[str, float]]:
        """
        Determine whether and where to hop

        :param probs: [nstates] numpy array of individual hopping probabilities
        :return: [(target_state, weight)] list of (target_state, weight) pairs
        """
        probs = np.zeros_like(gkndt)
        if self.hopping_probability == "tully":
            probs = gkndt
        elif self.hopping_probability == "poisson":
            probs = gkndt * poisson_prob_scale(np.sum(gkndt))
        self.hopping = np.sum(probs).item()  # store total hopping probability

        if self.hopping_method == "cumulative":
            accumulated = np.longdouble(self.prob_cum)
            gkdt = np.sum(gkndt)
            accumulated += (accumulated - 1.0) * np.expm1(-gkdt)
            if accumulated > self.zeta:  # then hop
                # where to hop
                hop_choice = gkndt / gkdt
                zeta = self.zeta
                target = self.random_state.choice(list(range(self.model.nstates())), p=hop_choice)

                # reset probabilities and random
                self.prob_cum = 0.0
                self.zeta = self.draw_new_zeta()

                return [{"target": target, "weight": 1.0, "zeta": zeta, "prob": accumulated}]

            self.prob_cum = accumulated
            return []
        else:  # instantaneous
            self.zeta = self.draw_new_zeta()
            acc_prob = np.cumsum(probs)
            hops = np.less(self.zeta, acc_prob)
            if any(hops):
                hop_to = -1
                for i in range(self.model.nstates()):
                    if hops[i]:
                        hop_to = i
                        break

                return [{"target": hop_to, "weight": 1.0, "zeta": self.zeta, "prob": acc_prob[hop_to]}]
            else:
                return []

    def hop_update(self, hop_from, hop_to):
        """Handle any extra operations that need to occur after a hop"""
        return

    def hop_to_it(self, hop_targets: List[Dict[str, Union[float,int]]], electronics: ElectronicT = None) -> None:
        """
        Hop from the current active state to the given state, including
        rescaling the momentum

        :param hop_targets: list of (target, weight) pairs
        :param electronics: ElectronicStates for current step
        """
        hop_dict = hop_targets[0]
        hop_to = int(hop_dict["target"])
        elec_states = electronics if electronics is not None else self.electronics
        H = elec_states.hamiltonian()
        new_potential, old_potential = H[hop_to, hop_to], H[self.state, self.state]
        delV = new_potential - old_potential
        rescale_vector = self.direction_of_rescale(self.state, hop_to)
        component_kinetic = self.mode_kinetic_energy(rescale_vector)
        hop_from = self.state

        if self.hop_allowed(rescale_vector, -delV):
            self.state = hop_to
            self.rescale_component(rescale_vector, -delV)
            self.hop_update(hop_from, hop_to)
            self.tracer.hop(self.time, hop_from, hop_to, float(hop_dict["zeta"]), float(hop_dict["prob"]))
        else:
            self.tracer.frustrated_hop(self.time, hop_from, hop_to, float(hop_dict["zeta"]), float(hop_dict["prob"]))

    def simulate(self) -> 'Trace':
        """
        Simulate

        :return: Trace of trajectory
        """

        if not self.continue_simulating():
            return self.tracer

        if self.electronics is None:
            self.electronics = self.model.update(self.position)

        if not self.restarting:
            self.trace()

        # propagation
        while True:
            self.propagator(self, 1) # pylint: disable=not-callable

            # ending condition
            if not self.continue_simulating():
                break

            self.trace()

        self.trace(force=True)

        return self.tracer
