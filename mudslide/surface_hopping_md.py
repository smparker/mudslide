# -*- coding: utf-8 -*-
"""Propagate FSSH trajectory"""

import copy as cp
from typing import List, Dict, Union, Any


import numpy as np
from numpy.typing import ArrayLike

from .util import check_options
from .constants import boltzmann, fs_to_au
from .propagation import propagate_exponential, propagate_interpolated_rk4
from .tracer import Trace
from .math import poisson_prob_scale
from .surface_hopping_propagator import SHPropagator

class SurfaceHoppingMD:
    """Class to propagate a single FSSH trajectory.

    This class implements the Fewest Switches Surface Hopping (FSSH) algorithm
    for nonadiabatic molecular dynamics simulations. It handles the propagation
    of both nuclear and electronic degrees of freedom, including surface hopping
    events between electronic states.
    """
    recognized_options = [ "propagator", "last_velocity", "bounds",
        "dt", "t0", "previous_steps",
        "duration", "max_steps", "max_time",
        "seed_sequence",
        "outcome_type", "trace_every",
        "electronics",
        "electronic_integration", "max_electronic_dt", "starting_electronic_intervals",
        "weight",
        "restarting",
        "hopping_probability", "zeta_list",
        "state0",
        "hopping_method"
        ]

    def __init__(self,
                 model: Any,
                 x0: np.ndarray,
                 v0: ArrayLike,
                 rho0: Union[ArrayLike, int, str],
                 tracer: Any = "default",
                 queue: Any = None,
                 strict_option_check: bool = True,
                 **options: Any):
        """Initialize the SurfaceHoppingMD class.

        Parameters
        ----------
        model : Any
            Model object defining problem
        x0 : ArrayLike
            Initial position
        v0 : ArrayLike
            Initial velocity
        rho0 : ArrayLike, int, or str
            Initial density matrix or state index. If an integer, populates a single state.
            If a matrix, populates a density matrix (requires state0 for active state).
        tracer : Any, optional
            spawn from TraceManager to collect results, by default "default"
        queue : Any, optional
            Trajectory queue, by default None
        strict_option_check : bool, optional
            Whether to strictly check options, by default True

        Other Parameters
        ----------------
        propagator : str or dict, optional
            The propagator to use for nuclear motion. Can be a string (e.g., 'vv', 'fssh') or a
            dictionary with more options. Default is 'vv'.
        last_velocity : array-like, optional
            The velocity from the previous step, used for restarts. Default is zeros.
        bounds : tuple or list, optional
            Tuple or list of (lower, upper) bounds for the simulation box. Used to determine if
            the trajectory is inside a region. Default is None.
        duration : dict, optional
            Dictionary controlling simulation duration (overrides max_steps, max_time, etc.).
            Default is auto-generated.
        dt : float, optional
            Time step for nuclear propagation (in atomic units). Default is fs_to_au.
        t0 : float, optional
            Initial time. Default is 0.0.
        previous_steps : int, optional
            Number of previous steps (for restarts). Default is 0.
        trace_every : int, optional
            Interval (in steps) at which to record trajectory data. Default is 1.
        max_steps : int, optional
            Maximum number of simulation steps. Default is 1000000.
        max_time : float, optional
            Maximum simulation time. Default is 1e25.
        seed_sequence : int or numpy.random.SeedSequence, optional
            Seed or SeedSequence for random number generation. Default is None.
        outcome_type : str, optional
            Type of outcome to record (e.g., 'state'). Default is 'state'.
        electronics : object, optional
            Initial electronic state object. Default is None.
        electronic_integration : str, optional
            Method for integrating electronic equations ('exp' or 'linear-rk4'). Default is 'exp'.
        max_electronic_dt : float, optional
            Maximum time step for electronic integration. Default is 0.1.
        starting_electronic_intervals : int, optional
            Initial number of intervals for electronic integration. Default is 4.
        weight : float, optional
            Statistical weight of the trajectory. Default is 1.0.
        restarting : bool, optional
            Whether this is a restarted trajectory. Default is False.
        hopping_probability : str, optional
            Method for computing hopping probability ('tully' or 'poisson'). Default is 'tully'.
        zeta_list : list, optional
            List of pre-determined random numbers for hopping decisions. Default is [].
        state0 : int, optional
            Initial electronic state (used if rho0 is a matrix). Required if rho0 is not scalar.
        hopping_method : str, optional
            Hopping method: 'cumulative', 'cumulative_integrated', or 'instantaneous'.
            Default is 'cumulative'.
        """
        check_options(options, self.recognized_options, strict=strict_option_check)

        self.model = model
        self.mass = model.mass
        self.tracer = Trace(tracer)
        self.queue: Any = queue

        # initial conditions
        self.position = np.array(x0, dtype=np.float64).reshape(model.ndof)
        self.last_position = np.zeros_like(self.position, dtype=np.float64)
        self.velocity = np.array(v0, dtype=np.float64).reshape(model.ndof)
        self.last_velocity = np.zeros_like(self.velocity, dtype=np.float64)
        if "last_velocity" in options:
            self.last_velocity[:] = options["last_velocity"]
        if np.isscalar(rho0):
            try:
                state = int(rho0)
                self.rho = np.zeros([model.nstates, model.nstates], dtype=np.complex128)
                self.rho[state, state] = 1.0
                self.state = state
            except:
                raise ValueError("Initial state rho0 must be convertible to an integer state "
                                 "index")
        else:
            try:
                self.rho = np.copy(rho0)
                self.state = int(options["state0"])
            except KeyError:
                raise KeyError("state0 option required when rho0 is a density matrix")
            except (ValueError, TypeError):
                raise ValueError("state0 option must be convertible to an integer state index")

        # function duration_initialize should get us ready to for future continue_simulating calls
        # that decide whether the simulation has finished
        if "duration" in options:
            self.duration = options["duration"]
        else:
            self.duration_initialize(options)

        # fixed initial parameters
        self.time = float(options.get("t0", 0.0))
        self.nsteps = int(options.get("previous_steps", 0))
        self.max_steps = int(options.get("max_steps", 1000000))
        self.max_time = float(options.get("max_time", 1e25))
        self.trace_every = int(options.get("trace_every", 1))
        self.dt = float(options.get("dt", fs_to_au))
        self.propagator = SHPropagator(self.model, options.get("propagator", "vv"))

        self.outcome_type = options.get("outcome_type", "state")

        ss = options.get("seed_sequence", None)
        self.seed_sequence = ss if isinstance(ss, np.random.SeedSequence) \
            else np.random.SeedSequence(ss)
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
        hopping_method = options.get("hopping_method", "cumulative")
        # allow shortened aliases
        aliases = {
            "i": "instantaneous",
            "c": "cumulative",
            "ci": "cumulative_integrated"
        }
        self.hopping_method = hopping_method
        if self.hopping_method in aliases:
            self.hopping_method = aliases[hopping_method]
        allowed_methods = ["instantaneous", "cumulative", "cumulative_integrated"]
        if self.hopping_method not in allowed_methods:
            raise ValueError(f"hopping_method should be one of {allowed_methods}")

        if self.hopping_method in ["cumulative", "cumulative_integrated"]:
            self.prob_cum = np.longdouble(0.0)
            self.zeta = self.draw_new_zeta()
            if self.hopping_method == "cumulative_integrated":
                self.zeta = -np.log(1.0 - self.zeta)

    @classmethod
    def restart(cls, model, log, **options) -> 'SurfaceHoppingMD':
        """Restart a simulation from a previous trajectory log.

        Parameters
        ----------
        model : Any
            Model object defining problem
        log : Any
            Previous trajectory log
        **options : Any
            Additional options for the simulation

        Returns
        -------
        SurfaceHoppingMD
            New instance initialized from the log data
        """
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
        """Update weight held by trajectory and by trace.

        Parameters
        ----------
        weight : float
            New weight value
        """
        self.weight = weight
        self.tracer.weight = weight

        if self.weight == 0.0:
            self.force_quit = True

    def __deepcopy__(self, memo: Any) -> 'SurfaceHoppingMD':
        """Override deepcopy.

        Parameters
        ----------
        memo : Any
            Memo dictionary for deepcopy

        Returns
        -------
        SurfaceHoppingMD
            Deep copy of the instance
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        shallow_only = ["queue"]
        for k, v in self.__dict__.items():
            setattr(result, k, cp.deepcopy(v, memo) if v not in shallow_only else cp.copy(v))
        return result

    def clone(self) -> 'SurfaceHoppingMD':
        """Clone existing trajectory for spawning.

        Returns
        -------
        SurfaceHoppingMD
            Copy of current object
        """
        return cp.deepcopy(self)

    def random(self) -> np.float64:
        """Get random number for hopping decisions.

        Returns
        -------
        np.float64
            Uniform random number between 0 and 1
        """
        return self.random_state.uniform()

    def currently_interacting(self) -> bool:
        """Determine whether trajectory is currently inside an interaction region.

        Returns
        -------
        bool
            True if trajectory is inside interaction region, False otherwise
        """
        if self.duration["box_bounds"] is None:
            return False
        return np.all(self.duration["box_bounds"][0] < self.position) and np.all(
            self.position < self.duration["box_bounds"][1])

    def duration_initialize(self, options: Dict[str, Any]) -> None:
        """Initialize variables related to continue_simulating.

        Parameters
        ----------
        options : Dict[str, Any]
            Dictionary with options for simulation duration
        """
        duration = {}  # type: Dict[str, Any]
        duration['found_box'] = False

        bounds = options.get('bounds', None)
        if bounds:
            duration["box_bounds"] = (np.array(bounds[0], dtype=np.float64),
                                      np.array(bounds[1], dtype=np.float64))
        else:
            duration["box_bounds"] = None

        self.duration = duration

    def continue_simulating(self) -> bool:
        """Decide whether trajectory should keep running.

        Returns
        -------
        bool
            True if trajectory should keep running, False if it should finish
        """
        if self.force_quit:
            return False
        elif self.max_steps >= 0 and self.nsteps >= self.max_steps:
            return False
        elif self.time >= self.max_time or np.isclose(
                self.time, self.max_time, atol=1e-8, rtol=0.0):
            return False
        elif self.duration["found_box"]:
            return self.currently_interacting()
        else:
            if self.currently_interacting():
                self.duration["found_box"] = True
            return True

    def trace(self, force: bool = False) -> None:
        """Add results from current time point to tracing function.

        Only adds snapshot if nsteps%trace_every == 0, unless force=True.

        Parameters
        ----------
        force : bool, optional
            Force snapshot regardless of trace_every, by default False
        """
        if force or (self.nsteps % self.trace_every) == 0:
            self.tracer.collect(self.snapshot())
            #self.trouble_shooter()

    def snapshot(self) -> Dict[str, Any]:
        """Collect data from run for logging.

        Returns
        -------
        Dict[str, Any]
            Dictionary with all data from current time step
        """
        out = {
            "time": float(self.time),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "potential": float(self.potential_energy()),
            "kinetic": float(self.kinetic_energy()),
            "temperature": float(2 * self.kinetic_energy() / ( boltzmann * self.model.ndof)),
            "energy": float(self.total_energy()),
            "density_matrix": self.rho.view(dtype=np.float64).tolist(),
            "active": int(self.state),
            "electronics": self.electronics.as_dict(),
            "hopping": float(self.hopping),
            "zeta": float(self.zeta)
        }
        if self.hopping_method in ["cumulative", "cumulative_integrated"]:
            out["prob_cum"] = float(self.prob_cum)
        return out

    def kinetic_energy(self) -> np.float64:
        """Calculate kinetic energy.

        Returns
        -------
        np.float64
            Kinetic energy
        """
        return 0.5 * np.sum(self.mass * self.velocity**2)

    def potential_energy(self, electronics: 'ElectronicModel_' = None) -> np.floating:
        """Calculate potential energy.

        Parameters
        ----------
        electronics : ElectronicModel, optional
            electronic states from current step, by default None

        Returns
        -------
        np.floating
            Potential energy
        """
        if electronics is None:
            electronics = self.electronics
        return electronics.hamiltonian[self.state, self.state]

    def total_energy(self, electronics: 'ElectronicModel_' = None) -> np.floating:
        """Calculate total energy (kinetic + potential).

        Parameters
        ----------
        electronics : ElectronicModel, optional
            Electronic states from current step, by default None

        Returns
        -------
        np.floating
            Total energy
        """
        potential = self.potential_energy(electronics)
        kinetic = self.kinetic_energy()
        return potential + kinetic

    def _force(self, electronics: 'ElectronicModel_' = None) -> ArrayLike:
        """Compute force on active state.

        Parameters
        ----------
        electronics : ElectronicModel, optional
            Electronic states from current step, by default None

        Returns
        -------
        ArrayLike
            Force on active electronic state
        """
        if electronics is None:
            electronics = self.electronics
        return electronics.force(self.state)

    def NAC_matrix(self, electronics: 'ElectronicModel_' = None,
                   velocity: ArrayLike = None) -> ArrayLike:
        """Calculate nonadiabatic coupling matrix.

        Parameters
        ----------
        electronics : ElectronicModel, optional
            electronic states from current step, by default None
        velocity : ArrayLike, optional
            Velocity used to compute NAC, by default None

        Returns
        -------
        ArrayLike
            NAC matrix
        """
        velo = velocity if velocity is not None else self.velocity
        if electronics is None:
            electronics = self.electronics
        return electronics.NAC_matrix(velo)

    def mode_kinetic_energy(self, direction: ArrayLike) -> np.float64:
        """Calculate kinetic energy along given momentum mode.

        Parameters
        ----------
        direction : ArrayLike
            Numpy array defining direction

        Returns
        -------
        np.float64
            Kinetic energy along specified direction
        """
        u = direction / np.linalg.norm(direction)
        momentum = self.velocity * self.mass
        component = np.dot(u, momentum) * u
        return 0.5 * np.einsum('m,m,m', 1.0 / self.mass, component, component)

    def draw_new_zeta(self) -> float:
        """Get a new zeta value for hopping.

        First checks the input list of zeta values in self.zeta_list.
        If no value is found in zeta_list, then a random number is pulled.

        Returns
        -------
        float
            Zeta value for hopping
        """
        if self.zeta_list:
            return self.zeta_list.pop(0)
        else:
            return self.random()

    def hop_allowed(self, direction: ArrayLike, dE: float) -> bool:
        """Determine if a hop with given rescale direction and energy change is allowed.

        Parameters
        ----------
        direction : ArrayLike
            Momentum unit vector
        dE : float
            Change in energy such that Enew = Eold + dE

        Returns
        -------
        bool
            Whether the hop is allowed
        """
        if dE > 0.0:
            return True
        u = direction / np.linalg.norm(direction)
        a = np.sum(u**2 / self.mass)
        b = 2.0 * np.dot(self.velocity, u)
        c = -2.0 * dE
        return b * b > 4.0 * a * c

    def direction_of_rescale(self, source: int, target: int,
                             electronics: 'ElectronicModel_' = None) -> np.ndarray:
        """
        Return direction in which to rescale momentum.

        Parameters
        ----------
        source : int
            Active state before hop
        target : int
            Active state after hop
        electronics : ElectronicModel, optional
            Electronic model information (used to pull derivative coupling), by default None

        Returns
        -------
        np.ndarray
            Unit vector pointing in direction of rescale
        """
        elec_states = self.electronics if electronics is None else electronics
        out = elec_states.derivative_coupling(source, target)
        return np.copy(out)

    def rescale_component(self, direction: ArrayLike, reduction: np.floating) -> None:
        """
        Update velocity by rescaling the *momentum* in the specified direction and amount.

        Parameters
        ----------
        direction : ArrayLike
            The direction of the *momentum* to rescale
        reduction : np.floating
            How much kinetic energy should be damped
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
                               last_electronics: 'ElectronicModel_',
                               this_electronics: 'ElectronicModel_',
                               velo: ArrayLike = None) -> np.ndarray:
        """
        Compute the Hamiltonian used to propagate the electronic wavefunction.

        Parameters
        ----------
        last_electronics : ElectronicModel
            Electronic states at previous time step
        this_electronics : ElectronicModel
            Electronic states at current time step
        velo : ArrayLike, optional
            Velocity at midpoint between current and previous time steps, by default None

        Returns
        -------
        np.ndarray
            Nonadiabatic coupling Hamiltonian at midpoint between current and previous time steps
        """
        if velo is None:
            velo = 0.5 * (self.velocity + self.last_velocity)
        if last_electronics is None:
            last_electronics = this_electronics

        H = 0.5 * (this_electronics.hamiltonian + last_electronics.hamiltonian)  # type: ignore
        this_tau = this_electronics.derivative_coupling_tensor
        last_tau = last_electronics.derivative_coupling_tensor
        TV = 0.5 * np.einsum("ijx,x->ij", this_tau + last_tau, velo)
        return H - 1j * TV

    def propagate_electronics(self, last_electronics: 'ElectronicModel_',
                              this_electronics: 'ElectronicModel_',
                              dt: np.floating) -> None:
        """
        Propagate density matrix from t to t+dt.

        The propagation assumes the electronic energies and couplings are static throughout.
        This will only be true for fairly small time steps.

        Parameters
        ----------
        last_electronics : ElectronicModel
            Electronic states at t
        this_electronics : ElectronicModel
            Electronic states at t+dt
        dt : np.floating
            Time step
        """
        if self.electronic_integration == "exp":
            # Use midpoint propagator
            W = self.hamiltonian_propagator(last_electronics, this_electronics)

            propagate_exponential(self.rho, W, self.dt)
        elif self.electronic_integration == "linear-rk4":
            nsteps = self.starting_electronic_intervals
            while (dt / nsteps) > self.max_electronic_dt:
                nsteps *= 2

            this_tau = this_electronics.derivative_coupling_tensor
            last_tau = last_electronics.derivative_coupling_tensor
            propagate_interpolated_rk4(self.rho,
                    last_electronics.hamiltonian, last_tau, self.last_velocity,
                    this_electronics.hamiltonian, this_tau, self.velocity,
                    self.dt, nsteps)
        else:
            raise ValueError(
                f"Unrecognized electronic integration option: {self.electronic_integration}. "
                "Must be one of ['exp', 'linear-rk4']"
            )

    def surface_hopping(self, last_electronics: 'ElectronicModel_',
                        this_electronics: 'ElectronicModel_'):
        """
        Compute probability of hopping, generate random number, and perform hops.

        Parameters
        ----------
        last_electronics : ElectronicModel
            Electronic states at previous time step
        this_electronics : ElectronicModel
            Electronic states at current time step
        """
        H = self.hamiltonian_propagator(last_electronics, this_electronics)

        gkndt = 2.0 * np.imag(self.rho[self.state, :] * H[:, self.state]) * \
            self.dt / np.real(self.rho[self.state, self.state])

        # zero out 'self-hop' for good measure (numerical safety)
        gkndt[self.state] = 0.0

        # clip probabilities to make sure they are between zero and one
        gkndt = np.maximum(gkndt, 0.0)

        hop_targets = self.hopper(gkndt)
        if hop_targets:
            self.hop_to_it(hop_targets, this_electronics)

    def hopper(self, gkndt: np.ndarray) -> List[Dict[str, float]]:
        """
        Determine whether and where to hop.

        Parameters
        ----------
        gkndt : np.ndarray
            Array of individual hopping probabilities

        Returns
        -------
        List[Dict[str, float]]
            List of dictionaries with (target_state, weight) pairs
        """
        probs = np.zeros_like(gkndt)
        if self.hopping_probability == "tully":
            probs = gkndt
        elif self.hopping_probability == "poisson":
            probs = gkndt * poisson_prob_scale(np.sum(gkndt))
        self.hopping = np.sum(probs).item()  # store total hopping probability

        if self.hopping_method in ["cumulative", "cumulative_integrated"]:
            accumulated = np.longdouble(self.prob_cum)
            gkdt = np.sum(gkndt)
            if self.hopping_method == "cumulative":
                accumulated += (accumulated - 1.0) * np.expm1(-gkdt)
            elif self.hopping_method == "cumulative_integrated":
                accumulated += gkdt
            else:
                raise ValueError(f"Unrecognized hopping method: {self.hopping_method}")

            if accumulated > self.zeta:  # then hop
                # where to hop
                hop_choice = gkndt / gkdt
                zeta = self.zeta
                target = self.random_state.choice(list(range(self.model.nstates)), p=hop_choice)

                # reset probabilities and random
                self.prob_cum = 0.0
                self.zeta = self.draw_new_zeta()
                if self.hopping_method == "cumulative_integrated":
                    self.zeta = -np.log(1.0 - self.zeta)

                return [{"target": target, "weight": 1.0, "zeta": zeta, "prob": accumulated}]

            self.prob_cum = accumulated
            return []
        else:  # instantaneous
            self.zeta = self.draw_new_zeta()
            acc_prob = np.cumsum(probs)
            hops = np.less(self.zeta, acc_prob)
            if any(hops):
                hop_to = -1
                for i in range(self.model.nstates):
                    if hops[i]:
                        hop_to = i
                        break

                return [{
                    "target": hop_to,
                    "weight": 1.0,
                    "zeta": self.zeta,
                    "prob": acc_prob[hop_to]
                }]
            else:
                return []

    def hop_update(self, hop_from, hop_to):  # pylint: disable=unused-argument
        """
        Handle any extra operations that need to occur after a hop.

        Parameters
        ----------
        hop_from : int
            State before hop
        hop_to : int
            State after hop
        """
        return

    def hop_to_it(self, 
                  hop_targets: List[Dict[str, Union[float,int]]], 
                  electronics: 'ElectronicModel_' = None) -> None:
        """
        Hop from the current active state to the given state, including rescaling the momentum.

        Parameters
        ----------
        hop_targets : List[Dict[str, Union[float, int]]]
            List of (target, weight) pairs
        electronics : ElectronicModel, optional
            Electronic states for current step, by default None
        """
        hop_dict = hop_targets[0]
        hop_to = int(hop_dict["target"])
        elec_states = electronics if electronics is not None else self.electronics
        H = elec_states.hamiltonian
        new_potential, old_potential = H[hop_to, hop_to], H[self.state, self.state]
        delV = new_potential - old_potential
        rescale_vector = self.direction_of_rescale(self.state, hop_to)
        hop_from = self.state

        if self.hop_allowed(rescale_vector, -delV):
            self.state = hop_to
            self.rescale_component(rescale_vector, -delV)
            self.hop_update(hop_from, hop_to)
            self.tracer.record_event(
                event_dict={
                    "hop_from": int(hop_from),
                    "hop_to": int(hop_to),
                    "zeta": float(hop_dict["zeta"]),
                    "prob": float(hop_dict["prob"])
                },
                event_type="hop"
            )
        else:
            self.tracer.record_event(
                event_dict={
                    "hop_from": int(hop_from),
                    "hop_to": int(hop_to),
                    "zeta": float(hop_dict["zeta"]),
                    "prob": float(hop_dict["prob"])
                },
                event_type="frustrated_hop"
            )
    def simulate(self) -> 'Trace':
        """
        Run the surface hopping molecular dynamics simulation.

        Returns
        -------
        Trace
            Trace of trajectory
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
