# -*- coding: utf-8 -*-
"""Propagate FSSH trajectory"""

from __future__ import annotations

from typing import List, Dict, Union, Any, TYPE_CHECKING

import numpy as np

from .constants import boltzmann
from .propagation import propagate_exponential, propagate_interpolated_rk4
from .math import poisson_prob_scale
from .propagator import Propagator_
from .trajectory_md import TrajectoryMD
from .surface_hopping_propagator import SHPropagator

if TYPE_CHECKING:
    from .models.electronics import ElectronicModel_


class SurfaceHoppingMD(TrajectoryMD):  # pylint: disable=too-many-instance-attributes
    """Class to propagate a single FSSH trajectory.

    This class implements the Fewest Switches Surface Hopping (FSSH) algorithm
    for nonadiabatic molecular dynamics simulations. It handles the propagation
    of both nuclear and electronic degrees of freedom, including surface hopping
    events between electronic states.
    """
    recognized_options = TrajectoryMD.recognized_options + [
        "electronic_integration", "max_electronic_dt",
        "starting_electronic_intervals", "hopping_probability", "zeta_list",
        "state0", "hopping_method", "forced_hop_threshold"
    ]

    def __init__(self,
                 model: Any,
                 x0: np.ndarray,
                 v0: np.ndarray,
                 rho0: Union[np.ndarray, int, str],
                 tracer: Any = "default",
                 queue: Any = None,
                 strict_option_check: bool = True,
                 **options: Any):
        """Initialize the SurfaceHoppingMD class.

        Parameters
        ----------
        model : Any
            Model object defining problem
        x0 : np.ndarray
            Initial position
        v0 : np.ndarray
            Initial velocity
        rho0 : np.ndarray, int, or str
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
        electronic_integration : str, optional
            Method for integrating electronic equations ('exp' or 'linear-rk4').
            Default is 'exp'.
        max_electronic_dt : float, optional
            Maximum time step for electronic integration. Default is 0.1.
        starting_electronic_intervals : int, optional
            Initial number of intervals for electronic integration. Default is 4.
        hopping_probability : str, optional
            Method for computing hopping probability ('tully' or 'poisson').
            Default is 'tully'.
        zeta_list : list, optional
            List of pre-determined random numbers for hopping decisions. Default is [].
        state0 : int, optional
            Initial electronic state (used if rho0 is a matrix). Required if rho0
            is not scalar.
        hopping_method : str, optional
            Hopping method: 'cumulative', 'cumulative_integrated', or 'instantaneous'.
            Default is 'cumulative'.
        forced_hop_threshold : float, optional
            When the energy gap between the two lowest states falls below this threshold
            and the active state is not the lowest, force a hop to the lowest state.
            Default is None (off).
        """
        super().__init__(model,
                         x0,
                         v0,
                         tracer=tracer,
                         queue=queue,
                         strict_option_check=strict_option_check,
                         **options)

        # Process initial electronic state
        if np.isscalar(rho0):
            try:
                state = int(rho0)  # type: ignore[arg-type]
                self.rho = np.zeros([model.nstates, model.nstates],
                                    dtype=np.complex128)
                self.rho[state, state] = 1.0
                self.state = state
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Initial state rho0 must be convertible to an integer state "
                    "index") from exc
        else:
            try:
                self.rho = np.copy(rho0)
                self.state = int(options["state0"])
            except KeyError as exc:
                raise KeyError(
                    "state0 option required when rho0 is a density matrix"
                ) from exc
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "state0 option must be convertible to an integer state index"
                ) from exc

        # Surface hopping specific initialization
        self.hopping = 0.0

        self.electronic_integration = options.get("electronic_integration",
                                                  "exp").lower()
        self.max_electronic_dt = options.get("max_electronic_dt", 0.1)
        self.starting_electronic_intervals = options.get(
            "starting_electronic_intervals", 4)

        self.hopping_probability = options.get("hopping_probability", "tully")
        if self.hopping_probability not in ["tully", "poisson"]:
            raise ValueError(
                "hopping_probability accepts only \"tully\" or \"poisson\" options"
            )

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
        allowed_methods = [
            "instantaneous", "cumulative", "cumulative_integrated"
        ]
        if self.hopping_method not in allowed_methods:
            raise ValueError(
                f"hopping_method should be one of {allowed_methods}")

        self.forced_hop_threshold = options.get("forced_hop_threshold", None)

        if self.hopping_method in ["cumulative", "cumulative_integrated"]:
            self.prob_cum = np.longdouble(0.0)
            self.zeta = self.draw_new_zeta()
            if self.hopping_method == "cumulative_integrated":
                self.zeta = -np.log(1.0 - self.zeta)

    def make_propagator(self, model: Any,
                        options: Dict[str, Any]) -> Propagator_:
        """Create the surface hopping propagator.

        Parameters
        ----------
        model : Any
            Model object defining problem.
        options : Dict[str, Any]
            Options dictionary.

        Returns
        -------
        Propagator_
            Surface hopping propagator instance.
        """
        return SHPropagator(model, options.get("propagator", "vv"))  # type: ignore[return-value]

    @classmethod
    def restart(cls, model: Any, log: Any,
                **options: Any) -> 'SurfaceHoppingMD':
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

    def snapshot(self) -> Dict[str, Any]:
        """Collect data from run for logging.

        Returns
        -------
        Dict[str, Any]
            Dictionary with all data from current time step
        """
        out = super().snapshot()
        out["density_matrix"] = self.rho.view(dtype=np.float64).tolist()
        out["active"] = int(self.state)
        out["hopping"] = float(self.hopping)
        out["zeta"] = float(self.zeta)
        if self.hopping_method in ["cumulative", "cumulative_integrated"]:
            out["prob_cum"] = float(self.prob_cum)
        return out

    def potential_energy(self,
                         electronics: ElectronicModel_ | None = None
                        ) -> float:
        """Calculate potential energy.

        Parameters
        ----------
        electronics : ElectronicModel_, optional
            electronic states from current step, by default None

        Returns
        -------
        float
            Potential energy
        """
        if electronics is None:
            electronics = self.electronics
        assert electronics is not None
        return electronics.hamiltonian[self.state, self.state]

    def force(self,
              electronics: ElectronicModel_ | None = None) -> np.ndarray:
        """Compute force on active state.

        Parameters
        ----------
        electronics : ElectronicModel_, optional
            Electronic states from current step, by default None

        Returns
        -------
        np.ndarray
            Force on active electronic state
        """
        if electronics is None:
            electronics = self.electronics
        assert electronics is not None
        return electronics.force(self.state)

    def needed_gradients(self) -> list[int] | None:
        """States whose forces are needed during normal propagation.

        Returns
        -------
        list[int] | None
            List of state indices for which gradients are needed.
            None means all states are needed.
            Standard FSSH only needs the active state gradient.
        """
        return [self.state]

    def NAC_matrix(self,
                   electronics: ElectronicModel_ | None = None,
                   velocity: np.ndarray | None = None) -> np.ndarray:
        """Calculate nonadiabatic coupling matrix.

        Parameters
        ----------
        electronics : ElectronicModel_, optional
            electronic states from current step, by default None
        velocity : np.ndarray, optional
            Velocity used to compute NAC, by default None

        Returns
        -------
        np.ndarray
            NAC matrix
        """
        velo = velocity if velocity is not None else self.velocity
        if electronics is None:
            electronics = self.electronics
        assert electronics is not None
        return electronics.NAC_matrix(velo)

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
        return self.random()

    def hop_allowed(self, direction: np.ndarray, dE: float) -> bool:
        """Determine if a hop with given rescale direction and energy change is allowed.

        Parameters
        ----------
        direction : np.ndarray
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

    def direction_of_rescale(
            self,
            source: int,
            target: int,
            electronics: ElectronicModel_ | None = None) -> np.ndarray:
        """
        Return direction in which to rescale momentum.

        Parameters
        ----------
        source : int
            Active state before hop
        target : int
            Active state after hop
        electronics : ElectronicModel_, optional
            Electronic model information (used to pull derivative coupling),
            by default None

        Returns
        -------
        np.ndarray
            Unit vector pointing in direction of rescale
        """
        elec_states = self.electronics if electronics is None else electronics
        assert elec_states is not None
        out = elec_states.derivative_coupling(source, target)
        return np.copy(out)

    def rescale_component(self, direction: np.ndarray,
                          reduction: float) -> None:
        """
        Update velocity by rescaling the *momentum* in the specified direction and amount.

        Parameters
        ----------
        direction : np.ndarray
            The direction of the *momentum* to rescale
        reduction : float
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
                               last_electronics: ElectronicModel_,
                               this_electronics: ElectronicModel_,
                               velo: np.ndarray | None = None) -> np.ndarray:
        """
        Compute the Hamiltonian used to propagate the electronic wavefunction.

        Parameters
        ----------
        last_electronics : ElectronicModel_
            Electronic states at previous time step
        this_electronics : ElectronicModel_
            Electronic states at current time step
        velo : np.ndarray, optional
            Velocity at midpoint between current and previous time steps,
            by default None

        Returns
        -------
        np.ndarray
            Nonadiabatic coupling Hamiltonian at midpoint between current
            and previous time steps
        """
        if velo is None:
            velo = 0.5 * (self.velocity + self.last_velocity)
        if last_electronics is None:
            last_electronics = this_electronics

        H = 0.5 * (this_electronics.hamiltonian + last_electronics.hamiltonian
                   )  # type: ignore
        this_tau = this_electronics.derivative_coupling_tensor
        last_tau = last_electronics.derivative_coupling_tensor
        TV = 0.5 * np.einsum("ijx,x->ij", this_tau + last_tau, velo)
        return H - 1j * TV

    def propagate_electronics(self, last_electronics: ElectronicModel_,
                              this_electronics: ElectronicModel_,
                              dt: float) -> None:
        """
        Propagate density matrix from t to t+dt.

        The propagation assumes the electronic energies and couplings are static throughout.
        This will only be true for fairly small time steps.

        Parameters
        ----------
        last_electronics : ElectronicModel_
            Electronic states at t
        this_electronics : ElectronicModel_
            Electronic states at t+dt
        dt : float
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
            propagate_interpolated_rk4(self.rho, last_electronics.hamiltonian,
                                       last_tau, self.last_velocity,
                                       this_electronics.hamiltonian, this_tau,
                                       self.velocity, self.dt, nsteps)
        else:
            raise ValueError(
                f"Unrecognized electronic integration option: {self.electronic_integration}. "
                "Must be one of ['exp', 'linear-rk4']")

    def surface_hopping(self, last_electronics: ElectronicModel_,
                        this_electronics: ElectronicModel_) -> None:
        """
        Compute probability of hopping, generate random number, and perform hops.

        Parameters
        ----------
        last_electronics : ElectronicModel_
            Electronic states at previous time step
        this_electronics : ElectronicModel_
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
            old_state = self.state
            self.hop_to_it(hop_targets, this_electronics)
            if self.state != old_state:
                assert self.electronics is not None
                self.electronics.compute_additional(gradients=[self.state])

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

        # Forced hop check — short-circuit normal hopping logic
        if self.forced_hop_threshold is not None:
            assert self.electronics is not None
            energies = np.diag(self.electronics.hamiltonian).real
            sorted_energies = np.sort(energies)
            gap = sorted_energies[1] - sorted_energies[0]
            if gap < self.forced_hop_threshold and self.state != np.argmin(
                    energies):
                lowest_state = int(np.argmin(energies))
                # Reset hopping state as a normal hop would
                if self.hopping_method in [
                        "cumulative", "cumulative_integrated"
                ]:
                    self.prob_cum = np.longdouble(0.0)
                self.zeta = self.draw_new_zeta()
                if self.hopping_method == "cumulative_integrated":
                    self.zeta = -np.log(1.0 - self.zeta)
                return [{
                    "target": lowest_state,
                    "weight": 1.0,
                    "zeta": self.zeta,
                    "prob": 1.0
                }]

        if self.hopping_method in ["cumulative", "cumulative_integrated"]:
            accumulated = np.longdouble(self.prob_cum)
            gkdt = np.sum(gkndt)
            if self.hopping_method == "cumulative":
                accumulated += (accumulated - 1.0) * np.expm1(-gkdt)
            elif self.hopping_method == "cumulative_integrated":
                accumulated += gkdt
            else:
                raise ValueError(
                    f"Unrecognized hopping method: {self.hopping_method}")

            if accumulated > self.zeta:  # then hop
                # where to hop
                hop_choice = gkndt / gkdt
                zeta = self.zeta
                target = self.random_state.choice(list(range(
                    self.model.nstates)),
                                                  p=hop_choice)

                # reset probabilities and random
                self.prob_cum = np.longdouble(0.0)
                self.zeta = self.draw_new_zeta()
                if self.hopping_method == "cumulative_integrated":
                    self.zeta = -np.log(1.0 - self.zeta)

                return [{
                    "target": target,
                    "weight": 1.0,
                    "zeta": zeta,
                    "prob": accumulated
                }]

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

    def hop_update(self, hop_from: int, hop_to: int) -> None:  # pylint: disable=unused-argument
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
                  hop_targets: List[Dict[str, Union[float, int]]],
                  electronics: ElectronicModel_ | None = None) -> None:
        """
        Hop from the current active state to the given state, including rescaling the momentum.

        Parameters
        ----------
        hop_targets : List[Dict[str, Union[float, int]]]
            List of (target, weight) pairs
        electronics : ElectronicModel_, optional
            Electronic states for current step, by default None
        """
        hop_dict = hop_targets[0]
        hop_to = int(hop_dict["target"])
        elec_states = electronics if electronics is not None else self.electronics
        assert elec_states is not None
        H = elec_states.hamiltonian
        new_potential, old_potential = H[hop_to, hop_to], H[self.state,
                                                            self.state]
        delV = new_potential - old_potential
        rescale_vector = self.direction_of_rescale(self.state, hop_to)
        hop_from = self.state

        if self.hop_allowed(rescale_vector, -delV):
            self.state = hop_to
            self.rescale_component(rescale_vector, -delV)
            self.hop_update(hop_from, hop_to)
            self.tracer.record_event(event_dict={
                "hop_from": int(hop_from),
                "hop_to": int(hop_to),
                "zeta": float(hop_dict["zeta"]),
                "prob": float(hop_dict["prob"])
            },
                                     event_type="hop")
        else:
            self.tracer.record_event(event_dict={
                "hop_from": int(hop_from),
                "hop_to": int(hop_to),
                "zeta": float(hop_dict["zeta"]),
                "prob": float(hop_dict["prob"])
            },
                                     event_type="frustrated_hop")
