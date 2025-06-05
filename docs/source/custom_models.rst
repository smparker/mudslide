Using Custom Models in Mudslide
===============================

This section briefly describes how to write your own model
class so you can run simulations on custom potential energy surfaces.

General Models
--------------

To use a custom model, you need to create a class that derives
from ``mudslide.models.Electronics_`` and that implements the following:

* ``ndim_``: member variable with the number of classical degrees of freedom
* ``nstates_``: member variable with the number of electronic states
* ``mass``: member variable that is an ndarray of shape ``(ndim)`` with the masses of the
  classical degrees of freedom
* ``compute(self, X, couplings: Any=None, gradients: Any=None, reference: Any=None) -> None``
  a member function that takes an ndarray of shape `(ndim_)` and computes all of the
  electronic properties at position `X`. All the required results should be stored
  in the class. More details below.
* ``clone() -> Electronics_``: a member function that returns a copy of the class,
  including whatever resources it should need.

The compute function
++++++++++++++++++++

The ``compute()`` function is the key part of the class in which all the
electronic properties are computed. The function needs to compute and
store the following quantities:

* ``self._hamiltonian``: an ndarray of shape ``(nstates_, nstates_)`` with the Hamiltonian
  matrix at position ``X``.
* ``self._force``: an ndarray of shape ``(nstates_, ndim_)`` with the forces on each
  electronic state at position ``X``.
* ``self._forces_available``: an ndarray of shape ``(nstates`)` and type ``bool`` that indicates which gradients
  were computed.
* ``self._derivative_coupling``: an ndarray of shape ``(nstates_, nstates_, ndim_)`` with
  the derivative coupling matrix at position ``X``.
* ``self._derivative_couplings_available``: an ndarray of shape ``(nstates_, nstates_)`` and type ``bool``
  that indicates which derivative couplings were computed.

The clone function
++++++++++++++++++

The clone function is only used for even sampling algorithms, in which new
trajectories are spawned at hopping points. When a new trajectory is spawned,
the clone function is in charge of duplicating the electronic model and
any resources the electronic model needs to operate. For example,
each turbomole calculation requires its own directory, so the clone function
should create a unique directory for the model calculations in the spawned trajectory.

By default, the ``clone()`` function returns a deep copy of the class.
For models that don't need unique disk space or other shared resources,
a deep copy will suffice. For models that do need unique resources, the
``clone()`` function should be overridden to create a new instance of the
class with the appropriate resources.

Diabatic Models
---------------

For diabatic models there is a convenience class, ``mudslide.models.DiabaticModel_``, to streamline
making a working mudslide model. To create a diabatic model for use with mudslide,
your model class should inherit from ``DiabaticModel_`` and implement the following:

* ``ndim_``: member variable with the number of classical degrees of freedom
* ``nstates_``: member variable with the number of electronic states
* ``mass``: member variable that is an ndarray of shape ``(ndim_)`` with the masses of the
  classical degrees of freedom
* ``V(self, X)``: a member function that takes an ndarray of shape ``(ndim_)`` and returns
  the Hamiltonian matrix at position ``X``. The Hamiltonian matrix should be of shape
  ``(nstates_, nstates_)``.
* ``dV(self, X)``: a member function that takes an ndarray of shape ``(ndim_)`` and returns
  the gradient Hamiltonian matrix at position ``X``. The gradient Hamiltonian matrix should
  be of shape ``(ndim_, nstates_, nstates_)``.
