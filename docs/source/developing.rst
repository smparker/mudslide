Developer Guide
====================================

This page describes how to set up a development environment
and run the checks that are required to pass CI.

Setting Up a Development Environment
-------------------------------------
Clone the repository and install in editable mode with the
development dependencies:

.. code-block:: shell

    $ git clone https://github.com/smparker/mudslide.git
    $ cd mudslide
    $ pip install -e ".[dev]"

This installs the package along with ``pytest``, ``mypy``, ``pylint``,
and ``yapf``.

Running the CI Checks Locally
------------------------------
The GitHub Actions CI runs three checks on every push and pull request:
tests, linting, and type checking. You should run all three locally
before pushing to make sure CI will pass.

Tests (pytest)
^^^^^^^^^^^^^^
Run the full test suite with:

.. code-block:: shell

    $ pytest

All tests must pass. To run a single test file:

.. code-block:: shell

    $ pytest test/test_math.py

Linting (pylint)
^^^^^^^^^^^^^^^^^
Run pylint with:

.. code-block:: shell

    $ pylint mudslide/

CI requires a minimum score of **9.5/10**. The project's ``.pylintrc``
file configures allowed variable names, disabled checks, and other
settings. If pylint reports a score below 9.5, you can see individual
messages to understand what needs to be fixed.

You can also check a specific file:

.. code-block:: shell

    $ pylint mudslide/batch.py

Type Checking (mypy)
^^^^^^^^^^^^^^^^^^^^^
Run mypy with:

.. code-block:: shell

    $ mypy mudslide/

This must exit with **zero errors**. The mypy configuration in
``pyproject.toml`` enforces strict settings including
``disallow_untyped_defs`` and ``disallow_incomplete_defs``, so all
functions must have type annotations.

If you are adding new code, make sure to include type annotations on
all function signatures. Common patterns in the codebase:

.. code-block:: python

    import numpy as np
    from numpy.typing import ArrayLike, NDArray

    def compute_energy(x: ArrayLike) -> NDArray:
        ...

Code Formatting (yapf)
^^^^^^^^^^^^^^^^^^^^^^^
The project uses ``yapf`` with a custom style. To format
a file in place:

.. code-block:: shell

    $ yapf -i mudslide/batch.py

To format all source files:

.. code-block:: shell

    $ yapf -i mudslide/*.py

Formatting is not currently enforced in CI, but consistent formatting
is expected for contributions.

Running All Checks
^^^^^^^^^^^^^^^^^^^
To run all three CI checks in sequence:

.. code-block:: shell

    $ pytest && pylint mudslide/ --fail-under=9.5 && mypy mudslide/

If all three commands succeed, your changes should pass CI.
