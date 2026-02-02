NAMD using Turbomole
====================

This section briefly describes how to run nonadiabatic molecular dynamics
(NAMD) with mudslide using Turbomole as a QM engine.

.. warning::
   This is a work in progress. Expect some kinks.

Setup Turbomole
---------------
The first step will be to setup a directory that contains all the
input information you will need for a turbomole calculation.
This means run define, TmoleX, or whatever else you would normally
run to set up the calculation. Please see the Turbomole documentation
for more information on how to do this. When the directory is set up,
change to it:

.. code-block:: bash

    cd /path/to/turbomole/directory


Load TMModel class
------------------
Next, you will set up a python script that will create a TMModel
and run the simulation. An example script is shown below:

.. code-block:: python

    import mudslide

    model = mudslide.models.TMModel(states=[0, 1]) # run using ground and first excited states
    positions = model._position # read position from the coord file
    momenta = mudslide.math.boltzmann_velocities(model._mass, 300) * model._mass

    log = mudslide.YAMLTrace() # log trajectory using YAML files
    traj = mudslide.SurfaceHoppingMD(model, # run using turbomole model
                                 positions, # start at position from coord
                                 momenta, # use boltzmann momenta
                                 1, # start on 1st excited state
                                 tracer=log, # use yaml log
                                 dt=20, # 20 a.u. time step
                                 max_time=80, # run for 80 a.u.
                                 hopping_method="cumulative", # use cumulative hopping
                                 )
    results = traj.simulate()


Running Turbomole via Docker or Singularity
--------------------------------------------
If Turbomole is not installed locally, you can run it through a container
using a **command prefix**. The command prefix is prepended to every
Turbomole command that mudslide executes. There are three ways to set it,
listed from highest to lowest priority:

1. **Pass it directly** to ``TMModel``:

   .. code-block:: python

       model = mudslide.models.TMModel(
           states=[0, 1],
           command_prefix="docker run --rm -i --platform linux/amd64 -v $(pwd):$(pwd) -w $(pwd) smparker/tm:latest"
       )

2. **Set an environment variable** before running your script:

   .. code-block:: bash

       export MUDSLIDE_TURBOMOLE_PREFIX="docker run --rm -i --platform linux/amd64 -v $(pwd):$(pwd) -w $(pwd) smparker/tm:latest"

3. **Add it to the config file** at ``~/.config/mudslide/config.yaml``
   (or ``$XDG_CONFIG_HOME/mudslide/config.yaml``):

   .. code-block:: yaml

       turbomole:
         command_prefix: "docker run --rm -i --platform linux/amd64 -v $(pwd):$(pwd) -w $(pwd) smparker/tm:latest"

   With the config file in place, no extra arguments or environment variables
   are needed — ``TMModel`` will pick up the prefix automatically:

   .. code-block:: python

       model = mudslide.models.TMModel(states=[0, 1])

When more than one source is configured, the explicit ``command_prefix``
argument wins over the environment variable, which wins over the config file.


Advice
------

*  Make sure your control file specifies all the options you need for dynamics. For example:

   * For NAMD you will need to have

     * ``$soes`` data group with information on the excitations.
     * ``$nacme`` (for just states 0,1) or ``$nacme pseudo`` (for state-to-state)
     * weight derivatives

   * And it is recommended to have

     * ``$phaser``
     * ``$do_etf``
