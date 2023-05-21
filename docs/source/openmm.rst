Molecular Mechanics using OpenMM
================================

This section briefly describes how to run molecular mechanics
simulations with mudslide using OpenMM as the engine.

.. warning::
   This interface only exists to support QM/MM simulations.
   If you want to run normal molecular mechanics simulations,
   then it would be much smarter to use OpenMM directly.

OpenMM Required
---------------
OpenMM needs to be available for this interface to work.
Since OpenMM is not distributed on PyPI, you will need to
install it yourself. The easiest way is through conda.

Load OpenMM class
------------------
Next, you will set up a python script that will create an OpenMM model
and run the simulation. An example script is shown below:

.. code-block:: python

    import mudslide
    import openmm
    import openmm.app

    pdb = openmm.app.PDBFile("h2o5.pdb")
    ff = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = ff.createSystem(pdb.topology,
                             nonbondedMethod=openmm.app.NoCutoff,
                             constraints=None,
                             rigidWater=False)
    mm = mudslide.models.OpenMM(pdb, ff, system)
    positions = mm._position # read position from the coord file
    momenta = mudslide.math.boltzmann_velocities(mm._mass, 300) * mm._mass

    log = mudslide.YAMLTrace() # log trajectory using YAML files
    traj = mudslide.AdiabaticMD(model, # run using turbomole model
                                positions, # start at position from coord
                                momenta, # use boltzmann momenta
                                tracer=log, # use yaml log
                                dt=20, # 20 a.u. time step
                                max_time=80, # run for 80 a.u.
                               )
    results = traj.simulate()


Advice
------

Only use this class for debugging QM/MM calculations.
