QM/MM NAMD using Turbomole and OpenMM
=====================================

This section briefly describes how to run nonadiabatic molecular dynamics
(NAMD) simulations with mudslide using a QM/MM setup where
Turbomole provides the QM engine and OpenMM provides the MM engine.

.. warning::
   This is a work in progress. Expect it to be incorrect.

This page is assuming you are already somewhat familiar with the
python interface of OpenMM.

Setup Turbomole
---------------
Just like for a regular ``TMModel`` setup, start by preparing a
Turbomole run in a directory and then

.. code-block:: bash

    cd /path/to/turbomole/directory

Prepare PDB files
-----------------
Prepare PDB files for **the QM system** and for the **MM system**.
The PDB file for the QM system will be used to extract
the topology which will be used in OpenMM (the position will be ignored).
The force constants
in the QM region will not be used, but the topology is still
needed to define nonbonded interaction parameters.

Run Simulation
------------------
Next, you will set up a python script that will create
a ``TMModel`` object, an ``OpenMM`` object, and then combine
them to make a ``QMMM`` object.

.. code-block:: python

    import mudslide
    import openmm
    import openmm.app

    qm = mudslide.models.TMModel(states=[0, 1]) # run using ground and first excited states
    pdb_qm = openmm.app.PDBFile('qm.pdb') # read topology from qm.pdb

    pdb_mm = openmm.app.PDBFile('mm.pdb') # read topology from mm.pdb

    # use modeller to combine topologies and positions
    modeller = openmm.app.Modeller(pdb_qm.topology, pdb_qm.positions)
    modeller.add(pdb_mm.topology, pdb_mm.positions)

    ff = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml') # load force field
    system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.NoCutoff,
            constraints=None, rigidWater=False, removeCMMotion=False)
    mm = mudslide.models.OpenMM(modeller, ff, system)

    qmmm = mudslide.models.QMMM(qm, mm)
    momenta = mudslide.math.boltzmann_velocities(qmmm._mass, 300) # generate boltzmann momenta

    log = mudslide.YAMLTrace() # log trajectory using YAML files
    traj = mudslide.TrajectoryCum(qmmm, # run using turbomole model
                                 positions, # start at position from coord
                                 momenta, # use boltzmann momenta
                                 1, # start on 1st excited state
                                 tracer=log, # use yaml log
                                 dt=20, # 20 a.u. time step
                                 max_time=80, # run for 80 a.u.
                                 )
    results = traj.simulate()

Limitations
-----------

* QM region and MM region must not be bonded at all (i.e., MM should probably only be solvent)

Advice
------
Be very wary.
