import mudslide
import numpy as np

model = mudslide.models.TMModel(states=[0])

X = model._position
velocities = mudslide.math.boltzmann_velocities(model.mass, temperature=500.0, seed=1234)
velocities = mudslide.util.remove_angular_momentum(velocities.reshape((-1,3)), model.mass.reshape((-1,3))[:,0], X.reshape((-1,3))).flatten()

traj = mudslide.AdiabaticMD(model, X, velocities,
                            propagator={ "type": "nhc", "temperature": 300},
                            dt=0.5*mudslide.fs_to_au,
                            max_steps=100,
                            remove_com_every=1,
                            remove_angular_momentum_every=1)

results = traj.simulate()

results.print()

mudslide.io.write_trajectory_xyz(model, results, 'azobenzene_trajectory.xyz')
