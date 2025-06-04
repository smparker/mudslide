import json
import mudslide
import numpy as np

water_model = json.load(open('harmonic_water.json'))

model = mudslide.models.HarmonicModel.from_dict(water_model)

x = np.array(model.x0)
velocities = mudslide.math.boltzmann_velocities(model.mass, temperature=10000.0, seed=1234)
velocities = mudslide.util.remove_angular_momentum(velocities.reshape((-1,3)), model.mass.reshape((-1,3))[:,0], x.reshape((-1,3))).flatten()

traj = mudslide.AdiabaticMD(model, x, velocities, dt=40, max_steps=500,
                            remove_com_every=1,
                            remove_angular_momentum_every=1)

results = traj.simulate()

results.print()

mudslide.io.write_trajectory_xyz(model, results, 'harmonic_water.xyz')
