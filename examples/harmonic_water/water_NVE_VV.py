import mudslide
from mudslide.units import *

import json
import numpy as np

mudslide.print_header()

water_model = json.load(open('harmonic_water.json'))

model = mudslide.models.HarmonicModel.from_dict(water_model)

x = np.array(model.x0)
velocities = mudslide.math.boltzmann_velocities(model.mass,
                                                temperature=500.0,
                                                coords=x,
                                                seed=1234)

traj = mudslide.AdiabaticMD(model,
                            x,
                            velocities,
                            dt=0.5 * fs,
                            max_steps=500,
                            remove_com_every=1,
                            remove_angular_momentum_every=1)

results = traj.simulate()

with open('harmonic_water_NVE_VV_egy.log', 'w') as f:
    results.print_egylog(file=f)

mudslide.io.write_trajectory_xyz(model, results, 'harmonic_water.xyz')
