import mudslide
from mudslide.units import *

import numpy as np

if __name__ == "__main__":
    mudslide.print_header()

    model = mudslide.models.TMModel(states=[0])

    X = model._position
    velocities = mudslide.math.boltzmann_velocities(model.mass,
                                                    temperature=300.0,
                                                    coords=X,
                                                    seed=1234)

    traj = mudslide.AdiabaticMD(model,
                                X,
                                velocities,
                                propagator={
                                    "type": "nhc",
                                    "temperature": 300
                                },
                                tracer="yaml",
                                dt=0.5 * fs,
                                max_steps=100,
                                remove_com_every=1,
                                remove_angular_momentum_every=1)

    results = traj.simulate()

    with open('azobenzene_egylog.txt', 'w') as f:
        results.print_egylog(file=f)

    mudslide.io.write_trajectory_xyz(model, results,
                                     'azobenzene_trajectory.xyz')
