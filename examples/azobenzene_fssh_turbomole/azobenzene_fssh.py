import mudslide
from mudslide.units import *

import numpy as np

if __name__ == "__main__":
    mudslide.print_header()

    model = mudslide.models.TMModel(states=[0, 1])

    X = model._position
    velocities = mudslide.math.boltzmann_velocities(model.mass,
                                                    temperature=200.0,
                                                    coords=X,
                                                    seed=1234)

    traj = mudslide.SurfaceHoppingMD(model,
                                     X,
                                     velocities,
                                     1,
                                     tracer="yaml",
                                     dt=fs,
                                     max_steps=10)

    results = traj.simulate()

    with open('azobenzene_egylog.txt', 'w') as f:
        results.print_egylog(file=f)

    mudslide.io.write_trajectory_xyz(model, results,
                                     'azobenzene_trajectory.xyz')
