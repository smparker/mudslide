import numpy as np
import mudslide
import sys
import re
import queue
from mudslide.tracer import YAMLTrace
from mudslide.even_sampling import SpawnStack
from numpy.random import default_rng
from mudslide.batch import TrajGenConst, TrajGenNormal, BatchedTraj
from mudslide.even_sampling import EvenSamplingTrajectory
from mudslide.tracer import InMemoryTrace, YAMLTrace, TraceManager

def run():
    tm_model = mudslide.models.TMModel(states=[0, 1, 2, 3], run_turbomole_dir="run_turbomole")
    mom = [
        5.583286976987380000,
        -2.713959745507320000,
        0.392059702162967000,
        -0.832994241764031000,
        -0.600752326053757000,
        -0.384006560250834000,
        -1.656414687719690000,
        1.062437820195600000,
        -1.786171104341720000,
        -2.969087779972610000,
        1.161804203506510000,
        -0.785009852486148000,
        2.145175145340160000,
        0.594918215579156000,
        1.075977514428970000,
        -2.269965412856570000,
        0.495551832268249000,
        1.487150300486560000,
    ]

    positions = tm_model.X
    mass = tm_model.mass
    velocities = np.array(mom) / mass
    q = queue.Queue()
    dt = 20
    max_time=41
    t0 = 1
    sample_stack = SpawnStack.from_quadrature(nsamples=[2, 2, 2])
    sample_stack.sample_stack[0]["zeta"]=0.003
    samples = 1
    nprocs = 1
    trace_type = YAMLTrace
    trace_options = {}
    electronic_integration = 'exp'
    trace_options["location"] = ""
    every = 1
    model=tm_model

    traj_gen = TrajGenConst(positions, velocities, 3, dt)

    fssh = mudslide.BatchedTraj(model,
                   traj_gen,
                   trajectory_type=EvenSamplingTrajectory,
                   mom=mom,
                   positions=positions,
                   samples=samples,
                   max_time = max_time,
                   nprocs=nprocs,
                   dt=dt,
                   t0=t0,
                   tracemanager=TraceManager(TraceType=trace_type, trace_kwargs=trace_options),
                   trace_every=every,
                   spawn_stack=sample_stack,
                   electronic_integration=electronic_integration)#,
#                   hopping_probability=hopping_probability) # ,
    results = fssh.compute()
    outcomes = results.outcomes

    with open("TMModel_sh_testing.dat", "w") as file:
        file.write("{}".format(results[0]))

def main():
    run()

if __name__ == "__main__":
    main()
