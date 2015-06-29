import sys
import argparse
import numpy as np
import tullymodels as tm
import fssh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate potential energy surface scans of two-state models")
    parser.add_argument('-m', '--model', default='simple', choices=('simple', 'dual', 'extended'), help="Tully model to plot")
    parser.add_argument('-r', '--range', default=(-10.0,10.0), nargs=2, type=float, help="range over which to plot PES (default: %(default)s)")
    parser.add_argument('-n', default=100, type=int, help="number of points to plot")
    args = parser.parse_args()

    if args.model == "simple":
        model = tm.TullySimpleAvoidedCrossing()
    elif args.model == "dual":
        model = tm.TullyDualAvoidedCrossing()
    elif args.model == "extended":
        model = tm.TullyExtendedCouplingReflection()
    else:
        raise Exception("Unknown model chosen") # the argument parser should prevent this throw from being possible

    start, end = args.range
    samples = args.n

    xr = np.linspace(start, end, samples)
    last_coeff = np.array([[1.0, 0.0], [0.0, 1.0]])
    print "#%12s %12s %12s %12s %12s %12s" % ("x", "E_0", "E_1", "d_01", "dE_0", "dE_1")
    for x in xr:
        elec = fssh.ElectronicStates(model.V(x), model.dV(x), last_coeff)
        print "%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (x, elec.energies[0], elec.energies[1], elec.compute_derivative_coupling(0,1),
                                                               -elec.compute_force(0), -elec.compute_force(1))
        last_coeff = elec.coeff
