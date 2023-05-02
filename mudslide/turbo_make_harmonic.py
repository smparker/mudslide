# -*- coding: utf-8 -*-
"""
Extract harmonic parameters from a vibrational analysis
"""

import sys
import argparse

import numpy as np

from .turbomole_model import TurboControl, turbomole_is_installed
from .harmonic_model import HarmonicModel


def main(argv=None, file=sys.stdout):
    parser = argparse.ArgumentParser(description="Generate a harmonic model from a vibrational analysis")
    parser.add_argument("-c", "--control", default="control", help="Control file")
    parser.add_argument("-o", "--output", default="harmonic.json", help="Output file")

    args = parser.parse_args(argv)

    if not turbomole_is_installed():
        raise RuntimeError("Turbomole is not available")

    print(f"Reading Turbomole control file from {args.control}", file=file)
    print(file=file)

    # get Turbomole control file
    turbo = TurboControl(args.control)

    # read coords
    symbols, coords = turbo.read_coords()
    masses = turbo.get_masses(symbols)

    print("Reference geometry:", file=file)
    print(f"{'el':>3s} {'x':>20s} {'y':>20s} {'z':>20s} {'mass':>20s}", file=file)
    print("-" * 100, file=file)

    ms = masses.reshape(-1, 3)[:, 0]
    for symbol, coord, mass in zip(symbols, coords.reshape(-1, 3), ms):
        print(f"{symbol:3s} {coord[0]: 20.16g} {coord[1]: 20.16g} {coord[2]: 20.16g} {mass: 20.16g})", file=file)
    print(file=file)

    # read Hessian
    hessian = turbo.read_hessian()
    print("Hessian loaded with eigenvalues", file=file)
    print(np.linalg.eigvalsh(hessian), file=file)
    print(file=file)

    harmonic = HarmonicModel(coords, 0.0, hessian, masses)

    print(f"Writing harmonic model to {args.output}", file=file)
    harmonic.to_file(args.output)
