# -*- coding: utf-8 -*-
"""Tests for the harmonic model"""

import sys
import json

import numpy as np
import pytest
import mudslide

water_json = """
{"x0": [0.0, 0.0, -0.12178983933899, 1.41713420892173, 0.0, 0.96657854674257, -1.41713420892173, 0.0, 0.96657854674257], "E0": 0.0, "H0": [[0.783125763, 0.0, 5e-10, -0.3915628813, 0.0, -0.3007228667, -0.3915628817, 0.0, 0.3007228662], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5e-10, 0.0, 0.482147457, -0.245482284, 0.0, -0.2410737287, 0.2454822835, 0.0, -0.2410737283], [-0.3915628813, 0.0, -0.245482284, 0.4189939059, 0.0, 0.2731025751, -0.0274310246, 0.0, -0.0276202911], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.3007228667, 0.0, -0.2410737287, 0.2731025751, 0.0, 0.2360154334, 0.0276202915, 0.0, 0.0050582953], [-0.3915628817, 0.0, 0.2454822835, -0.0274310246, 0.0, 0.0276202915, 0.4189939063, 0.0, -0.273102575], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3007228662, 0.0, -0.2410737283, -0.0276202911, 0.0, 0.0050582953, -0.273102575, 0.0, 0.236015433]], "mass": [29156.945697766205, 29156.945697766205, 29156.945697766205, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108, 1837.1526473562108], "atom_types": ["O", "H", "H"]}"""

water_model = json.loads(water_json)


def test_load_water():
    """Load water model from json"""
    model = mudslide.models.HarmonicModel.from_dict(water_model)

    assert model.E0 == 0.0
    assert model.x0.shape == (9,)
    assert model.mass.shape == (9,)
    assert model.H0.shape == (9, 9)
    assert model.dimensionality == (3, 3)
    assert model.nparticles == 3
    assert model.ndims == 3


def test_harmonic_energy_force():
    """Test reference values for energy and force from harmonic model"""
    model = mudslide.models.HarmonicModel.from_dict(water_model)
    x = np.copy(model.x0)

    x[2] = -0.10
    model.compute(x)

    assert np.allclose(model.energies, 0.00011446)

    force_ref = np.array([
        -1.08949197e-11, -0.00000000e+00, -1.05059156e-02, 5.34901953e-03,
        -0.00000000e+00, 5.25295782e-03, -5.34901952e-03, -0.00000000e+00,
        5.25295781e-03
    ])
    assert np.allclose(model.force(0), force_ref)


def test_from_file_json(tmp_path):
    """Load and round-trip through JSON file"""
    filepath = str(tmp_path / "model.json")
    with open(filepath, "w") as f:
        json.dump(water_model, f)

    model = mudslide.models.HarmonicModel.from_file(filepath)
    assert model.E0 == 0.0
    assert model.ndof == 9
    assert model.nparticles == 3


def test_from_file_yaml(tmp_path):
    """Load from YAML file"""
    import yaml
    filepath = str(tmp_path / "model.yaml")
    with open(filepath, "w") as f:
        yaml.dump(water_model, f)

    model = mudslide.models.HarmonicModel.from_file(filepath)
    assert model.E0 == 0.0
    assert model.ndof == 9


def test_from_file_unknown_format(tmp_path):
    """Error on unknown file format"""
    filepath = str(tmp_path / "model.txt")
    with open(filepath, "w") as f:
        f.write("hello")

    with pytest.raises(ValueError, match="Unknown file format"):
        mudslide.models.HarmonicModel.from_file(filepath)


def test_to_file_json(tmp_path):
    """Write to JSON and read back"""
    model = mudslide.models.HarmonicModel.from_dict(water_model)
    filepath = str(tmp_path / "out.json")
    model.to_file(filepath)

    loaded = mudslide.models.HarmonicModel.from_file(filepath)
    assert np.allclose(loaded.x0, model.x0)
    assert np.isclose(loaded.E0, model.E0)
    assert np.allclose(loaded.H0, model.H0)


def test_to_file_yaml(tmp_path):
    """Write to YAML and read back"""
    model = mudslide.models.HarmonicModel.from_dict(water_model)
    filepath = str(tmp_path / "out.yaml")
    model.to_file(filepath)

    loaded = mudslide.models.HarmonicModel.from_file(filepath)
    assert np.allclose(loaded.x0, model.x0)
    assert np.isclose(loaded.E0, model.E0)


def test_to_file_unknown_format(tmp_path):
    """Error on unknown output format"""
    model = mudslide.models.HarmonicModel.from_dict(water_model)
    filepath = str(tmp_path / "out.txt")

    with pytest.raises(ValueError, match="Unknown file format"):
        model.to_file(filepath)


def test_to_file_with_atom_types(tmp_path):
    """Round-trip preserves atom_types"""
    model = mudslide.models.HarmonicModel.from_dict(water_model)
    assert model.atom_types == ["O", "H", "H"]

    filepath = str(tmp_path / "out.json")
    model.to_file(filepath)

    loaded = mudslide.models.HarmonicModel.from_file(filepath)
    assert loaded.atom_types == ["O", "H", "H"]


def test_to_file_without_atom_types(tmp_path):
    """Round-trip works without atom_types"""
    d = dict(water_model)
    del d["atom_types"]
    model = mudslide.models.HarmonicModel.from_dict(d)
    assert model.atom_types is None

    filepath = str(tmp_path / "out.json")
    model.to_file(filepath)

    loaded = mudslide.models.HarmonicModel.from_file(filepath)
    assert loaded.atom_types is None
