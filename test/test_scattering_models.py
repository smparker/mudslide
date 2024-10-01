#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit testing for OpenMM functionalities"""

import numpy as np
import os
import shutil
import unittest
import pytest

import mudslide
import yaml

testdir = os.path.dirname(__file__)
_refdir = os.path.join(testdir, "ref", "scattering_models")

def get_ref_data(model_name, nstates, ndim):
    H_ref = np.loadtxt(os.path.join(_refdir, model_name, "H.txt"))
    force_ref = np.loadtxt(os.path.join(_refdir, model_name, "force.txt"))
    dc_ref = np.loadtxt(os.path.join(_refdir, model_name, "dc.txt")).reshape([nstates, nstates, ndim])

    return H_ref, force_ref, dc_ref

def test_subotnik_model_w():
    model = mudslide.models.SubotnikModelW()

    assert model.ndim() == 1
    assert model.nstates() == 8

    X = np.zeros([1])
    X[:] = -0.5

    model.compute(X, reference=np.eye(model.nstates()))

    H = model.hamiltonian()
    force = model.force()
    dc = model._derivative_coupling

    H_ref, force_ref, dc_ref = get_ref_data('model_w', model.nstates(), model.ndim())

    assert np.allclose(H, H_ref)
    assert np.allclose(force, force_ref)
    assert np.allclose(dc, dc_ref)

def test_subotnik_model_z():
    model = mudslide.models.SubotnikModelZ()

    assert model.ndim() == 1
    assert model.nstates() == 8

    X = np.zeros([1])
    X[:] = -0.5

    model.compute(X, reference=np.eye(model.nstates()))

    H = model.hamiltonian()
    force = model.force()
    dc = model._derivative_coupling

    H_ref, force_ref, dc_ref = get_ref_data('model_z', model.nstates(), model.ndim())

    assert np.allclose(H, H_ref)
    assert np.allclose(force, force_ref)
    assert np.allclose(dc, dc_ref)
