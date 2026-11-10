"""
Tests for ExternalPythonModel, the C++ wrapper of functional models written in Python.

Each wrapped model is compared with the same Python class called directly (the reference).

Regression tests for issue #67: toPhysic / fromPhysic returned values read from freed memory (zeros on macOS,
correct by chance on Linux when L <= 16, garbage when L > 16) and ignored the value returned by the Python method.

Test models are in `../dataRef/externalPythonModels` (`conversionmodels.py` contains the models specific to these tests).
"""

import importlib
import os
import sys

import numpy as np
import pytest
import xllim

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "..", "dataRef", "externalPythonModels")

# (class name, module name) of the external models
MODELS = [
    ("RPVModel", "rpvmodel"),
    ("ShkuratovModel5p", "ShkuratovModel5pPython"),
    ("AffineModelL20", "conversionmodels"),  # L = 20 > 16: vectors stored on the heap by Armadillo
    ("ReturnStyleModel", "conversionmodels"),  # toPhysic / fromPhysic return a new array
]

RTOL = 1e-12


def load_reference(class_name, module_name):
    """Instantiates the Python class directly, without the C++ wrapper."""
    if MODELS_DIR not in sys.path:
        sys.path.append(MODELS_DIR)
    return getattr(importlib.import_module(module_name), class_name)()


def random_inputs(L, n=10, seed=0):
    """n vectors of dimension L in the mathematical space."""
    return np.random.default_rng(seed).uniform(0.05, 0.95, size=(n, L))


@pytest.fixture(params=MODELS, ids=[class_name for class_name, _ in MODELS])
def models(request):
    """Returns (wrapped model, reference Python model)."""
    class_name, module_name = request.param
    wrapped = xllim.ExternalPythonModel(class_name, module_name, MODELS_DIR)
    return wrapped, load_reference(class_name, module_name)


def test_dimensions(models):
    wrapped, reference = models
    assert wrapped.getDimensionX() == reference.getDimensionX()
    assert wrapped.getDimensionY() == reference.getDimensionY()


@pytest.mark.parametrize("method", ["toPhysic", "fromPhysic"])
def test_conversion_matches_python(models, method):
    wrapped, reference = models
    for x in random_inputs(wrapped.getDimensionX()):
        expected = getattr(reference, method)(x.copy())
        np.testing.assert_allclose(getattr(wrapped, method)(x), expected, rtol=RTOL)


def test_conversion_round_trip(models):
    wrapped, _ = models
    for x in random_inputs(wrapped.getDimensionX()):
        np.testing.assert_allclose(wrapped.fromPhysic(wrapped.toPhysic(x)), x, rtol=1e-10)


@pytest.mark.parametrize("method", ["toPhysic", "fromPhysic"])
def test_conversion_does_not_modify_input(models, method):
    wrapped, _ = models
    x = random_inputs(wrapped.getDimensionX(), n=1)[0]
    x_before = x.copy()
    getattr(wrapped, method)(x)
    np.testing.assert_array_equal(x, x_before)


def test_F_matches_python(models):
    wrapped, reference = models
    for x in random_inputs(wrapped.getDimensionX()):
        np.testing.assert_allclose(wrapped.F(x), reference.F(x.copy()), rtol=RTOL)


def test_rpv_known_values():
    """Calls reported in issue #67, expected values computed by hand from rpvmodel.py."""
    model = xllim.ExternalPythonModel("RPVModel", "rpvmodel", MODELS_DIR)
    np.testing.assert_allclose(model.toPhysic(np.ones(3)), [1.0, 1.2, 1.0], rtol=RTOL)
    np.testing.assert_allclose(model.fromPhysic(np.zeros(3)), [0.0, 0.0, 0.5], rtol=RTOL)


def test_invalid_model_raises():
    model = xllim.ExternalPythonModel("InvalidModel", "conversionmodels", MODELS_DIR)
    with pytest.raises(RuntimeError, match=r"toPhysic\(\) returned 2 values, 3 expected"):
        model.toPhysic(np.ones(3))
    with pytest.raises(RuntimeError, match=r"fromPhysic\(\) returned 4 values, 3 expected"):
        model.fromPhysic(np.ones(3))
    with pytest.raises(RuntimeError, match=r"F\(\) returned None"):
        model.F(np.ones(3))
