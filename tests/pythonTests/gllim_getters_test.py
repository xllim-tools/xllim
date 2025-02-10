"""
Integration Tests for GLLiM Model "Getter-like" methods.

This script uses pytest to validate the correctness of the GLLiM model's getters. The tests compare the results 
with precomputed reference outputs to ensure reproducibility and correctness.

Tested methods:
    - getDimensions
    - getConstraints
    - getParams
    - getParamPi
    - getParamA
    - getParamB
    - getParamC
    - getParamGamma
    - getParamSigma
"""

import pytest
import numpy as np
import xllim


# Model parameters
COVARIANCE_TYPE_LIST = ["full", "diag", "iso"]
K, D, L  = 5, 9, 4

# Data reference
GLLIM_PARAMS_REF = {
    "Pi": np.ones(K) / K,
    "A": np.zeros((K, D, L)),
    "B": np.zeros((K, D)),
    "C": np.zeros((K, L)),
    "Gamma": {
        "full": np.tile(np.eye(L), (K, 1, 1)),
        "diag": np.ones((K, L)),
        "iso": np.ones(K),
    },
    "Sigma": {
        "full": np.tile(np.eye(D), (K, 1, 1)),
        "diag": np.ones((K, D)),
        "iso": np.ones(K),
    },
}
GLLIM_PARAMS_NAME_LIST = ["Pi", "A", "B", "C", "Gamma", "Sigma"]


@pytest.mark.parametrize("gamma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("sigma_type", COVARIANCE_TYPE_LIST)
def test_getDimensions(gamma_type, sigma_type):
    gllim = xllim.GLLiM(K, D, L, gamma_type, sigma_type, 0)
    gllim_dimensions_ref = f"GLLiM dimensions are (L={L}, D={D}, K={K})"
    assert gllim.getDimensions() == gllim_dimensions_ref


@pytest.mark.parametrize("gamma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("sigma_type", COVARIANCE_TYPE_LIST)
def test_getConstraints(gamma_type, sigma_type):
    gllim = xllim.GLLiM(K, D, L, gamma_type, sigma_type, 0)
    gllim_constraints_ref = f"GLLiM constraints are gamma_type = '{gamma_type}', sigma_type = '{sigma_type}'"
    assert gllim.getConstraints() == gllim_constraints_ref


@pytest.mark.parametrize("gamma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("sigma_type", COVARIANCE_TYPE_LIST)
def test_getParams(gamma_type, sigma_type):
    gllim = xllim.GLLiM(K, D, L, gamma_type, sigma_type, 0)
    gllim_params = gllim.getParams()

    assert np.all(gllim_params.Pi       == GLLIM_PARAMS_REF["Pi"]),                "Pi"
    assert np.all(gllim_params.A        == GLLIM_PARAMS_REF["A"]),                 "A"
    assert np.all(gllim_params.B        == GLLIM_PARAMS_REF["B"]),                 "B"
    assert np.all(gllim_params.C        == GLLIM_PARAMS_REF["C"]),                 "C"
    assert np.all(gllim_params.Gamma    == GLLIM_PARAMS_REF["Gamma"][gamma_type]), "Gamma"
    assert np.all(gllim_params.Sigma    == GLLIM_PARAMS_REF["Sigma"][sigma_type]), "Sigma"


@pytest.mark.parametrize("gamma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("sigma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("gllim_param_name", GLLIM_PARAMS_NAME_LIST)
def test_getParam(gllim_param_name, gamma_type, sigma_type):
    gllim = xllim.GLLiM(K, D, L, gamma_type, sigma_type, 0)
    getParamX = getattr(gllim, "getParam" + gllim_param_name)
    gllim_param = getParamX()
    if gllim_param_name == "Gamma":
        assert np.all(gllim_param == GLLIM_PARAMS_REF[gllim_param_name][gamma_type])
    elif gllim_param_name == "Sigma":
        assert np.all(gllim_param == GLLIM_PARAMS_REF[gllim_param_name][sigma_type])
    else:
        assert np.all(gllim_param == GLLIM_PARAMS_REF[gllim_param_name])
