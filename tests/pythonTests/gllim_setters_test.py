"""
Integration Tests for GLLiM Model "Setter-like" methods.

This script uses pytest to validate the correctness of the GLLiM model's setters. The tests compare the results 
with precomputed reference outputs to ensure reproducibility and correctness.
Note that this test depends on gllim_getters_test.

Tested methods:
    - setParams
    - setParamPi
    - setParamA
    - setParamB
    - setParamC
    - setParamGamma
    - setParamSigma
"""

import pytest
import numpy as np
import xllim


# Model parameters
COVARIANCE_TYPE_LIST = ["full", "diag", "iso"]
K, D, L  = 5, 9, 4

# Data reference
GLLIM_PARAMS_REF = {
    "Pi"    : np.concatenate(([1], np.zeros(K - 1))),
    "A"     : np.ones((K,D,L)) * 3,
    "B"     : np.ones((K,D)) * 2.2,
    "C"     : np.ones((K,L)) * 1.4,
    "Gamma" : {
        "full" : np.tile(np.eye(L), (K, 1, 1)) * 8.6,
        "diag" : np.ones((K,L)) * 8.6,
        "iso"  : np.ones(K) * 8.6,
    },
    "Sigma" : {
        "full" : np.tile(np.eye(D), (K, 1, 1)) * 8.6,
        "diag" : np.ones((K,D)) * 8.6,
        "iso"  : np.ones(K) * 8.6,
    }
}
GLLIM_PARAMS_NAME_LIST = ["Pi", "A", "B", "C", "Gamma", "Sigma"]


@pytest.mark.parametrize("gamma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("sigma_type", COVARIANCE_TYPE_LIST)
def test_setParams(gamma_type, sigma_type):

    # Set up
    gllim_params_ref = xllim.GLLiMParameters(K, D, L, gamma_type, sigma_type)
    gllim_params_ref.Pi = GLLIM_PARAMS_REF["Pi"]
    gllim_params_ref.A = GLLIM_PARAMS_REF["A"]
    gllim_params_ref.B = GLLIM_PARAMS_REF["B"]
    gllim_params_ref.C = GLLIM_PARAMS_REF["C"]
    gllim_params_ref.Gamma = GLLIM_PARAMS_REF["Gamma"][gamma_type]
    gllim_params_ref.Sigma = GLLIM_PARAMS_REF["Sigma"][sigma_type]

    gllim = xllim.GLLiM(K, D, L, gamma_type, sigma_type, 0)
    gllim.setParams(gllim_params_ref)
    gllim_params = gllim.getParams()

    assert np.all(gllim_params.Pi       == GLLIM_PARAMS_REF["Pi"]),                 "Pi"
    assert np.all(gllim_params.A        == GLLIM_PARAMS_REF["A"]),                  "A"
    assert np.all(gllim_params.B        == GLLIM_PARAMS_REF["B"]),                  "B"
    assert np.all(gllim_params.C        == GLLIM_PARAMS_REF["C"]),                  "C"
    assert np.all(gllim_params.Gamma    == GLLIM_PARAMS_REF["Gamma"][gamma_type]),  "Gamma"
    assert np.all(gllim_params.Sigma    == GLLIM_PARAMS_REF["Sigma"][sigma_type]),  "Sigma"


@pytest.mark.parametrize("gamma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("sigma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("gllim_param_name", GLLIM_PARAMS_NAME_LIST)
def test_setParam(gllim_param_name, gamma_type, sigma_type):

    # Set up
    gllim = xllim.GLLiM(K, D, L, gamma_type, sigma_type, 0)

    #setParam
    setParamX = getattr(gllim, "setParam" + gllim_param_name)
    if gllim_param_name == "Gamma":
        setParamX(GLLIM_PARAMS_REF[gllim_param_name][gamma_type])
    elif gllim_param_name == "Sigma":
        setParamX(GLLIM_PARAMS_REF[gllim_param_name][sigma_type])
    else:
        setParamX(GLLIM_PARAMS_REF[gllim_param_name])
    

    # getParam #! (not tested here)
    getParamX = getattr(gllim, "getParam" + gllim_param_name)
    gllim_param = getParamX()

    if gllim_param_name == "Gamma":
        assert np.all(gllim_param == GLLIM_PARAMS_REF[gllim_param_name][gamma_type])
    elif gllim_param_name == "Sigma":
        assert np.all(gllim_param == GLLIM_PARAMS_REF[gllim_param_name][sigma_type])
    else:
        assert np.all(gllim_param == GLLIM_PARAMS_REF[gllim_param_name])
    