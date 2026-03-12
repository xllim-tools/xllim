"""
Integration Tests for GLLiM Model getInverse method.

This script uses pytest to validate the correctness of the GLLiM model's getInverse() method. The tests compare the results 
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

Reference Data:
    Precomputed reference outputs are stored in `../dataRef/gllim_params_star_ref/` for validation.
"""

import pytest
import numpy as np
import pickle
import xllim
import os


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

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_REF_DIR = os.path.join(BASE_PATH, "..", "dataRef")


@pytest.mark.parametrize("gamma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("sigma_type", COVARIANCE_TYPE_LIST)
def test_getInverse(gamma_type, sigma_type):

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

    gllim_params_star = gllim.getInverse()

    file_name = "gllim_params_star_ref_{}_{}.file".format(gamma_type, sigma_type)
    file_path = os.path.join(DATA_REF_DIR, "gllim_params_star_ref", file_name)
    # # ! Only run once to generate ref
    # with open(file_path, "wb") as f:
    #     pickle.dump(gllim_params_star, f)
    #     f.close()

    with open(file_path, "rb") as f:
        gllim_params_star_ref = pickle.load(f)
        f.close()

    assert np.allclose(gllim_params_star.Pi   , gllim_params_star_ref.Pi),      "Pi"
    assert np.allclose(gllim_params_star.A    , gllim_params_star_ref.A),       "A"
    assert np.allclose(gllim_params_star.B    , gllim_params_star_ref.B),       "B"
    assert np.allclose(gllim_params_star.C    , gllim_params_star_ref.C),       "C"
    assert np.allclose(gllim_params_star.Gamma, gllim_params_star_ref.Gamma),   "Gamma"
    assert np.allclose(gllim_params_star.Sigma, gllim_params_star_ref.Sigma),   "Sigma"
