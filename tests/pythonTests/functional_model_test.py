"""
Integration Tests for Functional Model Importance Sampling method.

This script uses pytest to validate the correctness of the importanceSampling step. The tests compare the 
results with precomputed reference outputs to ensure reproducibility and correctness.

Modules:
    - pytest: For organizing and running test cases.
    - numpy: For numerical operations.
    - pickle: For serializing and deserializing reference data.
    - json: For loading photometric geometries.
    - xllim: Custom library containing GLLiM implementations.

Reference Data:
    Precomputed reference outputs are stored in `../dataRef` for validation.
"""

import pytest
import numpy as np
import pickle
import json
import xllim


# Fix the random seed for reproducibility across all tests
SEED = 12345

def set_up_models():
    """
    Sets up a list of physical models for testing.

    Returns:
        list[dict]: A list of dictionaries, each containing the model name and an instance of the corresponding xllim model.
    """

    physical_models = []

    # Add the basic Test model
    physical_models.append({"name": "TestModel", "xllim": xllim.TestModel()})

    # Load geometries for JSC1 from JSON
    with open("../dataRef/JSC1_BRDF.json", "r") as f:
        data = json.load(f)
    geometries_JSC1 = {
        "name": "JSC1",
        "data": np.array(data["JSC1_analogue"]["geometries"], dtype=float),
    }

    # Load geometries for Mukundpura from JSON
    with open("../dataRef/mukundpura_bloc_poudre_BRDF.json", "r") as f:
        data = json.load(f)
    geometries_muk = {
        "name": "muk",
        "data": np.array(data["Mukundpura"]["geometries"], dtype=float),
    }

    # Generate models for each geometry
    for geometries in [geometries_JSC1, geometries_muk]:

        # Add Hapke models (3p, 4p, 6p)
        variant = "2002"
        theta_bar_scaling = 30.0
        b0 = 1
        h = 0
        for adapter, adapter_num in [("three", "3"), ("four", "4"), ("six", "6")]:
            physical_models.append(
                {
                    "name": "Hapke_" + adapter_num + "p_" + geometries["name"],
                    "xllim": xllim.HapkeModel(
                        geometries["data"], variant, adapter, theta_bar_scaling, b0, h
                    ),
                }
            )

        # Add Shkuratov models (3p, 5p)
        scalingCoeffs = [1.0, 1.5, 1.5, 1.5, 1.5]
        offset = [0, 0, 0.2, 0, 0]
        for variant in ["3p", "5p"]:
            physical_models.append(
                {
                    "name": "Shkuratov_" + variant + "_" + geometries["name"],
                    "xllim": xllim.ShkuratovModel(
                        geometries["data"], variant, scalingCoeffs, offset
                    ),
                }
            )

    return physical_models


# ! #####################  TEST : importanceSampling()  #######################

K, N_obs = 5, 10
N_0, B, J = 60, 8, 5
physical_models = set_up_models()


@pytest.mark.parametrize("physical_model", physical_models, ids=lambda pm: pm["name"])
def test_importanceSampling(physical_model):
    # physical_models = set_up_models()
    # for physical_model in physical_models:
    L = physical_model["xllim"].getDimensionX()
    D = physical_model["xllim"].getDimensionY()
    covariance = np.ones(D) * 1e-5

    np.random.seed(SEED)
    y_obs = np.random.rand(N_obs, D)

    proposition_gmms = []
    for n in range(N_obs):
        weight = np.ones(K) * 1 / K
        np.random.seed(SEED)
        mean = np.random.rand(L, K)
        cube = np.ones((L, L, K)) * 0.01
        np.random.seed(SEED)
        cube += np.random.rand(L, L, K) * 0.1
        for k in range(cube.shape[2]):
            cube[:, :, k] += np.eye(L) * 0.1
            cube[:, :, k] = np.dot(cube[:, :, k], cube[:, :, k].T) * 0.001

        proposition_gmms.append((weight.T, mean, cube))
    y_err = y_obs * 0.001

    is_results = physical_model["xllim"].importanceSampling(
        proposition_gmms, y_obs, y_err, covariance, N_0, B, J, 0, SEED
    )

    # # ! Only run once to generate ref
    # with open("../dataRef/is_results_ref/is_results_ref_{}.file".format(physical_model["name"]), "wb") as f:
    #     pickle.dump(is_results, f)
    #     f.close()

    with open("../dataRef/is_results_ref/is_results_ref_{}.file".format(physical_model["name"]), "rb") as f:
        is_results_ref = pickle.load(f)
        f.close()

    # compare results
    error_msg = "importanceSampling" + " > " + physical_model["name"]
    assert np.allclose(is_results.predictions, is_results_ref.predictions), error_msg + " > " + "predictions"
    assert np.allclose(is_results.predictions_variance, is_results_ref.predictions_variance), error_msg + " > " + "predictions_variance"
    assert np.allclose(is_results.nb_effective_sample, is_results_ref.nb_effective_sample), error_msg + " > " + "nb_effective_sample"
    assert np.allclose(is_results.effective_sample_size, is_results_ref.effective_sample_size), error_msg + " > " + "effective_sample_size"
    assert np.allclose(is_results.qn, is_results_ref.qn), error_msg + " > " + "qn"
