"""
Integration Tests for GLLiM Model Initialization, Training, and Prediction.

This script uses pytest to validate the correctness of the GLLiM model's initialization, training, and
prediction steps. The tests compare the results with precomputed reference outputs to ensure reproducibility
and correctness.

Modules:
    - pytest: For organizing and running test cases.
    - numpy: For numerical operations.
    - pickle: For serializing and deserializing reference data.
    - xllim: Custom library containing GLLiM implementations.

Reference Data:
    Precomputed reference outputs are stored in `../dataRef` for validation.
"""

import pytest
import numpy as np
import pickle
import xllim

# Fix the random seed for reproducibility across all tests
SEED = 4444

# General GLLiM parameters
GLLIM_EM_ITERATION = 10
GLLIM_EM_FLOOR = 1e-12
GMM_KMEANS_ITERATION = 10
GMM_EM_ITERATION = 10
GMM_FLOOR = 1e-12
NB_EXPERIENCES = 10

TRAIN_MAX_ITERATION = 100
TRAIN_RATIO_LL = -1000  # Force training to reach max iterations
TRAIN_FLOOR = 1e-12
K_MERGED = 2

# Model parameters
COVARIANCE_TYPE_LIST = ["full", "diag", "iso"]
K, D, L, N_GEN, N_TEST = 5, 9, 4, 1000, 10  # Dimensions and dataset sizes

# Load pre-generated datasets
# X_GEN, Y_GEN = xllim.TestModel().genData(N_gen, "sobol", 20, seed)
# X_TEST, Y_TEST = xllim.TestModel().genData(N_test, "sobol", 20, seed)
# ! Yet GenData() is not returning the expected same dataset with a fixed seed
# X_GEN = np.load("../dataRef/x_gen_TestModel.npy")
# Y_GEN = np.load("../dataRef/y_gen_TestModel.npy")
# Y_TEST = np.load("../dataRef/y_test_TestModel.npy")
np.random.seed(SEED)
X_GEN = np.random.rand(N_GEN, L)
np.random.seed(SEED)
Y_GEN = np.random.rand(N_GEN, D)
np.random.seed(SEED)
Y_TEST = np.random.rand(N_TEST, D)

@pytest.mark.parametrize("gamma_type", COVARIANCE_TYPE_LIST)
@pytest.mark.parametrize("sigma_type", COVARIANCE_TYPE_LIST)
def test_gllim_workflow(gamma_type, sigma_type):
    """
    Test the full GLLiM workflow: initialization, training, and prediction.

    Args:
        gamma_type (str): Covariance type for gamma (e.g., "full").
        sigma_type (str): Covariance type for sigma (e.g., "full").
    """

    # Initialize the GLLiM model
    gllim = xllim.GLLiM(K, D, L, gamma_type, sigma_type, 0)  # hidden_values = 0

    # ! #######################  TEST : initialize()  #########################

    gllim.initialize(
        np.array(X_GEN.T),
        np.array(Y_GEN.T),
        GLLIM_EM_ITERATION,
        GLLIM_EM_FLOOR,
        GMM_KMEANS_ITERATION,
        GMM_EM_ITERATION,
        GMM_FLOOR,
        NB_EXPERIENCES,
        SEED,
        1, # verbose
    )
    gllim_params_initialised = gllim.getParams()

    # # ! Only run once to generate ref
    # with open("../dataRef/gllim_params_initialised_ref/gllim_params_initialised_ref_{}_{}.file".format(gamma_type, sigma_type), "wb") as f:
    #     pickle.dump(gllim_params_initialised, f)
    #     f.close()

    with open("../dataRef/gllim_params_initialised_ref/gllim_params_initialised_ref_{}_{}.file".format(gamma_type, sigma_type), "rb") as f:
        gllim_params_initialised_ref = pickle.load(f)
        f.close()

    # compare results
    error_msg = "initialize" + " > " + gamma_type + "/" + sigma_type
    assert np.allclose(gllim_params_initialised.Pi, gllim_params_initialised_ref.Pi), error_msg + " > " + "Pi"
    assert np.allclose(gllim_params_initialised.A, gllim_params_initialised_ref.A), error_msg + " > " + "A"
    assert np.allclose(gllim_params_initialised.B, gllim_params_initialised_ref.B), error_msg + " > " + "B"
    assert np.allclose(gllim_params_initialised.C, gllim_params_initialised_ref.C), error_msg + " > " + "C"
    assert np.allclose(gllim_params_initialised.Gamma, gllim_params_initialised_ref.Gamma), error_msg + " > " + "Gamma"
    assert np.allclose(gllim_params_initialised.Sigma, gllim_params_initialised_ref.Sigma), error_msg + " > " + "Sigma"

    # ! #########################  TEST : train()  ############################

    if gamma_type == "full" and sigma_type == "full":
        gllim.train(
            np.array(X_GEN.T),
            np.array(Y_GEN.T),
            GMM_KMEANS_ITERATION,
            TRAIN_RATIO_LL,
            TRAIN_FLOOR,
            0, # verbose
        )
    else:
        gllim.train(
            np.array(X_GEN.T),
            np.array(Y_GEN.T),
            TRAIN_MAX_ITERATION,
            TRAIN_RATIO_LL,
            TRAIN_FLOOR,
            0, # verbose
        )
    gllim_params_trained = gllim.getParams()

    # # ! Only run once to generate ref
    # with open("../dataRef/gllim_params_trained_ref/gllim_params_trained_ref_{}_{}.file".format(gamma_type, sigma_type), "wb") as f:
    #     pickle.dump(gllim_params_trained, f)
    #     f.close()

    with open("../dataRef/gllim_params_trained_ref/gllim_params_trained_ref_{}_{}.file".format(gamma_type, sigma_type), "rb") as f:
        gllim_params_trained_ref = pickle.load(f)
        f.close()

    # compare results
    error_msg = "train" + " > " + gamma_type + "/" + sigma_type
    assert np.allclose(gllim_params_trained.Pi, gllim_params_trained_ref.Pi), error_msg + " > " + "Pi"
    assert np.allclose(gllim_params_trained.A, gllim_params_trained_ref.A), error_msg + " > " + "A"
    assert np.allclose(gllim_params_trained.B, gllim_params_trained_ref.B), error_msg + " > " + "B"
    assert np.allclose(gllim_params_trained.C, gllim_params_trained_ref.C), error_msg + " > " + "C"
    assert np.allclose(gllim_params_trained.Gamma, gllim_params_trained_ref.Gamma), error_msg + " > " + "Gamma"
    assert np.allclose(gllim_params_trained.Sigma, gllim_params_trained_ref.Sigma), error_msg + " > " + "Sigma"


    # ! ####################  TEST : inverseDensities()  ######################

    prediction_results = gllim.inverseDensities(
        np.array(Y_TEST.T),
        np.zeros(D),
        K_MERGED,
        1e-10, # prediction_floor
    )

    # # ! Only run once to generate ref
    # with open("../dataRef/prediction_results_ref/prediction_results_ref_{}_{}.file".format(gamma_type, sigma_type), "wb") as f:
    #     pickle.dump(prediction_results, f)
    #     f.close()

    with open("../dataRef/prediction_results_ref/prediction_results_ref_{}_{}.file".format(gamma_type, sigma_type), "rb") as f:
        prediction_results_ref = pickle.load(f)
        f.close()

    # compare results
    # ! Note:   These assertions are evaluated with rtol=1e-3 (lower than the fedault value of 1e-5) because of some issue implying 
    # !         approximation on matrix inversion by Armadillo v10 method. For more details, check the related Gitalb Issue untitled
    # !         "intergation test failure implying Armadillo inversion" (https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is/-/issues/34)
    error_msg = "inverseDensities" + " > " + gamma_type + "/" + sigma_type
    assert np.allclose(prediction_results.fullGMM.weights, prediction_results_ref.fullGMM.weights), error_msg + " > " + "fullGMM.weights"
    assert np.allclose(prediction_results.fullGMM.means, prediction_results_ref.fullGMM.means), error_msg + " > " + "fullGMM.means"
    assert np.allclose(prediction_results.fullGMM.covs, prediction_results_ref.fullGMM.covs), error_msg + " > " + "fullGMM.covs"
    assert np.allclose(prediction_results.fullGMM.mean, prediction_results_ref.fullGMM.mean), error_msg + " > " + "fullGMM.mean"
    assert np.allclose(prediction_results.fullGMM.variance, prediction_results_ref.fullGMM.variance), error_msg + " > " + "fullGMM.variance"
    assert np.allclose(prediction_results.mergedGMM.weights, prediction_results_ref.mergedGMM.weights), error_msg + " > " + "mergedGMM.weights"
    assert np.allclose(prediction_results.mergedGMM.means, prediction_results_ref.mergedGMM.means), error_msg + " > " + "mergedGMM.means"
    assert np.allclose(prediction_results.mergedGMM.covs, prediction_results_ref.mergedGMM.covs), error_msg + " > " + "mergedGMM.covs"
    assert np.allclose(prediction_results.mergedGMM.mean, prediction_results_ref.mergedGMM.mean), error_msg + " > " + "mergedGMM.mean"
    assert np.allclose(prediction_results.mergedGMM.variance, prediction_results_ref.mergedGMM.variance), error_msg + " > " + "mergedGMM.variance"
