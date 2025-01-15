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
SEED = 777

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

# Seed-generated datasets
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
    Test the complete GLLiM workflow, including initialization, training, and prediction, for various 
    covariance configurations.

    This test validates the behavior of the GLLiM model when using different covariance types (`gamma_type` and 
    `sigma_type`) for the latent and observation spaces. It ensures that each step of the workflow produces 
    consistent results that match precomputed reference data.

    Parameters:
        gamma_type (str): Covariance type for the latent space, e.g., "full", "diag", or "iso".
        sigma_type (str): Covariance type for the observation space, e.g., "full", "diag", or "iso".

    Test Steps:
        1. **Initialization Test**:
        - Initialize the GLLiM model with specified parameters and generate initial model parameters.
        - Compare the initial parameters (`Pi`, `A`, `B`, `C`, `Gamma`, `Sigma`) to precomputed reference values 
            stored in files under `../dataRef/gllim_params_initialised_ref/`.
        2. **Training Test**:
        - Train the GLLiM model using synthetic input-output data (`X_GEN`, `Y_GEN`).
        - For "full/full" covariance types, use a specific Armadillo EM-based training method; otherwise, use the xllim 
            GLLiM-EM method.
        - Compare the trained parameters to precomputed reference values stored in 
            `../dataRef/gllim_params_trained_ref/`.
        3. **Prediction Test**:
        - Perform predictions using the `inverseDensities` method, which calculates posterior densities 
            for given observations (`Y_TEST`).
        - Compare the predicted GMM components (`weights`, `means`, `covariances`, `mean`, `variance`) for both 
            the full and merged GMMs to precomputed reference results stored in 
            `../dataRef/prediction_results_ref/`.

    Reference Data:
        Precomputed reference results are stored in separate files for each covariance configuration:
        - `gllim_params_initialised_ref_<gamma_type>_<sigma_type>.file`: Initial model parameters.
        - `gllim_params_trained_ref_<gamma_type>_<sigma_type>.file`: Trained model parameters.
        - `prediction_results_ref_<gamma_type>_<sigma_type>.file`: Prediction results (posterior densities).

    Assertions:
        - Ensures that:
            - Initial and trained model parameters (`Pi`, `A`, `B`, `C`, `Gamma`, `Sigma`) match reference values.
            - Predicted posterior densities (weights, means, and covariances for full and merged GMMs) match reference values.

    Error Reporting:
        - Includes the `gamma_type` and `sigma_type` in error messages for precise traceability.
        - Each mismatch specifies the parameter or prediction output that failed.

    Marks:
        - `pytest.mark.parametrize`: Dynamically tests multiple combinations of `gamma_type` and `sigma_type`.

    Inputs and Global Variables:
        - `X_GEN` (array): Synthetic input data for training.
        - `Y_GEN` (array): Synthetic output data for training.
        - `Y_TEST` (array): Synthetic observation data for prediction.
        - `K`, `D`, `L` (int): GLLiM model parameters (number of clusters, observation dimension, latent dimension).
        - `GLLIM_EM_ITERATION`, `TRAIN_MAX_ITERATION` (int): Maximum iterations for training.
        - `GLLIM_EM_FLOOR`, `TRAIN_FLOOR` (float): Minimum value floors for training steps.
        - `GMM_KMEANS_ITERATION`, `GMM_EM_ITERATION` (int): Iterations for GMM initialization.
        - `NB_EXPERIENCES` (int): Number of training experiences.
        - `SEED` (int): Random seed for reproducibility.
        - `K_MERGED` (int): Number of clusters for the merged GMM in prediction.

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


# ! #####################  TODO  #######################
# !     - getInverse()
# !     - directDensities() (2 signatures)
# !     - inverseDensities() (2 signatures)
# !     - getInsights()
# !     - getters
# !     - setters
