import xllim
import numpy as np
import json
import pickle
import logging

logging.getLogger().setLevel(logging.INFO)

# fix the RNG seed for all tests
seed = 12345

# ? ###########################################################################
# ?                              FunctionalModel                              #
# ? ###########################################################################


##########################  Set up physical models  ###########################
physical_models = []

# TestModel
physical_models.append({"name": "TestModel", "xllim": xllim.TestModel()})

# Get JSC1 geometries from JSON file
with open("../dataRef/JSC1_BRDF.json", "r") as f:
    data = json.load(f)
geometries_JSC1 = {
    "name": "JSC1",
    "data": np.array(data["JSC1_analogue"]["geometries"], dtype=float),
}

# Get Mukundpura geometries from JSON file
with open("../dataRef/mukundpura_bloc_poudre_BRDF.json", "r") as f:
    data = json.load(f)
geometries_muk = {
    "name": "muk",
    "data": np.array(data["Mukundpura"]["geometries"], dtype=float),
}

for geometries in [geometries_JSC1, geometries_muk]:

    # Hapke 4p
    variant = "2002"
    adapter = "four"
    theta_bar_scaling = 30.0
    b0 = 0
    h = 0.1

    physical_models.append(
        {
            "name": "Hapke 4p " + geometries["name"],
            "xllim": xllim.HapkeModel(
                geometries["data"], variant, adapter, theta_bar_scaling, b0, h
            ),
        }
    )

    # Hapke 6p
    variant = "2002"
    adapter = "six"
    theta_bar_scaling = 30.0
    b0 = 0
    h = 0.1

    physical_models.append(
        {
            "name": "Hapke 6p " + geometries["name"],
            "xllim": xllim.HapkeModel(
                geometries["data"], variant, adapter, theta_bar_scaling, b0, h
            ),
        }
    )

    # Schkuratov 5p
    variant = "5p"
    scalingCoeffs = [1.0, 1.5, 1.5, 1.5, 1.5]
    offset = [0, 0, 0.2, 0, 0]

    physical_models.append(
        {
            "name": "Shkuratov 5p " + geometries["name"],
            "xllim": xllim.ShkuratovModel(
                geometries["data"], variant, scalingCoeffs, offset
            ),
        }
    )

    # Schkuratov 5p
    variant = "3p"
    scalingCoeffs = [1.0, 1.5, 1.5, 1.5, 1.5]
    offset = [0, 0, 0.2, 0, 0]

    physical_models.append(
        {
            "name": "Shkuratov 3p " + geometries["name"],
            "xllim": xllim.ShkuratovModel(
                geometries["data"], variant, scalingCoeffs, offset
            ),
        }
    )


# ! #####################  TEST : importanceSampling()  #######################

K, N_obs = 5, 10
N_0, B, J = 60, 8, 5

for physical_model in physical_models:
    L = physical_model["xllim"].getDimensionX()
    D = physical_model["xllim"].getDimensionY()
    covariance = np.ones(D) * 1e-5

    np.random.seed(seed)
    y_obs = np.random.rand(N_obs, D)

    proposition_gmms = []
    for n in range(N_obs):
        weight = np.ones(K) * 1 / K
        np.random.seed(seed)
        mean = np.random.rand(L, K)
        cube = np.ones((L, L, K)) * 0.01
        np.random.seed(seed)
        cube += np.random.rand(L, L, K) * 0.1
        for k in range(cube.shape[2]):
            cube[:, :, k] += np.eye(L) * 0.1
            cube[:, :, k] = np.dot(cube[:, :, k], cube[:, :, k].T) * 0.001

        proposition_gmms.append((weight.T, mean, cube))
    y_err = y_obs * 0.001

    is_results = physical_model["xllim"].importanceSampling(
        proposition_gmms,
        y_obs,
        y_err,
        covariance,
        N_0,
        B,
        J,
        0,
        seed
    )

    # # ! Only run once to generate ref
    # with open("../dataRef/is_results_ref/is_results_ref_{}.file".format(physical_model["name"]), "wb") as f:
    #     pickle.dump(is_results, f)
    #     f.close()

    with open("../dataRef/is_results_ref/is_results_ref_{}.file".format(physical_model["name"]), "rb") as f:
        is_results_ref = pickle.load(f)
        f.close()

    # compare results
    print(physical_model["name"])
    print("is_results")
    print(is_results.predictions)
    print("is_results_ref")
    print(is_results_ref.predictions)
    

    assert np.allclose(is_results.predictions, is_results_ref.predictions)
    assert np.allclose(is_results.predictions_variance, is_results_ref.predictions_variance)
    assert np.allclose(is_results.nb_effective_sample, is_results_ref.nb_effective_sample)
    assert np.allclose(is_results.effective_sample_size, is_results_ref.effective_sample_size)
    assert np.allclose(is_results.qn, is_results_ref.qn)


# ? ###########################################################################
# ?                                  GLLiM                                    #
# ? ###########################################################################

#####################  Set up general GLLiM parameters  #######################

# initialisation parameters
gllim_em_iteration = 10
gllim_em_floor = 1e-12
gmm_kmeans_iteration = 10
gmm_em_iteration = 10
gmm_floor = 1e-12
nb_experiences = 10

# training parameters
train_max_iteration = 100
train_ratio_ll = -1000  # Force train to reach max iteration
train_floor = 1e-12
K_merged = 2

covariance_type_list = ["Full", "Diag", "Iso"]
K, D, L, N = 5, 9, 4, 100

for gamma_type in covariance_type_list:
    for sigma_type in covariance_type_list:
        np.random.seed(seed)
        x_gen_random = np.random.rand(N, L)
        np.random.seed(seed)
        y_gen_random = np.random.rand(N, D)

        #####################  Set up specific GLLiM model  #######################

        gllim = xllim.GLLiM(
            K, D, L, gamma_type.lower(), sigma_type.lower(), 0
        )  # hidden_values = 0

        # ! #######################  TEST : initialize()  #########################

        gllim.initialize(
            np.array(x_gen_random.T),
            np.array(y_gen_random.T),
            gllim_em_iteration,
            gllim_em_floor,
            gmm_kmeans_iteration,
            gmm_em_iteration,
            gmm_floor,
            nb_experiences,
            seed,
            1,
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
        assert np.allclose(gllim_params_initialised.Pi, gllim_params_initialised_ref.Pi)
        assert np.allclose(gllim_params_initialised.A, gllim_params_initialised_ref.A)
        assert np.allclose(gllim_params_initialised.B, gllim_params_initialised_ref.B)
        assert np.allclose(gllim_params_initialised.C, gllim_params_initialised_ref.C)
        assert np.allclose(gllim_params_initialised.Gamma, gllim_params_initialised_ref.Gamma)
        assert np.allclose(gllim_params_initialised.Sigma, gllim_params_initialised_ref.Sigma)

        # ! #########################  TEST : train()  ############################

        if gamma_type == "Full" and sigma_type == "Full":
            gllim.train(
                np.array(x_gen_random.T),
                np.array(y_gen_random.T),
                gmm_kmeans_iteration,
                train_ratio_ll,
                train_floor,
                1,
            )
        else:
            gllim.train(
                np.array(x_gen_random.T),
                np.array(y_gen_random.T),
                train_max_iteration,
                train_ratio_ll,
                train_floor,
                1,
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
        assert np.allclose(gllim_params_trained.Pi, gllim_params_trained_ref.Pi)
        assert np.allclose(gllim_params_trained.A, gllim_params_trained_ref.A)
        assert np.allclose(gllim_params_trained.B, gllim_params_trained_ref.B)
        assert np.allclose(gllim_params_trained.C, gllim_params_trained_ref.C)
        assert np.allclose(gllim_params_trained.Gamma, gllim_params_trained_ref.Gamma)
        assert np.allclose(gllim_params_trained.Sigma, gllim_params_trained_ref.Sigma)


        # ! ####################  TEST : inverseDensities()  ######################

        prediction_floor = 1e-10

        prediction_results = gllim.inverseDensities(
            np.array(y_gen_random.T),
            np.zeros(D),
            K_merged,
            prediction_floor,
        )  # vectorized

        # ! Only run once to generate ref
        with open("../dataRef/prediction_results_ref/prediction_results_ref_{}_{}.file".format(gamma_type, sigma_type), "wb") as f:
            pickle.dump(prediction_results, f)
            f.close()

        with open("../dataRef/prediction_results_ref/prediction_results_ref_{}_{}.file".format(gamma_type, sigma_type), "rb") as f:
            prediction_results_ref = pickle.load(f)
            f.close()

        # compare results
        assert np.allclose(prediction_results.fullGMM.weights, prediction_results_ref.fullGMM.weights)
        assert np.allclose(prediction_results.fullGMM.means, prediction_results_ref.fullGMM.means)
        assert np.allclose(prediction_results.fullGMM.covs, prediction_results_ref.fullGMM.covs)
        assert np.allclose(prediction_results.fullGMM.mean, prediction_results_ref.fullGMM.mean)
        assert np.allclose(prediction_results.fullGMM.variance, prediction_results_ref.fullGMM.variance)
        assert np.allclose(prediction_results.mergedGMM.weights, prediction_results_ref.mergedGMM.weights)
        assert np.allclose(prediction_results.mergedGMM.means, prediction_results_ref.mergedGMM.means)
        assert np.allclose(prediction_results.mergedGMM.covs, prediction_results_ref.mergedGMM.covs)
        assert np.allclose(prediction_results.mergedGMM.mean, prediction_results_ref.mergedGMM.mean)
        assert np.allclose(prediction_results.mergedGMM.variance, prediction_results_ref.mergedGMM.variance)
