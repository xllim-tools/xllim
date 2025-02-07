import xllim
import kernelo
import numpy as np
import time
import csv
import json
import logging

logging.getLogger().setLevel(logging.INFO)

output_file = "comparative_performance_table.csv"  # output csv table file


def write_in_csv(tested_method, result_table):
    with open(output_file, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([tested_method])
        writer.writerows(result_table)
        writer.writerow([])
    print("{} time performance table saved in csv.".format(tested_method))


# ? ###########################################################################
# ?                              FunctionalModel                              #
# ? ###########################################################################


##########################  Set up physical models  ###########################
physical_models = []

# TestModel
physical_models.append(
    {
        "name": "TestModel",
        "xllim": xllim.TestModel(),
        "kernelo": kernelo.TestModelConfig().create(),
    }
)

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
    adapterConfig = kernelo.FourParamsHapkeAdapterConfig(b0, h)

    physical_models.append(
        {
            "name": "Hapke 4p " + geometries["name"],
            "xllim": xllim.HapkeModel(
                geometries["data"], variant, adapter, theta_bar_scaling, b0, h
            ),
            "kernelo": kernelo.HapkeModelConfig(
                variant, adapterConfig, geometries["data"], theta_bar_scaling
            ).create(),
        }
    )

    # Hapke 6p
    variant = "2002"
    adapter = "six"
    theta_bar_scaling = 30.0
    b0 = 0
    h = 0.1
    adapterConfig = kernelo.SixParamsHapkeAdapterConfig()

    physical_models.append(
        {
            "name": "Hapke 6p " + geometries["name"],
            "xllim": xllim.HapkeModel(
                geometries["data"], variant, adapter, theta_bar_scaling, b0, h
            ),
            "kernelo": kernelo.HapkeModelConfig(
                variant, adapterConfig, geometries["data"], theta_bar_scaling
            ).create(),
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
            "kernelo": kernelo.ShkuratovModelConfig(
                geometries["data"], variant, scalingCoeffs, offset
            ).create(),
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
            "kernelo": kernelo.ShkuratovModelConfig(
                geometries["data"], variant, scalingCoeffs, offset
            ).create(),
        }
    )


# ! #############################  TEST : F()  ################################

result_table = [["model", "D", "L", "N", "xllim", "kernelo", "ratio"]]
N_test = 100_000

for physical_model in physical_models:
    L = physical_model["xllim"].getDimensionX()
    D = physical_model["xllim"].getDimensionY()

    start = time.perf_counter()
    for i in range(N_test):
        x = np.random.rand(L)
        y = physical_model["xllim"].F(x)
    time_xllim = time.perf_counter() - start

    start = time.perf_counter()
    for i in range(N_test):
        x = np.random.rand(L)
        y = physical_model["kernelo"].F(x)
    time_kernelo = time.perf_counter() - start

    row = [
        physical_model["name"],
        D,
        L,
        N_test,
        time_xllim,
        time_kernelo,
        time_kernelo / time_xllim,
    ]
    print(row)
    result_table.append(row)

write_in_csv("F()", result_table)


# ! ############################  TEST : genData()  ###########################

result_table = [["model", "D", "L", "N", "generator", "xllim", "kernelo", "ratio"]]

N = 100_000  # number of generated observation
generator_types = ["sobol", "random"]  # "latin"
covariance = np.ones(D) * 1e-5

for physical_model in physical_models:
    L = physical_model["xllim"].getDimensionX()
    D = physical_model["xllim"].getDimensionY()
    for generator in generator_types:
        seed = np.random.randint(1000000)  # seed number for random generators

        start = time.perf_counter()
        x_gen, y_gen = physical_model["xllim"].genData(N, generator, covariance, seed)
        time_xllim = time.perf_counter() - start

        start = time.perf_counter()
        stat_model_kernelo = kernelo.GaussianStatModelConfig(
            generator, physical_model["kernelo"], covariance, seed
        ).create()
        x_gen_ker, y_gen_ker = stat_model_kernelo.gen_data(N)
        time_kernelo = time.perf_counter() - start

        row = [
            physical_model["name"],
            D,
            L,
            N_test,
            generator,
            time_xllim,
            time_kernelo,
            time_kernelo / time_xllim,
        ]
        print(row)
        result_table.append(row)

write_in_csv("genData()", result_table)


# ! #####################  TEST : importanceSampling()  #######################

result_table = [["model", "D", "L", "N_obs", "K", "N_0", "B", "J", "xllim", "kernelo", "ratio"]]
covariance = np.ones(D) * 1e-5

for physical_model in physical_models:
    L = physical_model["xllim"].getDimensionX()
    D = physical_model["xllim"].getDimensionY()
    for K in [5, 50]:
        for N_obs in [10, 1000]:
            x_obs, y_obs = physical_model["xllim"].genData(
                N_obs, "sobol", covariance, seed
            )
            proposition_gmms = []
            for n in range(N_obs):
                weight = np.ones(K) * 1 / K
                mean = np.random.rand(L, K)
                cube = np.ones((L, L, K)) * 0.01
                cube += np.random.rand(L, L, K) * 0.1
                for k in range(cube.shape[2]):
                    cube[:, :, k] += np.eye(L) * 0.1
                    cube[:, :, k] = np.dot(cube[:, :, k], cube[:, :, k].T) * 0.001

                proposition_gmms.append((weight.T, mean, cube))
            y_err = y_obs * 0.001

            for N_0, B, J in [
                (100, 0, 0),
                (1000, 0, 0),
                (100, 10, 5),
                (100, 10, 20),
                (100, 90, 5),
            ]:
                start = time.perf_counter()
                is_results_xllim = physical_model["xllim"].importanceSampling(
                    proposition_gmms,
                    y_obs,
                    y_err,
                    N_0,
                    B=B,
                    J=J,
                    covariance=covariance,
                    verbose=0,
                )
                time_xllim = time.perf_counter() - start

                stat_model_kernelo = kernelo.GaussianStatModelConfig(
                    generator, physical_model["kernelo"], covariance, seed
                ).create()
                imis_sampler = kernelo.ImisConfig(
                    N_0, B, J, stat_model_kernelo
                ).create()
                start = time.perf_counter()
                for i in range(N_obs):
                    proposition_gmm_kernelo = kernelo.GaussianMixturePropositionConfig(
                        proposition_gmms[i][0],
                        proposition_gmms[i][1],
                        proposition_gmms[i][2],
                    ).create()
                    is_results_kernelo = imis_sampler.execute(
                        proposition_gmm_kernelo, y_obs[i], y_err[i]
                    )
                time_kernelo = time.perf_counter() - start

                row = [
                    physical_model["name"],
                    D,
                    L,
                    N_obs,
                    K,
                    N_0,
                    B,
                    J,
                    time_xllim,
                    time_kernelo,
                    time_kernelo / time_xllim,
                ]
                print(row)
                result_table.append(row)

write_in_csv("importanceSampling()", result_table)


# ? ###########################################################################
# ?                                  GLLiM                                    #
# ? ###########################################################################

#####################  Set up general GLLiM parameters  #######################

initialize_result_table = [
    ["gamma_type", "sigma_type", "N", "D", "L", "K", "xllim", "kernelo", "ratio"]
]
train_result_table = [
    ["gamma_type", "sigma_type", "N", "D", "L", "K", "xllim", "kernelo", "ratio"]
]
prediction_result_table = [
    [
        "gamma_type",
        "sigma_type",
        "N",
        "D",
        "L",
        "K",
        "K_merged",
        "xllim",
        "kernelo",
        "ratio",
    ]
]

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

covariance_type_list = ["Full", "Diag", "Iso"]

for gamma_type in covariance_type_list:
    for sigma_type in covariance_type_list:
        for N in [100, 1000]:
            for D in [9, 99]:
                for L in [4, 14]:
                    x_gen_random = np.random.rand(L, N)
                    y_gen_random = np.random.rand(D, N)
                    for K in [5, 50]:

                        #####################  Set up specific GLLiM model  #######################

                        seed = np.random.randint(1000000)

                        gllim_xllim = xllim.GLLiM(
                            K, D, L, gamma_type.lower(), sigma_type.lower(), 0
                        )  # hidden_values = 0

                        if gamma_type == "Full" and sigma_type == "Full":
                            learningConfig = kernelo.GMMLearningConfig(
                                gmm_kmeans_iteration, train_max_iteration, train_floor
                            )
                        else:
                            learningConfig = kernelo.EMLearningConfig(
                                train_max_iteration, train_ratio_ll, train_floor
                            )
                        initConfig = kernelo.MultInitConfig(
                            seed=seed,
                            nb_iter_EM=gllim_em_iteration,
                            nb_experiences=nb_experiences,
                            gmmLearningConfig=kernelo.GMMLearningConfig(
                                gmm_kmeans_iteration, gmm_em_iteration, gmm_floor
                            ),
                        )
                        gllim_kernelo = kernelo.GLLiM(
                            D, L, K, gamma_type, sigma_type, initConfig, learningConfig
                        )

                        # ! #######################  TEST : initialize()  #########################

                        start = time.perf_counter()
                        gllim_xllim.initialize(
                            x_gen_random,
                            y_gen_random,
                            gllim_em_iteration,
                            gllim_em_floor,
                            gmm_kmeans_iteration,
                            gmm_em_iteration,
                            gmm_floor,
                            nb_experiences,
                            seed,
                            1,
                        )
                        time_xllim = time.perf_counter() - start

                        start = time.perf_counter()
                        gllim_kernelo.initialize(x_gen_random, y_gen_random)
                        time_kernelo = time.perf_counter() - start

                        row = [
                            gamma_type,
                            sigma_type,
                            N,
                            D,
                            L,
                            K,
                            time_xllim,
                            time_kernelo,
                            time_kernelo / time_xllim,
                        ]
                        print(row)
                        initialize_result_table.append(row)

                        # ! #########################  TEST : train()  ############################

                        if gamma_type == "Full" and sigma_type == "Full":
                            start = time.perf_counter()
                            gllim_xllim.train(
                                x_gen_random,
                                y_gen_random,
                                gmm_kmeans_iteration,
                                train_ratio_ll,
                                train_floor,
                                1,
                            )
                            time_xllim = time.perf_counter() - start
                        else:
                            start = time.perf_counter()
                            gllim_xllim.train(
                                x_gen_random,
                                y_gen_random,
                                train_max_iteration,
                                train_ratio_ll,
                                train_floor,
                                1,
                            )
                            time_xllim = time.perf_counter() - start

                        start = time.perf_counter()
                        gllim_kernelo.train(x_gen_random, y_gen_random)
                        time_kernelo = time.perf_counter() - start

                        row = [
                            gamma_type,
                            sigma_type,
                            N,
                            D,
                            L,
                            K,
                            time_xllim,
                            time_kernelo,
                            time_kernelo / time_xllim,
                        ]
                        print(row)
                        train_result_table.append(row)

                        # ! ####################  TEST : inverseDensities()  ######################

                        # method = "prediction test data"
                        for K_merged in [2, 5]:
                            prediction_floor = 1e-10

                            predicator_kernelo = kernelo.PredictionConfig(
                                K_merged, K_merged, prediction_floor, gllim_kernelo
                            ).create()  # create Kernelo predictor once

                            start = time.perf_counter()
                            prediction_dataset_xllim = gllim_xllim.inverseDensities(
                                y_gen_random,
                                np.zeros(D),
                                K_merged,
                                prediction_floor,
                            )  # vectorized
                            time_xllim = time.perf_counter() - start

                            start = time.perf_counter()
                            for i in range(N):
                                pred = predicator_kernelo.predict(
                                    y_gen_random[i], np.zeros(D)
                                )
                            time_kernelo = time.perf_counter() - start

                            row = [
                                gamma_type,
                                sigma_type,
                                N,
                                D,
                                L,
                                K,
                                K_merged,
                                time_xllim,
                                time_kernelo,
                                time_kernelo / time_xllim,
                            ]
                            print(row)
                            prediction_result_table.append(row)


write_in_csv("initialize()", initialize_result_table)
write_in_csv("train()", train_result_table)
write_in_csv("inverseDensities()", prediction_result_table)
