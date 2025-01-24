"""
Performance Analysis of Importance Sampling methods.

This script evaluates and compares the computational performance and quality performance of several 'prediction'
methods (GLLiM predictions, IS, IMIS-1, IMIS-2) for centroids and for the mean. The comparison is performed by 
measuring the time taken for prediction, error on x and reconstruction error.

Outputs:
--------
1. A plot showing:
    - Computation time (in seconds) for both C++ and Python implementations as a function of the number of observations.
    - The legend differentiates between the two implementations, with the x-axis representing the number of observations 
      and the y-axis representing computation time.

1. Severeal plot showing:
    - Errors

Usage:
------
This script is designed to be run standalone. Ensure that the `xllim` module is correctly installed and that the 
`../dataRef/externalPythonModels` directory contains the necessary Python model files. 

"""

import os.path
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import time
import json

import xllim


def compute_reconstruction_error(reconstruction, observation):
    return np.linalg.norm(observation - reconstruction) / np.linalg.norm(observation)


def compute_prediction_error(prediction, x):
    return np.linalg.norm(prediction - x) / np.linalg.norm(x)


def best_error_on_x(x_obs, x_pred_1, x_pred_2):
    n_observations = len(x_pred_1)
    error_on_x = []
    x_obs_2 = np.copy(x_obs)
    x_obs_2[:, 2] = 1 - x_obs_2[:, 2] # because bimodal has 2 solutions on x3 (x3 and 1-x3)
    for i in range(n_observations):
        err_1 = np.linalg.norm(x_pred_1[i] - x_obs[i], np.inf)
        err_2 = np.linalg.norm(x_pred_1[i] - x_obs_2[i], np.inf)
        if err_1 < err_2:
            test = np.linalg.norm(x_pred_2[i] - x_obs_2[i], np.inf)
            error_on_x.append(
                np.nanmean([err_1, np.linalg.norm(x_pred_2[i] - x_obs_2[i], np.inf)])
            )
        else:
            error_on_x.append(
                np.nanmean([err_2, np.linalg.norm(x_pred_2[i] - x_obs[i], np.inf)])
            )
    return error_on_x


# function which returns the index of minimum value in the list
def get_minvalue(inputlist):
    # get the minimum value in the list
    min_value = min(inputlist)
    # return the index of minimum value
    min_index = []
    for i in range(0, len(inputlist)):
        if min_value == inputlist[i]:
            min_index.append(i)
    return min_index


def set_up_gllim(physical_model_name, datasize):
    number_of_tests = 1
    # datasize = 50000  # 50000
    nb_centers = 2
    # Create physical model
    # physicalModel = ker.TestModelConfig().create()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if physical_model_name == "hapke":
        # geometries = [
        #     [26.543688820608587, 59.70151372254296, 49.75026883595537, 22.25687341017779, 29.37477052338286, 45.92667930975748, 0.8378149317275863, 0.9539443924366908, 50.6751036681658, 30.939391754687584, 45.35750723379789, 77.4098063051496, 8.550137962131798, 78.75041993019249, 57.21323807619601, 64.45682060312284, 61.27808641039031, 29.87156688024389, 58.23236626065648, 0.46252284502360075, 61.414629361746286, 68.38870017628031, 26.738228752953002, 62.23719667072418, 37.97518925487721, 11.71217249013775, 44.21083972798597, 54.80504984567808, 6.681918781680803, 77.75380284700469, 18.248735054007334, 80.74809445945607, 27.883083585044094, 30.667505907879136, 23.753533303937083, 79.98537185251999, 57.116427002329445, 11.093398071099992, 21.05442110717268, 39.39417817217474, 67.3088045536496, 45.67330930572437, 66.22342831254535, 16.299202757933486, 23.17100000877112, 33.81214522345879, 5.674512543132457, 46.31197332927556, 4.425076559963095, 43.39587018527353],
        #     [57.82353475394024, 75.1388961438802, 53.34082125952007, 59.16388136864589, 56.82524519395523, 13.861455932687909, 61.01513462943814, 0.08022416723222348, 45.20326439963487, 49.75804377781624, 47.57479946505285, 40.83414539871288, 42.582107181671304, 19.543259862080465, 79.42231822683355, 5.608973502992881, 8.519518228991334, 14.653496103650852, 83.62967972867986, 63.630581768018686, 50.97355037385619, 43.39685725973246, 75.09920079639205, 64.31108516343195, 25.878527299490525, 59.95252508789103, 7.817590072337685, 63.917416324181204, 49.224108815946195, 11.8651829920254, 52.905039469008344, 40.1809318771301, 18.01508428188126, 26.55789502901616, 58.87354290360373, 27.541622847565314, 27.640845197335114, 7.317915528927937, 27.26977361165818, 43.424177588459685, 50.72108586970755, 21.515785742893073, 58.810741521507, 74.65446772899669, 65.1588001626688, 57.74835037606868, 40.286095753565505, 80.72344684958227, 46.045282404078506, 31.16502580002508],
        #     [56.860734390493455, 73.9693641883178, 37.73213502869909, 115.17115948730651, 173.9808170538202, 27.381659255810384, 145.5789039136927, 161.34338905048256, 69.03017040464387, 42.873394998955916, 51.61878013029947, 18.286412087024594, 94.0626977818526, 14.107568014786057, 63.49011301889791, 51.18375024794363, 133.34658272658174, 157.10530541937266, 146.0687864919587, 166.71021816930005, 175.97262739732273, 77.64637076284835, 113.89679858896427, 90.10458760238879, 178.93891790686803, 139.34645381672362, 103.98331202999472, 153.69553818327597, 35.75205008184436, 172.68559130787182, 112.44694565309588, 66.94556052034761, 142.26012240479503, 135.8978116867964, 144.40908529405644, 21.29090671425231, 103.1093776827147, 44.744140970189484, 175.96850407152732, 59.46378165314184, 71.48685119783887, 144.2711967755729, 113.06470564914261, 26.905165233458316, 123.66223897723546, 5.7269090111509335, 157.16123545573817, 76.01843764637619, 112.82685894796201, 117.10375378471507]
        # ] # Old geometries where only 3 parameters are constrained.
        geometries = []
        inc = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
        ]
        eme = [
            0.5,
            1.0,
            2.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            10.0,
            18.0,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            22.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            10.0,
            20.0,
            30.0,
            38.0,
            39.0,
            39.5,
            40.0,
            40.5,
            41.0,
            42.0,
            50.0,
            60.0,
            70.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            58.0,
            59.0,
            59.5,
            60.0,
            60.5,
            61.0,
            62.0,
            70.0,
            0.5,
            1.0,
            2.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            10.0,
            18.0,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            22.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            10.0,
            20.0,
            30.0,
            38.0,
            39.0,
            39.5,
            40.0,
            40.5,
            41.0,
            42.0,
            50.0,
            60.0,
            70.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            58.0,
            59.0,
            59.5,
            60.0,
            60.5,
            61.0,
            62.0,
            70.0,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
        ]
        azi = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
            150,
            150,
            150,
            150,
            150,
            150,
            150,
            135,
            135,
            135,
            135,
            135,
            135,
            135,
            180,
            180,
            180,
            180,
            180,
            180,
            180,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
            150,
            150,
            150,
            150,
            150,
            150,
            150,
            135,
            135,
            135,
            135,
            135,
            135,
            135,
            180,
            180,
            180,
            180,
            180,
            180,
            180,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
            150,
            150,
            150,
            150,
            150,
            150,
            150,
            135,
            135,
            135,
            135,
            135,
            135,
            135,
            180,
            180,
            180,
            180,
            180,
            180,
            180,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
            150,
            150,
            150,
            150,
            150,
            150,
            150,
            135,
            135,
            135,
            135,
            135,
            135,
            135,
            180,
            180,
            180,
            180,
            180,
            180,
            180,
        ]
        geometries.append(inc)
        geometries.append(eme)
        geometries.append(azi)
        # # incidences
        # geometries.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 20, 20, 20, 20, 20, 20, 20, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60, 60, 60, 60])
        # # emergence
        # geometries.append([0.5, 1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 10.0, 18.0, 19.0, 19.5, 20.5, 21.0, 22.0, 30.0, 40.0, 50.0, 60.0, 70.0, 10.0, 20.0, 30.0, 40.0, 50.0, 58.0, 59.0, 59.5, 60.5, 61.0, 62.0, 70.0, 10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60, 70])
        # # azimuts
        # geometries.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180])
        geometries = np.array(geometries).T  # transpose geometries for HapkeModel
        physicalModel = xllim.HapkeModel(geometries, "2002", "six", 30.0, 0, 0.1)
        print(physicalModel.getDimensionX())
        print(physicalModel.getDimensionY())

    elif physical_model_name == "Shkuratov.cpp":

        geometries = []
        mukundpura_geometries = {
            "sza": [
                0.0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                60,
                60,
                60,
                60,
                60,
                60,
                60,
                60,
                60,
                60,
                60,
                60,
                60,
                60,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
            ],
            "vza": [
                70.0,
                60,
                50,
                40,
                30,
                20,
                10,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                70,
                60,
                50,
                40,
                30,
                20,
                10,
                0,
                10,
                30,
                40,
                50,
                60,
                70,
                70,
                60,
                50,
                30,
                20,
                10,
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                70,
                60,
                50,
                40,
                30,
                20,
                10,
                0,
                10,
                20,
                30,
                40,
                50,
                70,
                70,
                60,
                50,
                40,
                30,
                10,
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
            ],
            "phi": [
                0.0,
                0,
                0,
                0,
                0,
                0,
                0,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                180,
                0,
                0,
                0,
                0,
                0,
                0,
                30,
                30,
                30,
                30,
                30,
                30,
                150,
                150,
                150,
                150,
                150,
                150,
                150,
                150,
            ],
        }
        geometries.append(mukundpura_geometries["sza"])
        geometries.append(mukundpura_geometries["vza"])
        geometries.append(mukundpura_geometries["phi"])
        geometries = np.array(
            geometries
        ).T  # transpose geometries for Shkuratov.cpp model
        scalingCoeffs = [1.0, 1.5, 1.5, 1.5, 1.5]
        offset = [0, 0, 0.2, 0, 0]
        variant = "5p"
        physicalModel = xllim.ShkuratovModel(geometries, variant, scalingCoeffs, offset)
        print(physicalModel.getDimensionX())
        print(physicalModel.getDimensionY())

    else:
        print(dir_path + "../dataRef/externalPythonModels/")
        physicalModel = xllim.ExternalPythonModel(physical_model_name, physical_model_name.lower(), dir_path + "/../dataRef/externalPythonModels/")


    L = physicalModel.getDimensionX()
    D = physicalModel.getDimensionY()
    print("Test model transform a vector X of dimension L = {} into a vector Y of dimension D = {}\n".format(L, D))


    # gllim_model_file_name = "pytest/savedFiles/" + "gllim_" + physical_model_name + ".file"
    # if not os.path.isfile(gllim_model_file_name):

    print("Generating dataset")
    generation_time = time.time()
    x_gen, y_gen = physicalModel.genData(datasize, "sobol", np.ones(D) * 1e-5, 12345)
    generation_time = time.time() - generation_time

    gllim = xllim.GLLiM(10, D, L, "full", "diag")

    # initialisation parameters
    gllim_em_iteration = 10
    gllim_em_floor = 1e-12
    gmm_kmeans_iteration = 5
    gmm_em_iteration = 10
    gmm_floor = 1e-12
    nb_experiences = 5

    print("initializing GLLIM model")
    initialisation_time = time.time()
    gllim.initialize(
        x_gen,
        y_gen,
        gllim_em_iteration,
        gllim_em_floor,
        gmm_kmeans_iteration,
        gmm_em_iteration,
        gmm_floor,
        nb_experiences,
        12345,
        1,
    )
    initialisation_time = time.time() - initialisation_time

    # training parameters
    train_max_iteration = 100
    train_ratio_ll = 1e-5
    train_floor = 1e-12

    print("training model")
    training_time = time.time()
    gllim.train(x_gen, y_gen, train_max_iteration, train_ratio_ll, train_floor, 1)
    training_time = time.time() - training_time

    #     gllim_parameters = gllim.exportModel()
    #     with open(gllim_model_file_name, "wb") as f:
    #         pickle.dump(gllim_parameters, f, pickle.HIGHEST_PROTOCOL)
    #     f.close()
    #     print("GLLIM model saved")
    # else:
    #     with open(gllim_model_file_name, "rb") as f:
    #         gllim_parameters = pickle.load(f)
    #         gllim.importModel(gllim_parameters)
    #     print("GLLIM model loaded")

    # return physicalModel, statModel, gllim
    return generation_time, initialisation_time, training_time


def generate_test_observations(physical_model, n_observations, type):
    x_obs = np.zeros((n_observations, physical_model.getDimensionX()))
    y_obs = np.zeros((n_observations, physical_model.getDimensionY()))
    y_obs_noised = np.zeros((n_observations, physical_model.getDimensionY()))
    y_obs_noise = np.zeros((n_observations, physical_model.getDimensionY()))

    if type == "sinus":
        for i in range(n_observations):
            for j in range(physical_model.getDimensionX()):
                x_obs[i, j] = (
                    0.4
                    * math.sin(2.0 * math.pi * i / n_observations + (j * math.pi / 4.0))
                    + 0.5
                )
                # x_test_secondary_solution[i, j] = 0.4 * math.sin(2.*math.pi*i/n_samples + (j * math.pi/4.)) + 0.5
                # if j == 2:
                #     x_test_secondary_solution[i, j] = 0.4 * math.sin(2.*math.pi*i/n_samples + (j * math.pi/4.) + math.pi) + 0.5
                # else:
                #     x_test_secondary_solution[i, j] = 0.4 * math.sin(2.*math.pi*i/n_samples + (j * math.pi/4.)) + 0.5
            # x_obs[i, 3] = 0.5
        for i in range(n_observations):
            y_obs[i] = np.nan_to_num(physical_model.F(x_obs[i]))
            # Add noise for each Y component
            for j in range(physical_model.getDimensionY()):
                y_obs_noise[i][j] = (y_obs[i][j] / 1000.0 + 1e-8) * np.random.normal(
                    0, math.pow(y_obs[i][j] / 1000.0 + 1e-8, 2), 1
                )
        y_obs_noised = y_obs + y_obs_noise

    elif type == "randu":
        x_obs = np.random.rand(
            n_observations, physical_model.getDimensionX()
        )  # uniform distribution
        for i in range(n_observations):
            y_obs[i] = physical_model.F(x_obs[i])
            y_obs_noise[i] = y_obs[i] / 50.0
        y_obs_noised = np.copy(y_obs)

    elif type == "shkuratov_obs_100":
        with open("pytest/test_shkuratov.json", "r") as f:
            data = json.load(f)
        variables = ["an", "mu1", "nu", "m", "mu2"]
        scalingCoeffs = [1.0, 1.5, 1.5, 1.5, 1.5]
        offset = [0, 0, 0.2, 0, 0]
        for i in range(n_observations):
            for j in range(physical_model.getDimensionX()):
                v = data[variables[j]][i]
                x_obs[i, j] = (float(v) - offset[j]) / scalingCoeffs[j]
            for j in range(physical_model.getDimensionY()):
                y_obs_noised[i, j] = float(data["y"][i][j])
        y_obs_noise = y_obs_noised / 50

    return x_obs, y_obs_noised, y_obs_noise


def run_one_experience(
    physical_model,
    stat_model,
    gllim,
    n_observations,
    r_experience,
    test_observations_type,
):
    nb_centers = 2
    L = physical_model.getDimensionX()
    D = physical_model.getDimensionY()

    # Generate observations
    x_obs, y_obs, var_obs = generate_test_observations(
        physical_model, n_observations, test_observations_type
    )

    # compute predictions
    prediction_error_on_x = [[] for i in range(nb_centers + 1)]
    prediction_reconstruction_error = [[] for i in range(nb_centers + 1)]
    prediction_time = [[] for i in range(nb_centers + 1)]

    ts = time.time()
    predictions = gllim.inverseDensities(y_obs, var_obs, nb_centers, 1e-10, 0)
    prediction_time[0].append(time.time() - ts)


    # compute IS
    N_samples = 100 * (1 + 5 * (r_experience - 1))

    is_error_on_x = [[] for i in range(nb_centers + 1)]
    is_reconstruction_error = [[] for i in range(nb_centers + 1)]
    is_time = [[] for i in range(nb_centers + 1)]

    ts = time.time()
    is_results = physical_model.importanceSampling(predictions.mergedGMM,y_obs,var_obs, N_samples, 0, 0, np.ones(D) * 0.001, verbose=1)
    is_time[0].append(time.time() - ts)

    # compute IMIS-1
    N_0 = N_samples / 10
    B = N_samples / 20
    J = 18

    imis_1_error_on_x = [[] for i in range(nb_centers + 1)]
    imis_1_reconstruction_error = [[] for i in range(nb_centers + 1)]
    imis_1_time = [[] for i in range(nb_centers + 1)]

    ts = time.time()
    imis_1_results = physical_model.importanceSampling(predictions.mergedGMM,y_obs,var_obs, N_0, B, J, np.ones(D) * 0.001, verbose=1)
    imis_1_time[0].append(time.time() - ts)

    # compute IMIS-2
    N_0 = N_samples / 20
    B = N_samples / 40
    J = 9

    imis_2_error_on_x = [[] for i in range(nb_centers + 1)]
    imis_2_reconstruction_error = [[] for i in range(nb_centers + 1)]
    imis_2_time = [[] for i in range(nb_centers + 1)]

    ts = time.time()
    imis_2_results = physical_model.importanceSampling(predictions.mergedGMM,y_obs,var_obs, N_0, B, J, np.ones(D) * 0.001, verbose=1)
    imis_2_time[0].append(time.time() - ts)


    # for center in range(1, nb_centers + 1):
    #     # TODO


    # compute errors
    N_obs = y_obs.shape[1]
    for n in range(N_obs):

        # prediction
        prediction_reconstruction_error[0].append(compute_reconstruction_error(physical_model.F(predictions.fullGMM.mean[:,n]), y_obs[:,n]))
        prediction_error_on_x[0].append(np.linalg.norm(predictions.fullGMM.mean[:,n] - x_obs[n], np.inf))

        # IS
        is_reconstruction_error[0].append(compute_reconstruction_error(physical_model.F(is_results.predictions[n]), y_obs[:,n]))
        is_error_on_x[0].append(np.linalg.norm(is_results.predictions[n] - x_obs[n], np.inf))

        # IMIS-1
        imis_1_reconstruction_error[0].append(compute_reconstruction_error(physical_model.F(imis_1_results.predictions[n]), y_obs[:,n]))
        imis_1_error_on_x[0].append(np.linalg.norm(imis_1_results.predictions[n] - x_obs[n], np.inf))

        # IMIS-2
        imis_2_reconstruction_error[0].append(compute_reconstruction_error(physical_model.F(imis_2_results.predictions[n]), y_obs[:,n]))
        imis_2_error_on_x[0].append(np.linalg.norm(imis_2_results.predictions[n] - x_obs[n], np.inf))


    return {
        "observation": {"x_obs": x_obs, "y_obs": y_obs, "var_obs": var_obs},
        "prediction_mean": {
            "means": np.array(predictions.fullGMM.mean),
            "error_on_x": prediction_error_on_x[0],
            "reconstruction_error": prediction_reconstruction_error[0],
            "computation_time": prediction_time[0],
        },
        "is_mean": {
            "means": np.array(is_results.predictions),
            "error_on_x": is_error_on_x[0],
            "reconstruction_error": is_reconstruction_error[0],
            "computation_time": is_time[0],
        },
        "imis_1_mean": {
            "means": np.array(imis_1_results.predictions),
            "error_on_x": imis_1_error_on_x[0],
            "reconstruction_error": imis_1_reconstruction_error[0],
            "computation_time": imis_1_time[0],
        },
        "imis_2_mean": {
            "means": np.array(imis_2_results.predictions),
            "error_on_x": imis_2_error_on_x[0],
            "reconstruction_error": imis_2_reconstruction_error[0],
            "computation_time": imis_2_time[0],
        },
        # "prediction_center1": {
        #     "means": np.array(prediction_means[1]),
        #     "error_on_x": prediction_error_on_x[1],
        #     "reconstruction_error": prediction_reconstruction_error[1],
        #     "computation_time": prediction_time[1],
        # },
        # "is_center1": {
        #     "means": np.array(is_means[1]),
        #     "error_on_x": is_error_on_x[1],
        #     "reconstruction_error": is_reconstruction_error[1],
        #     "computation_time": is_time[1],
        # },
        # "imis_1_center1": {
        #     "means": np.array(imis_1_means[1]),
        #     "error_on_x": imis_1_error_on_x[1],
        #     "reconstruction_error": imis_1_reconstruction_error[1],
        #     "computation_time": imis_1_time[1],
        # },
        # "imis_2_center1": {
        #     "means": np.array(imis_2_means[1]),
        #     "error_on_x": imis_2_error_on_x[1],
        #     "reconstruction_error": imis_2_reconstruction_error[1],
        #     "computation_time": imis_2_time[1],
        # },
        # "prediction_center2": {
        #     "means": np.array(prediction_means[2]),
        #     "error_on_x": prediction_error_on_x[2],
        #     "reconstruction_error": prediction_reconstruction_error[2],
        #     "computation_time": prediction_time[2],
        # },
        # "is_center2": {
        #     "means": np.array(is_means[2]),
        #     "error_on_x": is_error_on_x[2],
        #     "reconstruction_error": is_reconstruction_error[2],
        #     "computation_time": is_time[2],
        # },
        # "imis_1_center2": {
        #     "means": np.array(imis_1_means[2]),
        #     "error_on_x": imis_1_error_on_x[2],
        #     "reconstruction_error": imis_1_reconstruction_error[2],
        #     "computation_time": imis_1_time[2],
        # },
        # "imis_2_center2": {
        #     "means": np.array(imis_2_means[2]),
        #     "error_on_x": imis_2_error_on_x[2],
        #     "reconstruction_error": imis_2_reconstruction_error[2],
        #     "computation_time": imis_2_time[2],
        # },
        # "prediction_center_best": {
        #     "error_on_x": best_error_on_x(
        #         x_obs, prediction_means[1], prediction_means[2]
        #     ),
        #     "reconstruction_error": [
        #         np.nanmean(
        #             [
        #                 prediction_reconstruction_error[1][i],
        #                 prediction_reconstruction_error[2][i],
        #             ]
        #         )
        #         for i in range(n_observations)
        #     ],
        #     "computation_time": [
        #         np.nanmean([prediction_time[1][i], prediction_time[2][i]])
        #         for i in range(n_observations)
        #     ],
        # },
        # "is_center_best": {
        #     "error_on_x": best_error_on_x(x_obs, is_means[1], is_means[2]),
        #     "reconstruction_error": [
        #         np.nanmean(
        #             [is_reconstruction_error[1][i], is_reconstruction_error[2][i]]
        #         )
        #         for i in range(n_observations)
        #     ],
        #     "computation_time": [
        #         np.nanmean([is_time[1][i], is_time[2][i]])
        #         for i in range(n_observations)
        #     ],
        # },
        # "imis_1_center_best": {
        #     "error_on_x": best_error_on_x(x_obs, imis_1_means[1], imis_1_means[2]),
        #     "reconstruction_error": [
        #         np.nanmean(
        #             [
        #                 imis_1_reconstruction_error[1][i],
        #                 imis_1_reconstruction_error[2][i],
        #             ]
        #         )
        #         for i in range(n_observations)
        #     ],
        #     "computation_time": [
        #         np.nanmean([imis_1_time[1][i], imis_1_time[2][i]])
        #         for i in range(n_observations)
        #     ],
        # },
        # "imis_2_center_best": {
        #     "error_on_x": best_error_on_x(x_obs, imis_2_means[1], imis_2_means[2]),
        #     "reconstruction_error": [
        #         np.nanmean(
        #             [
        #                 imis_2_reconstruction_error[1][i],
        #                 imis_2_reconstruction_error[2][i],
        #             ]
        #         )
        #         for i in range(n_observations)
        #     ],
        #     "computation_time": [
        #         np.nanmean([imis_2_time[1][i], imis_2_time[2][i]])
        #         for i in range(n_observations)
        #     ],
        # },
    }


def execute_plot(r_experience_list, result_list, experience_name, n_observations_list):

    fig, axs = plt.subplots(2, 3, num=experience_name)
    fig.suptitle(
        "Important sampling methods performance analysis\n" + experience_name,
        fontsize=16,
    )

    # plt.figure("Error on x with respects to experience index")
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["prediction_mean"]["error_on_x"]) for res in result_list], 'b.', label='prediction')
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["is_mean"]["error_on_x"]) for res in result_list], 'r.', label='is')
    axs[0, 0].plot(
        r_experience_list,
        [np.nanmean(res["imis_1_mean"]["error_on_x"]) for res in result_list],
        "g.",
        label="imis_1",
    )
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["imis_2_mean"]["error_on_x"]) for res in result_list], 'm.', label='imis_2 (light)')
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["prediction_center1"]["error_on_x"]) for res in result_list], 'b:', label='prediction_center1')
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["is_center1"]["error_on_x"]) for res in result_list], 'r:', label='is_center1')
    axs[0, 0].plot(
        r_experience_list,
        [np.nanmean(res["imis_1_center1"]["error_on_x"]) for res in result_list],
        "g:",
        label="imis_1_center1",
    )
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["imis_2_center1"]["error_on_x"]) for res in result_list], 'm:', label='imis_2_center1 (light)')
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["prediction_center2"]["error_on_x"]) for res in result_list], 'b--', label='prediction_center2')
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["is_center2"]["error_on_x"]) for res in result_list], 'r--', label='is_center2')
    axs[0, 0].plot(
        r_experience_list,
        [np.nanmean(res["imis_1_center2"]["error_on_x"]) for res in result_list],
        "g--",
        label="imis_1_center2",
    )
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["imis_2_center2"]["error_on_x"]) for res in result_list], 'm--', label='imis_2_center2 (light)')
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["prediction_center_best"]["error_on_x"]) for res in result_list], 'b^', label='prediction best centroid')
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["is_center_best"]["error_on_x"]) for res in result_list], 'r^', label='is best centroid')
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["imis_1_center_best"]["error_on_x"]) for res in result_list], 'g^', label='imis_1 best centroid')
    # axs[0,0].plot(r_experience_list, [np.nanmean(res["imis_2_center_best"]["error_on_x"]) for res in result_list], 'm^', label='imis_2 (light) best centroid')
    axs[0, 0].set_xlabel("Experience index")
    axs[0, 0].set_ylabel(r"$||x_{pred} - x_{obs}||_{\infty}$")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_title("Error on x")
    axs[0, 0].legend()

    # plt.figure("Reconstruction error with respects to experience index")
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["prediction_mean"]["reconstruction_error"]) for res in result_list], 'b.', label='prediction')
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["is_mean"]["reconstruction_error"]) for res in result_list], 'r.', label='is')
    axs[0, 1].plot(
        r_experience_list,
        [np.nanmean(res["imis_1_mean"]["reconstruction_error"]) for res in result_list],
        "g.",
        label="imis_1",
    )
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["imis_2_mean"]["reconstruction_error"]) for res in result_list], 'm.', label='imis_2 (light)')
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["prediction_center1"]["reconstruction_error"]) for res in result_list], 'b:', label='prediction_center1')
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["is_center1"]["reconstruction_error"]) for res in result_list], 'r:', label='is_center1')
    axs[0, 1].plot(
        r_experience_list,
        [
            np.nanmean(res["imis_1_center1"]["reconstruction_error"])
            for res in result_list
        ],
        "g:",
        label="imis_1_center1",
    )
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["imis_2_center1"]["reconstruction_error"]) for res in result_list], 'm:', label='imis_2_center1 (light)')
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["prediction_center2"]["reconstruction_error"]) for res in result_list], 'b--', label='prediction_center2')
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["is_center2"]["reconstruction_error"]) for res in result_list], 'r--', label='is_center2')
    axs[0, 1].plot(
        r_experience_list,
        [
            np.nanmean(res["imis_1_center2"]["reconstruction_error"])
            for res in result_list
        ],
        "g--",
        label="imis_1_center2",
    )
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["imis_2_center2"]["reconstruction_error"]) for res in result_list], 'm--', label='imis_2_center2 (light)')
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["prediction_center_best"]["reconstruction_error"]) for res in result_list], 'b^', label='prediction centroid')
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["is_center_best"]["reconstruction_error"]) for res in result_list], 'r^', label='is centroid')
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["imis_1_center_best"]["reconstruction_error"]) for res in result_list], 'g^', label='imis_1 centroid')
    # axs[0,1].plot(r_experience_list, [np.nanmean(res["imis_2_center_best"]["reconstruction_error"]) for res in result_list], 'm^', label='imis_2 (light) centroid')
    axs[0, 1].set_xlabel("Experience index")
    axs[0, 1].set_ylabel(r"$\frac{||y_{pred} - y_{obs}||_2}{||y_{obs}||_2}$")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_title("Reconstruction error")
    axs[0, 1].legend()

    # plt.figure("Computation time with respects to experience index")
    # axs[0,2].plot(r_experience_list, [np.nanmean(res["prediction_mean"]["computation_time"]) for res in result_list], 'b.', label='prediction')
    # axs[0,2].plot(r_experience_list, [np.nanmean(res["is_mean"]["computation_time"]) for res in result_list], 'r.', label='is')
    axs[0, 2].plot(
        r_experience_list,
        [np.nanmean(res["imis_1_mean"]["computation_time"]) for res in result_list],
        "g.",
        label="imis_1",
    )
    # axs[0,2].plot(r_experience_list, [np.nanmean(res["imis_2_mean"]["computation_time"]) for res in result_list], 'm.', label='imis_2 (light)')
    # axs[0,2].plot(r_experience_list, [np.nanmean(res["prediction_center_best"]["computation_time"]) for res in result_list], 'b^', label='prediction centroid')
    # axs[0,2].plot(r_experience_list, [np.nanmean(res["is_center_best"]["computation_time"]) for res in result_list], 'r^', label='is centroid')
    axs[0, 2].plot(
        r_experience_list,
        [
            np.nanmean(res["imis_1_center_best"]["computation_time"])
            for res in result_list
        ],
        "g^",
        label="imis_1 centroid",
    )
    # axs[0,2].plot(r_experience_list, [np.nanmean(res["imis_2_center_best"]["computation_time"]) for res in result_list], 'm^', label='imis_2 (light) centroid')
    axs[0, 2].set_xlabel("Experience index")
    axs[0, 2].set_ylabel("computation_time (sec)")
    axs[0, 2].set_yscale("log")
    axs[0, 2].set_title("Computation time")
    axs[0, 2].legend()

    # plt.figure("Error on x with respects to computation time")
    # axs[1,0].plot([np.nanmean(res["prediction_mean"]["computation_time"]) for res in result_list], [np.nanmean(res["prediction_mean"]["error_on_x"]) for res in result_list], 'b.', label='prediction')
    # axs[1,0].plot([np.nanmean(res["is_mean"]["computation_time"]) for res in result_list], [np.nanmean(res["is_mean"]["error_on_x"]) for res in result_list], 'r.', label='is')
    axs[1, 0].plot(
        [np.nanmean(res["imis_1_mean"]["computation_time"]) for res in result_list],
        [np.nanmean(res["imis_1_mean"]["error_on_x"]) for res in result_list],
        "g.",
        label="imis_1",
    )
    # axs[1,0].plot([np.nanmean(res["imis_2_mean"]["computation_time"]) for res in result_list], [np.nanmean(res["imis_2_mean"]["error_on_x"]) for res in result_list], 'm.', label='imis_2 (light)')
    # axs[1,0].plot([np.nanmean(res["prediction_center_best"]["computation_time"]) for res in result_list], [np.nanmean(res["prediction_center_best"]["error_on_x"]) for res in result_list], 'b^', label='prediction centroid')
    # axs[1,0].plot([np.nanmean(res["is_center_best"]["computation_time"]) for res in result_list], [np.nanmean(res["is_center_best"]["error_on_x"]) for res in result_list], 'r^', label='is centroid')
    axs[1, 0].plot(
        [
            np.nanmean(res["imis_1_center_best"]["computation_time"])
            for res in result_list
        ],
        [np.nanmean(res["imis_1_center_best"]["error_on_x"]) for res in result_list],
        "g^",
        label="imis_1 centroid",
    )
    # axs[1,0].plot([np.nanmean(res["imis_2_center_best"]["computation_time"]) for res in result_list], [np.nanmean(res["imis_2_center_best"]["error_on_x"]) for res in result_list], 'm^', label='imis_2 (light) centroid')
    axs[1, 0].set_xlabel("computation_time (sec)")
    axs[1, 0].set_ylabel(r"$||x_{pred} - x_{obs}||_{\infty}$")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_title("Error on x with respects to computation time")
    axs[1, 0].legend()

    # plt.figure("Reconstruction error with respects to computation time")
    # axs[1,1].plot([np.nanmean(res["prediction_mean"]["computation_time"]) for res in result_list], [np.nanmean(res["prediction_mean"]["reconstruction_error"]) for res in result_list], 'b.', label='prediction')
    # axs[1,1].plot([np.nanmean(res["is_mean"]["computation_time"]) for res in result_list], [np.nanmean(res["is_mean"]["reconstruction_error"]) for res in result_list], 'r.', label='is')
    axs[1, 1].plot(
        [np.nanmean(res["imis_1_mean"]["computation_time"]) for res in result_list],
        [np.nanmean(res["imis_1_mean"]["reconstruction_error"]) for res in result_list],
        "g.",
        label="imis_1",
    )
    # axs[1,1].plot([np.nanmean(res["imis_2_mean"]["computation_time"]) for res in result_list], [np.nanmean(res["imis_2_mean"]["reconstruction_error"]) for res in result_list], 'm.', label='imis_2 (light)')
    # axs[1,1].plot([np.nanmean(res["prediction_center_best"]["computation_time"]) for res in result_list], [np.nanmean(res["prediction_center_best"]["reconstruction_error"]) for res in result_list], 'b^', label='prediction')
    # axs[1,1].plot([np.nanmean(res["is_center_best"]["computation_time"]) for res in result_list], [np.nanmean(res["is_center_best"]["reconstruction_error"]) for res in result_list], 'r^', label='is')
    axs[1, 1].plot(
        [
            np.nanmean(res["imis_1_center_best"]["computation_time"])
            for res in result_list
        ],
        [
            np.nanmean(res["imis_1_center_best"]["reconstruction_error"])
            for res in result_list
        ],
        "g^",
        label="imis_1",
    )
    # axs[1,1].plot([np.nanmean(res["imis_2_center_best"]["computation_time"]) for res in result_list], [np.nanmean(res["imis_2_center_best"]["reconstruction_error"]) for res in result_list], 'm^', label='imis_2 (light)')
    axs[1, 1].set_xlabel("computation_time (sec)")
    axs[1, 1].set_ylabel(r"$\frac{||y_{pred} - y_{obs}||_2}{||y_{obs}||_2}$")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title("Reconstruction error with respects to computation time")
    axs[1, 1].legend()

    # plt.show()

    for r in r_experience_list:

        fig2, axs2 = plt.subplots(
            2, 3, num=experience_name + " - " + str(r) + "- Parameters"
        )
        fig2.suptitle("Parameters comparaison\n" + experience_name, fontsize=16)

        for l in range(result_list[0]["observation"]["x_obs"].shape[1]):

            i = 0 if (l < 3) else 1
            j = l if (l < 3) else l - 3

            axs2[i, j].plot(
                n_observations_list,
                result_list[r - 1]["observation"]["x_obs"][:, l],
                "k-",
                label="observation",
            )
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["prediction_mean"]["means"][:,l], 'b.', label='prediction')
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["is_mean"]["means"][:,l], 'b.', label='is mean')
            axs2[i, j].plot(
                n_observations_list,
                result_list[r - 1]["imis_1_mean"]["means"][:, l],
                "r.",
                label="imis_1 mean",
            )
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["imis_2_mean"]["means"][:,l], 'm.', label='imis_2 (light) mean')
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["prediction_center1"]["means"][:,l], 'b:', label='prediction_center1')
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["is_center1"]["means"][:,l], 'r:', label='is_center1')
            axs2[i, j].plot(
                n_observations_list,
                result_list[r - 1]["imis_1_center1"]["means"][:, l],
                "b.",
                label="imis_1_center1",
            )
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["imis_2_center1"]["means"][:,l], 'm:', label='imis_2_center1 (light)')
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["prediction_center2"]["means"][:,l], 'b--', label='prediction_center2')
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["is_center2"]["means"][:,l], 'r--', label='is_center2')
            axs2[i, j].plot(
                n_observations_list,
                result_list[r - 1]["imis_1_center2"]["means"][:, l],
                "g.",
                label="imis_1_center2",
            )
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["imis_2_center2"]["means"][:,l], 'm--', label='imis_2_center2 (light)')
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["prediction_center_best"]["means"][:,l], 'b^', label='prediction_center_best')
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["is_center_best"]["means"][:,l], 'r^', label='is_center_best')
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["imis_1_center_best"]["means"][:,l], 'g^', label='imis_1_center_best')
            # axs2[i,j].plot(n_observations_list, result_list[r-1]["imis_2_center_best"]["means"][:,l], 'm^', label='imis_2_center_best (light)')
            axs2[i, j].set_title("X_" + str(l))
            axs2[i, j].legend()

        fig3, axs3 = plt.subplots(
            2, 2, num=experience_name + " - " + str(r) + "- Errors"
        )
        fig3.suptitle("Errors comparaison\n" + experience_name, fontsize=16)

        # plt.figure("Computation time with respects to experience index")
        # axs3[0,0].plot(n_observations_list, result_list[r-1]["prediction_mean"]["reconstruction_error"], 'b-', label='prediction')
        # axs3[0,0].plot(n_observations_list, result_list[r-1]["is_mean"]["reconstruction_error"], 'r-', label='is mean')
        axs3[0, 0].plot(
            n_observations_list,
            result_list[r - 1]["imis_1_mean"]["reconstruction_error"],
            "r-",
            label="imis_1 mean",
        )
        # axs3[0,0].plot(n_observations_list, result_list[r-1]["imis_2_mean"]["reconstruction_error"], 'm-', label='imis_2 (light)')
        # axs3[0,0].plot(n_observations_list, result_list[r-1]["prediction_center_best"]["reconstruction_error"], 'b^', label='prediction centroid')
        # axs3[0,0].plot(n_observations_list, result_list[r-1]["is_center_best"]["reconstruction_error"], 'r^', label='is centroid')
        # axs3[0,0].plot(n_observations_list, result_list[r-1]["imis_1_center_best"]["reconstruction_error"], 'g^', label='imis_1 centroid')
        # axs3[0,0].plot(n_observations_list, result_list[r-1]["imis_2_center_best"]["reconstruction_error"], 'm^', label='imis_2 (light) centroid')
        axs3[0, 0].plot(
            n_observations_list,
            result_list[r - 1]["imis_1_center1"]["reconstruction_error"],
            "b-",
            label="imis_1_center1",
        )
        axs3[0, 0].plot(
            n_observations_list,
            result_list[r - 1]["imis_1_center2"]["reconstruction_error"],
            "g-",
            label="imis_1_center2",
        )
        axs3[0, 0].set_xlabel("Observations")
        axs3[0, 0].set_ylabel("Reconstruction error")
        # axs3[0,0].set_yscale('log')
        axs3[0, 0].set_title("Reconstruction error")
        axs3[0, 0].legend()

    plt.show()


def run():

    model_test = ["RPVModel", "Shkuratov.cpp"]
    datasize_list = [1000, 5000, 10_000]
    results_py_cpp = []
    for model in model_test:
        gen_list = []
        init_list = []
        train_list = []
        for size in datasize_list:
            physical_model_name = model
            datasize = size
            # n_observations = 100
            # r_experiences = 10
            # test_observations_type = "sinus"
            # result_list = []

            generation_time, initialisation_time, training_time = set_up_gllim(
                physical_model_name, datasize
            )
            gen_list.append(generation_time)
            init_list.append(initialisation_time)
            train_list.append(training_time)
        results_py_cpp.append([gen_list, init_list, train_list])

        # for r in range(1, r_experiences+1):
        #     experience_name = "experience_" + str(r) + "__" + physical_model_name + "_" + test_observations_type + "_" + str(datasize) + "_" + str(n_observations)
        #     if not os.path.isfile("pytest/savedFiles/" + experience_name):
        #         result = run_one_experience(physical_model, stat_model, gllim, n_observations, r, test_observations_type)
        #         with open("pytest/savedFiles/" + experience_name, "wb") as f:
        #             pickle.dump(result, f)
        #         f.close()
        #         print("Experience " + str(r) + " saved")
        #     else:
        #         with open("pytest/savedFiles/" + experience_name, "rb") as f:
        #             result = pickle.load(f)
        #         print("Experience " + str(r) + " loaded")
        #     result_list.append(result)
        # results_py_cpp.append(result_list)

    # Plots
    # r_experience_list = np.arange(1, r_experiences+1)
    # n_obervations_list = np.arange(1, n_observations+1)
    # execute_plot(r_experience_list, result_list, experience_name, n_obervations_list)

    plt.plot(datasize_list, results_py_cpp[0][0], "r.", label="generation time PYTHON")
    plt.plot(datasize_list, results_py_cpp[1][0], "g.", label="generation time CPP")
    plt.plot(datasize_list, results_py_cpp[0][1], "r^", label="initialiation PYTHON")
    plt.plot(datasize_list, results_py_cpp[1][1], "g^", label="initialisation CPP")
    plt.plot(datasize_list, results_py_cpp[0][2], "r:", label="training PYTHON")
    plt.plot(datasize_list, results_py_cpp[1][2], "g:", label="training CPP")
    plt.xlabel("datasize")
    plt.ylabel("Computation time")
    plt.yscale("log")
    plt.title("Comparison GLLiM computation time PYTHON / CPP")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
