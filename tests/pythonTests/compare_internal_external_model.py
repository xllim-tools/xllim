"""
Performance Analysis of C++ and Python Implementations of a Model Function `F(x)`

This script evaluates and compares the computational performance of two implementations of the function `F(x)`: 
one in C++ (`cppModel`) and the other in Python (`pythonModel`). The comparison is performed by measuring 
the time taken to compute `F(x)` for varying numbers of observations.


Outputs:
--------
1. A plot showing:
    - Computation time (in seconds) for both C++ and Python implementations as a function of the number of observations.
    - The legend differentiates between the two implementations, with the x-axis representing the number of observations 
      and the y-axis representing computation time.

Dependencies:
-------------
- `numpy`: For numerical computations and array manipulations.
- `matplotlib.pyplot`: For visualization.
- `time`: To measure computation time.
- `xllim`: For model creation and evaluation (C++ and Python models).

Usage:
------
This script is designed to be run standalone. Ensure that the `xllim` module is correctly installed and that the 
`../dataRef/externalPythonModels` directory contains the necessary Python model files. 

Notes:
------
- The C++ and Python models should produce identical results for `F(x)`, ensuring a fair comparison of computational 
  performance.

"""

import numpy as np
import matplotlib.pyplot as plt
import time

import xllim


# Parameters of the test
N_obs = 10000


# Creation of CPP model
inc = [
    26.543688820608587,
    59.70151372254296,
    49.75026883595537,
    22.25687341017779,
    29.37477052338286,
    45.92667930975748,
    0.8378149317275863,
    0.9539443924366908,
    50.6751036681658,
    30.939391754687584,
    45.35750723379789,
    77.4098063051496,
    8.550137962131798,
    78.75041993019249,
    57.21323807619601,
    64.45682060312284,
    61.27808641039031,
    29.87156688024389,
    58.23236626065648,
    0.46252284502360075,
    61.414629361746286,
    68.38870017628031,
    26.738228752953002,
    62.23719667072418,
    37.97518925487721,
    11.71217249013775,
    44.21083972798597,
    54.80504984567808,
    6.681918781680803,
    77.75380284700469,
    18.248735054007334,
    80.74809445945607,
    27.883083585044094,
    30.667505907879136,
    23.753533303937083,
    79.98537185251999,
    57.116427002329445,
    11.093398071099992,
    21.05442110717268,
    39.39417817217474,
    67.3088045536496,
    45.67330930572437,
    66.22342831254535,
    16.299202757933486,
    23.17100000877112,
    33.81214522345879,
    5.674512543132457,
    46.31197332927556,
    4.425076559963095,
    43.39587018527353,
]
eme = [
    57.82353475394024,
    75.1388961438802,
    53.34082125952007,
    59.16388136864589,
    56.82524519395523,
    13.861455932687909,
    61.01513462943814,
    0.08022416723222348,
    45.20326439963487,
    49.75804377781624,
    47.57479946505285,
    40.83414539871288,
    42.582107181671304,
    19.543259862080465,
    79.42231822683355,
    5.608973502992881,
    8.519518228991334,
    14.653496103650852,
    83.62967972867986,
    63.630581768018686,
    50.97355037385619,
    43.39685725973246,
    75.09920079639205,
    64.31108516343195,
    25.878527299490525,
    59.95252508789103,
    7.817590072337685,
    63.917416324181204,
    49.224108815946195,
    11.8651829920254,
    52.905039469008344,
    40.1809318771301,
    18.01508428188126,
    26.55789502901616,
    58.87354290360373,
    27.541622847565314,
    27.640845197335114,
    7.317915528927937,
    27.26977361165818,
    43.424177588459685,
    50.72108586970755,
    21.515785742893073,
    58.810741521507,
    74.65446772899669,
    65.1588001626688,
    57.74835037606868,
    40.286095753565505,
    80.72344684958227,
    46.045282404078506,
    31.16502580002508,
]
phi = [
    56.860734390493455,
    73.9693641883178,
    37.73213502869909,
    115.17115948730651,
    173.9808170538202,
    27.381659255810384,
    145.5789039136927,
    161.34338905048256,
    69.03017040464387,
    42.873394998955916,
    51.61878013029947,
    18.286412087024594,
    94.0626977818526,
    14.107568014786057,
    63.49011301889791,
    51.18375024794363,
    133.34658272658174,
    157.10530541937266,
    146.0687864919587,
    166.71021816930005,
    175.97262739732273,
    77.64637076284835,
    113.89679858896427,
    90.10458760238879,
    178.93891790686803,
    139.34645381672362,
    103.98331202999472,
    153.69553818327597,
    35.75205008184436,
    172.68559130787182,
    112.44694565309588,
    66.94556052034761,
    142.26012240479503,
    135.8978116867964,
    144.40908529405644,
    21.29090671425231,
    103.1093776827147,
    44.744140970189484,
    175.96850407152732,
    59.46378165314184,
    71.48685119783887,
    144.2711967755729,
    113.06470564914261,
    26.905165233458316,
    123.66223897723546,
    5.7269090111509335,
    157.16123545573817,
    76.01843764637619,
    112.82685894796201,
    117.10375378471507,
]
geometries = np.array([inc, eme, phi])

scalingCoeffs = [1.0, 1.5, 1.5, 1.5, 1.5]
offset = [0, 0, 0.2, 0, 0]
variant = "5p"
cppModel = xllim.ShkuratovModel(geometries, variant, scalingCoeffs, offset)


# Creation of Python model
pythonModel = xllim.ExternalPythonModel("ShkuratovModel5p", "ShkuratovModel5pPython", "../dataRef/externalPythonModels")

x_obs = np.zeros((N_obs, pythonModel.getDimensionX()))

for i in range(N_obs):
    for j in range(pythonModel.getDimensionX()):
        x_obs[i, j] = 0.4 * np.sin(2.0 * np.pi * i / N_obs + (j * np.pi / 4.0)) + 0.5

obs_list = [10, 100, 500, 1000, 5000, 10000]
cpp_time_list = []
python_time_list = []

for nb_obs in obs_list:
    ts = time.time()
    for i in range(nb_obs):
        y_pred = cppModel.F(x_obs[i])
    cpp_time_list.append(time.time() - ts)

    ts = time.time()
    for i in range(nb_obs):
        y_pred = pythonModel.F(x_obs[i])
    python_time_list.append(time.time() - ts)

plt.figure()
plt.plot(obs_list, cpp_time_list, "r", label="CPP")
plt.plot(obs_list, python_time_list, "b", label="Python")
plt.legend()
plt.xlabel("Number of observation calculatd")
plt.ylabel("Computation time (sec)")
plt.title("Comparison CPP/Python implementation of F(x) computation time")
plt.show()
