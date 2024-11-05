import numpy as np
import xllim as lib
import numpy.matlib
import time
import pickle
import os
import logging

logging.getLogger().setLevel(logging.INFO)


################## Global parameters ####################

gamma_type = "Full"
sigma_type = "Diag"
seed = 12345678
verbose = 1

# initialisation parameters
gllim_em_iteration = 10
gllim_em_floor = 1e-12
gmm_kmeans_iteration = 1
gmm_em_iteration = 1
gmm_floor = 1e-12
nb_experiences = 3

# training parameters
train_max_iteration = 100
train_ratio_ll = 1e-5
train_floor = 1e-12


K, D, L, N = 5, 33, 10, 1000

# ! Hybrid model
n_hidden_variables = 0
L_t = L - n_hidden_variables
L_w = n_hidden_variables

X_gen = np.random.rand(L_t, N)
Y_gen = np.random.rand(D, N)


################## setters ####################
# ! TODO : put this section inside the gllim model loop
# ! specific Gamma and Sigma are required
gllim = lib.GLLiM(
    K, D, L, gamma_type="full", sigma_type="diag", n_hidden_variables=n_hidden_variables
)

Pi = np.ones(K) * 1 / K
A = np.ones((K, D, L))
C = np.ones((K, L)) * 2
Gamma = np.ones((K, L, L)) * 0.2
B = np.ones((K, D)) * 3
Sigma_diag = np.ones((K, D)) * 0.3
gllim.setParamPi(Pi)
gllim.setParamA(A)
gllim.setParamC(C)
gllim.setParamGamma(Gamma)
gllim.setParamB(B)
gllim.setParamSigma(Sigma_diag)

new_theta = lib.GLLiMParameters(K, D, L, "full", "diag")
Gamma = np.ones((K, L, L)) + np.random.rand(K, L, L) * 0.1
Sigma = np.ones((K, D)) + np.random.rand(K, D) * 0.1
for k in range(K):
    Gamma[k] += np.eye(L) * 0.1
    Gamma[k] = np.dot(Gamma[k], Gamma[k].T) * 0.001
    # Sigma[k] += np.eye(D) * 0.1
    # Sigma[k] = np.dot(Sigma[k], Sigma[k].T) * 0.001
new_theta.Pi = Pi
new_theta.A = A * 3
new_theta.C = C * 2
new_theta.Gamma = Gamma
new_theta.B = B * 2
new_theta.Sigma = Sigma

gllim.setParams(new_theta)

# Test pickle/unpickle gllimParameters
with open('gllimParameters_pickle_test.file', 'wb') as f:
    pickle.dump(new_theta, f)
    f.close()

with open('gllimParameters_pickle_test.file', 'rb') as f:
    new_theta_unpickled = pickle.load(f)
    f.close()

os.remove('gllimParameters_pickle_test.file') # delete the generated file

print(np.all(new_theta.Pi == new_theta_unpickled.Pi))
print(np.all(new_theta.A == new_theta_unpickled.A))
print(np.all(new_theta.B == new_theta_unpickled.B))
print(np.all(new_theta.C == new_theta_unpickled.C))
print(np.all(new_theta.Gamma == new_theta_unpickled.Gamma))
print(np.all(new_theta.Sigma == new_theta_unpickled.Sigma))


################## loop on GLLiM models ####################

covariance_type_list = ["full", "diag", "iso"]

for gamma_type in covariance_type_list:
    for sigma_type in covariance_type_list:

        gllim = lib.GLLiM(K, D, L, gamma_type, sigma_type)

        ################## getters ####################
        gllim.getDimensions()
        gllim.getConstraints()
        gllim.getParams()
        gllim.getParamPi()
        gllim.getParamA()
        gllim.getParamC()
        gllim.getParamGamma()
        gllim.getParamB()
        gllim.getParamSigma()

        gllim.initialize(
            X_gen,
            Y_gen,
            gllim_em_iteration,
            gllim_em_floor,
            gmm_kmeans_iteration,
            gmm_em_iteration,
            gmm_floor,
            nb_experiences,
            seed,
            verbose,
        )
        gllim.train(
            X_gen, Y_gen, train_max_iteration, train_ratio_ll, train_floor, verbose
        )

        theta_star = gllim.getInverse()

        x = np.random.rand(L, 40)
        x_incertitudes = np.random.rand(L) * 0.02
        prediction_direct_results = gllim.directDensities(x, x_incertitudes)

        y = np.random.rand(D, 40)
        y_incertitudes = np.random.rand(D, 1) * 0.01
        tic = time.time()
        prediction_results = gllim.inverseDensities(y, y_incertitudes)
        print("Time One inversion = {}".format(time.time() - tic))

        y_incertitudes_temp = np.matlib.repmat(y_incertitudes, 1, 40)
        y_incertitudes_mat = np.copy(y_incertitudes_temp)

        tic = time.time()
        prediction_results_all = gllim.inverseDensities(y, y_incertitudes_mat, 1)
        print("Time Multi inversion = {}".format(time.time() - tic))


################## GLLiM Full model JGMM training ####################

kmeans_iteration = 10
em_iteration = 100
gllim = lib.GLLiM(K, D, L, "full", "full")

gllim.initialize(
    X_gen,
    Y_gen,
    gllim_em_iteration,
    gllim_em_floor,
    gmm_kmeans_iteration,
    gmm_em_iteration,
    gmm_floor,
    nb_experiences,
    seed,
    verbose,
)

gllim.trainJGMM(
    X_gen, Y_gen, kmeans_iteration, em_iteration, train_floor, verbose
)

theta_star = gllim.getInverse()

x = np.random.rand(L, 40)
x_incertitudes = np.random.rand(L) * 0.02
prediction_direct_results = gllim.directDensities(x, x_incertitudes)

y = np.random.rand(D, 40)
y_incertitudes = np.random.rand(D, 1) * 0.01
tic = time.time()
prediction_results = gllim.inverseDensities(y, y_incertitudes)
print("Time One inversion = {}".format(time.time() - tic))

y_incertitudes_temp = np.matlib.repmat(y_incertitudes, 1, 40)
y_incertitudes_mat = np.copy(y_incertitudes_temp)

tic = time.time()
prediction_results_all = gllim.inverseDensities(y, y_incertitudes_mat)
print("Time Multi inversion = {}".format(time.time() - tic))

K_merged = 2
merging_threshold = 1e-5

tic = time.time()
prediction_results_all = gllim.inverseDensities(y, y_incertitudes_mat, K_merged, merging_threshold, 1)
print("Time inversion with merging algorithm = {}".format(time.time()-tic))

series = prediction_results_all.mergedGMM.means
tic = time.time()
permutations = lib.regularize(series)
print("Time regularization of merged centers = {}".format(time.time()-tic))

# this sould returns an error
# [xllim] ERROR   : This method is only available with gamma_type = 'full' and sigma_type = 'full'
gllim = lib.GLLiM(K, D, L, "full", "diag")
gllim.trainJGMM(
    X_gen, Y_gen, kmeans_iteration, em_iteration, train_floor, verbose
)
