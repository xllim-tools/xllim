import os.path
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import kernelo as ker


def compute_reconstruction_error(reconstruction, observation):
    return np.linalg.norm(observation - reconstruction) / np.linalg.norm(observation)



# Create "physical" model
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path+"/pytest/models")
physical_model = ker.ExternalModelConfig("RPVModel", "rpv_model", dir_path + "/pytest/models").create()

# Create StatModel
covariances = np.random.uniform(0, 0.0001, physical_model.get_D_dimension())
stat_model = ker.GaussianStatModelConfig("sobol", physical_model, covariances, 12345).create()

# Create GLLIM model, including its initialization and training configuration
learningConfig = ker.EMLearningConfig(200, 1e-5, 1e-12)
initConfig = ker.MultInitConfig(seed=123456789, nb_iter_EM=10, nb_experiences=10, gmmLearningConfig=ker.GMMLearningConfig(15, 10, 1e-12))
gllim= ker.GLLiM(physical_model.get_D_dimension(), physical_model.get_L_dimension(), 50, "Full", "Diag", initConfig, learningConfig)

gllim_model_file_name = "pytest/savedFiles/" + "gllim_RPV_example.file"
if not os.path.isfile(gllim_model_file_name):  # This is very useful to not train Gllim model at every simulation

    # Initialize and train GLLIM model
    print("Generating dataset")
    x_gen, y_gen = stat_model.gen_data(50000)
    print("Initializing GLLIM model")
    gllim.initialize(x_gen, y_gen)
    print("Training model")
    gllim.train(x_gen, y_gen)

    gllim_parameters = gllim.exportModel() # you can export your gllim model parameters
    with open(gllim_model_file_name, "wb") as f:
        pickle.dump(gllim_parameters, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print("GLLIM model saved")
else:
    with open(gllim_model_file_name, "rb") as f:
        gllim_parameters = pickle.load(f)
        gllim.importModel(gllim_parameters)
    print("GLLIM model loaded")


### Load your data ###
print("Generating observations for example")
n_observations = 100
x_obs = np.zeros((n_observations, physical_model.get_L_dimension()))
y_obs = np.zeros((n_observations, physical_model.get_D_dimension()))
y_obs_noised = np.zeros((n_observations, physical_model.get_D_dimension()))
y_obs_noise = np.zeros((n_observations, physical_model.get_D_dimension()))

for i in range(n_observations):
    for j in range(physical_model.get_L_dimension()):
        x_obs[i, j] = 0.4 * math.sin(2.*math.pi*i/n_observations + (j * math.pi/4.)) + 0.5
for i in range(n_observations):
    y_obs[i] = physical_model.F(x_obs[i])
    # Add noise for each Y component
    for j in range(physical_model.get_D_dimension()):
        y_obs_noise[i][j] = (y_obs[i][j]/1000.) * np.random.normal(0, math.pow(y_obs[i][j]/1000., 2), 1)
y_obs_noised = y_obs + y_obs_noise


### Predictions ###
nb_centers = 2

# Gllim
predicator = ker.PredictionConfig(nb_centers, nb_centers, 1e-10, gllim).create()
prediction_means = [[] for i in range(nb_centers+1)] # list[0] is the list for mean ; list[1] is the list for center1 ; list[2] is the list for center2
prediction_reconstruction_error = [[] for i in range(nb_centers+1)]
mean_prop_laws = [] # Proposition laws for IS and IMIS
center_prop_laws = []
print("Computing predictions")
for i in range(n_observations):
    prediction = predicator.predict(y_obs_noised[i], y_obs_noise[i])
    x_pred = prediction.meansPred.mean
    y_pred = physical_model.F(x_pred)
    prediction_means[0].append(x_pred)
    prediction_reconstruction_error[0].append(compute_reconstruction_error(y_pred, y_obs_noised[i]))
    mean_prop_law = ker.GaussianMixturePropositionConfig( # Proposition law for IS and IMIS
        prediction.meansPred.gmm_weights, 
        prediction.meansPred.gmm_means,
        prediction.meansPred.gmm_covs).create()
    mean_prop_laws.append(mean_prop_law)

    for center in range(1, nb_centers+1):
        x_pred = prediction.centersPred.means[:, center-1]
        y_pred = physical_model.F(x_pred)
        prediction_means[center].append(x_pred)
        prediction_reconstruction_error[center].append(compute_reconstruction_error(y_pred, y_obs_noised[i]))
        center_prop_law = ker.GaussianRegularizedPropositionConfig(
            prediction.centersPred.means[:, center-1],
            prediction.centersPred.covs[center-1, :, :]).create()
        center_prop_laws.append(center_prop_law)

# Gllim-IS
sampler_is = ker.ImportanceSamplingConfig(1000, stat_model).create()
is_means = [[] for i in range(nb_centers+1)]
is_reconstruction_error = [[] for i in range(nb_centers+1)]
print("Computing IS")
for i in range(n_observations):
    result = sampler_is.execute(mean_prop_laws[i], y_obs_noised[i], y_obs_noise[i])
    x_pred = result.mean
    y_pred = physical_model.F(x_pred)
    is_means[0].append(x_pred)
    is_reconstruction_error[0].append(compute_reconstruction_error(y_pred, y_obs_noised[i]))

    for center in range(1, nb_centers+1):
        result = sampler_is.execute(center_prop_laws[i], y_obs_noised[i], y_obs_noise[i])
        x_pred = result.mean
        y_pred = physical_model.F(x_pred)
        is_means[center].append(x_pred)
        is_reconstruction_error[center].append(compute_reconstruction_error(y_pred, y_obs_noised[i]))

# Gllim-IMIS
sampler_imis = ker.ImisConfig(100, 50, 18, stat_model).create()
imis_means = [[] for i in range(nb_centers+1)]
imis_reconstruction_error = [[] for i in range(nb_centers+1)]
print("Computing IMIS")
for i in range(n_observations):
    result = sampler_imis.execute(mean_prop_laws[i], y_obs_noised[i], y_obs_noise[i])
    x_pred = result.mean
    y_pred = physical_model.F(x_pred)
    imis_means[0].append(x_pred)
    imis_reconstruction_error[0].append(compute_reconstruction_error(y_pred, y_obs_noised[i]))

    for center in range(1, nb_centers+1):
        result = sampler_imis.execute(center_prop_laws[i], y_obs_noised[i], y_obs_noise[i])
        x_pred = result.mean
        y_pred = physical_model.F(x_pred)
        imis_means[center].append(x_pred)
        imis_reconstruction_error[center].append(compute_reconstruction_error(y_pred, y_obs_noised[i]))


### Plot results ###
print("Plotting results")
fig, axs = plt.subplots(2, 2, num="Graphs : RPV model Example")
fig.suptitle("Parameters estimation and reconstruction error", fontsize=16)
n_observations_list = np.arange(1, n_observations + 1)

for l in range(physical_model.get_L_dimension()):
    i = 0 if (l<2) else 1
    j = l if (l<2) else l-2

    axs[i,j].plot(n_observations_list, x_obs[:,l], 'k-', label='observation')
    axs[i,j].plot(n_observations_list, [x[l] for x in prediction_means[0]], 'b.', label='prediction mean')
    axs[i,j].plot(n_observations_list, [x[l] for x in is_means[0]], 'r.', label='is mean')
    axs[i,j].plot(n_observations_list, [x[l] for x in imis_means[0]], 'g.', label='imis mean')
    # axs[i,j].plot(n_observations_list, [x[l] for x in prediction_means[1]], 'b.', label='prediction center_1')
    # axs[i,j].plot(n_observations_list, [x[l] for x in is_means[1]], 'r.', label='is center_1')
    # axs[i,j].plot(n_observations_list, [x[l] for x in imis_means[1]], 'g.', label='imis center_1')
    # axs[i,j].plot(n_observations_list, [x[l] for x in prediction_means[2]], 'b.', label='prediction center_2')
    # axs[i,j].plot(n_observations_list, [x[l] for x in is_means[2]], 'r.', label='is center_2')
    # axs[i,j].plot(n_observations_list, [x[l] for x in imis_means[2]], 'g.', label='imis center_2')
    axs[i,j].set_xlabel("Observations")
    axs[i,j].set_title('X_'+ str(l))
    axs[i,j].legend()

# axs[1,1].plot(n_observations_list, prediction_reconstruction_error[0], 'b--', label='prediction mean')
# axs[1,1].plot(n_observations_list, is_reconstruction_error[0], 'r--', label='is mean')
axs[1,1].plot(n_observations_list, imis_reconstruction_error[0], 'g--', label='imis mean')
# axs[1,1].plot(n_observations_list, prediction_reconstruction_error[1], 'b--', label='prediction center_1')
# axs[1,1].plot(n_observations_list, is_reconstruction_error[1], 'r--', label='is center_1')
# axs[1,1].plot(n_observations_list, imis_reconstruction_error[1], 'g--', label='imis center_1')
# axs[1,1].plot(n_observations_list, prediction_reconstruction_error[2], 'b--', label='prediction center_2')
# axs[1,1].plot(n_observations_list, is_reconstruction_error[2], 'r--', label='is center_2')
# axs[1,1].plot(n_observations_list, imis_reconstruction_error[2], 'g--', label='imis center_2')
axs[1,1].set_xlabel("Observations")
# axs[1,1].set_yscale('log')
axs[1,1].set_title('Reconstruction error')
axs[1,1].legend()
    
plt.show()