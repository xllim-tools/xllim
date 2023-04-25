""" This file is an example of the kernelo module usage """

### Import module ###
"""Kernelo must first be installed following the readme file."""
import kernelo as ker
import numpy as np
import matplotlib.pyplot as plt
import os.path


### Create physical model ###
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/pytest/"
physical_model = ker.ExternalModelConfig("Unimodal", "unimodal", dir_path).create()

### Create statistical model ###
stat_model = ker.GaussianStatModelConfig("sobol", physical_model, 0.001, 123456).create()

### Create GLLIM model ###
learningConfig = ker.EMLearningConfig(200, 1e-5, 1e-12) # including its training configuration 
gmmLearningConfig=ker.GMMLearningConfig(15, 10, 1e-12)
initConfig = ker.MultInitConfig(123456, 10, 10, gmmLearningConfig) # including its initialization
gllim = ker.GLLiM(physical_model.get_D_dimension(), physical_model.get_L_dimension(), 50, "Full", "Diag", initConfig, learningConfig)

### Train GLLiM model ###
x_gen, y_gen = stat_model.gen_data(5000) # Generate synthetic data
gllim.initialize(x_gen, y_gen) # Initialize
gllim.train(x_gen, y_gen) # Train

### Configure observartions Y ###
""" Generate synthetic observations for example. Use your own observations y_obs = F(x_obs) """
n_obervations = 100
x_obs = np.zeros((n_obervations, physical_model.get_L_dimension()))
y_obs = np.zeros((n_obervations, physical_model.get_D_dimension()))
y_obs_noised = np.zeros((n_obervations, physical_model.get_D_dimension()))
y_obs_noise = np.zeros((n_obervations, physical_model.get_D_dimension()))
for i in range(n_obervations):
    for j in range(physical_model.get_L_dimension()):
        x_obs[i, j] = 0.4 * np.sin(2.*np.pi*i/n_obervations + (j * np.pi/4.)) + 0.51
for i in range(n_obervations):
    y_obs[i] = physical_model.F(x_obs[i])
    # Add noise for each Y component
    for j in range(physical_model.get_D_dimension()):
        y_obs_noise[i][j] = (y_obs[i][j]/1000.) * np.random.normal(0, pow(y_obs[i][j]/1000., 2), 1)
y_obs_noised = y_obs + y_obs_noise


### Prediction with GLLiM ###
predicator = ker.PredictionConfig(2, 2, 1e-10, gllim).create() # Create predicator
predictions = []
for i in range(n_obervations):
    predictions.append(predicator.predict(y_obs_noised[i], y_obs_noise[i])) # Predict on observations data
    ### TODO : bug à ce niveau là ....


### Apply IMIS to improve estimation ###
sampler_imis_1 = ker.ImisConfig(100, 50, 18, stat_model).create() # Create IMIS sampler
propositions = []
result_imis = []
for i in range(n_obervations):
    """ This is the type of proposition law for the prediction by the mean. Currently, only gaussian mixture model is implemented."""
    propositions.append(ker.GaussianMixturePropositionConfig(
            predictions[i].meansPred.gmm_weights, 
            predictions[i].meansPred.gmm_means,
            predictions[i].meansPred.gmm_covs).create())
    result_imis.append(sampler_imis_1.execute(propositions[i], y_obs_noised[i], y_obs_noise[i])) # Execute IMIS on synthetic data with proposition law


### Exploit results ###

x_list = np.arange(n_obervations)

fig, axs = plt.subplots(2, 2, num="Example of a GLLiM-IMIS estimation")
fig.suptitle("Analysis of the parameters estimation", fontsize=16)

for l in range(physical_model.get_L_dimension()):
    i = 0 if (l<2) else 1
    j = l if (l<2) else 0
    axs[i,j].plot(x_list, x_obs[:,l], 'b.', label='Synthetic data')
    axs[i,j].plot(x_list, [res.mean[l] for res in result_imis], 'r.', label='GLLiM-IMIS estimation')
    axs[i,j].set_xlabel("samples/observations")
    axs[i,j].set_ylabel("Y")
    axs[i,j].set_title('Parameter ' + l)
    axs[i,j].legend()
