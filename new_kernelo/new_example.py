#############################################################################################################
#                                   Rework of Kernelo-GLLiM API                                             #
#############################################################################################################

import numpy as np
import kernelo as ker


### Functional model
physical_model = ker.XModel()
# .F(), .get_L, get_D, to_physic(), from_physic() OK


### StatModel ??
stat_model = ker.XStatModel("sobol", physical_model, covariances, seed)
x_gen, y_gen = stat_model.gen_data(50000)
# stat_model.gen_data()
# stat_model.density_X_Y() => used by ISTarget.h in target_law_densuty()..... useless ?

# Idée
# Direct method: ker.generate_data(N, "sobol", physical_model, covariances, seed)


### GLLiM solver
learningConfig = ker.XLearningConfig(args,...)
initConfig = ker.XInitConfig(seed, nb_iter_EM, nb_experiences, gmmLearningConfig=ker.GMMLearningConfig(args,...))
gllim= ker.GLLiM(physical_model.get_D_dimension(), physical_model.get_L_dimension(), 50, "Full", "Diag", initConfig, learningConfig)
gllim.initialize(x_gen, y_gen)
gllim.train(x_gen, y_gen)
gllim_parameters = gllim.exportModel()
# configs ??
# On fait des structure complexe dans Kernelo, tout ça pour que après dans Planet-GLLiM avoir besoin de faire des parsers de toute cette complexité
# ker.GLLiM() class     => .initialize(), .train() and .exportModel(), .importModel(), getInverse(), directLog(), InverseLog()         OK but unused in PlanetGLLiM
#                       => used in predictor class

# Idée
# gllim = ker.GLLiM(D,L,K, constraints) -> création de la classe (de theta)
# gllim.initialize(x_gen, y_gen, initConfig args) -> None + print some insights
# gllim.train(x_gen, y_gen, trainConfig args) -> None + print some insights
# gllim.getParams() -> theta as dict with A, b, c, Pi...
#   gllim.getParams("A", "b") -> "A", "b"    
# gllim.setParams(theta or A, b, Pi,Sigma...) -> None
# gllim.forwardDensities(x, verbose, log=False,..)) -> Posterior mean estimates E[yn|xn;θ] and gmm ?
# gllim.inverseDensities(y) -> Prior mean estimates E[xn|yn;θ] and gmm ?
# NEW gllim.insights() -> GLLiM, initialize and train args, learning time, memory use, Loglikelihood!!, gmm ?, center pred ?

# Ou bien ker.gllim_train_wrapper(D,L,K,constraints, initconfig, trainconfig) -> trained gllim   // équivalent à GLLiM() + initialize + train
# 



### Predictions
predicator = ker.PredictionConfig(nb_centers, nb_centers, 1e-10, gllim).create()
# .predict() et .regularize()
# predictor.predict(y, y_noise) ->
# {
#    meansPred: {
#        vec mean; /**< The mean of the GMM which stands for the prediction*/
#        vec variance; /**< The variance of the prediction*/
#        vec gmm_weights; /**< The weights of the components of the GMM*/
#        mat gmm_means;/**< The means of each component in the GMM*/
#        cube gmm_covs;/**< The covariance matrices of each component in the GMM*/
#    },
#    centersPred: {
#        vec weights; /**< The weights of the centers*/
#        mat means; /**< The centers that stands for the predictions*/
#        cube covs; /**< The covariance matrices of the centers*/
#    };
# }

for i in range(n_observations):
    prediction = predicator.predict(y_obs_noised[i], y_obs_noise[i])
    x_pred = prediction.meansPred.mean
    y_pred = physical_model.F(x_pred)
    compute_reconstruction_error(y_pred, y_obs_noised[i])
    mean_prop_law = ker.GaussianMixturePropositionConfig( # Proposition law for IS and IMIS
        prediction.meansPred.gmm_weights, 
        prediction.meansPred.gmm_means,
        prediction.meansPred.gmm_covs).create()

    for center in range(1, nb_centers+1):
        x_pred = prediction.centersPred.means[:, center-1]
        y_pred = physical_model.F(x_pred)
        compute_reconstruction_error(y_pred, y_obs_noised[i])
        center_prop_law = ker.GaussianRegularizedPropositionConfig(
            prediction.centersPred.means[:, center-1],
            prediction.centersPred.covs[center-1, :, :]).create()

# En fait cette class est un wrapper + analyse des résultats gllim. On va ajouter une partie insights dans gllim.
# A part la prediction correspondant à gllim.inverseDensities(y), il y a qq calculs moyenne et variance et les trucs par rapport aux centroïdes
# intérêt de cette classe ?
# on pourrait simplement faire un gllim.inverseDensities(y, y_noise, k_merged, k_pred_mean, threshold?) -> Prior mean estimates E[xn|yn;θ], variance, mean gmm, centers gmm
# intérêt pour Sampling ? Aucun
# NEW ker.FunctionalModel.reconstruction_error(X_pred, Y_obs)




### Sampling
sampler_is = ker.XSamplingConfig(args, stat_model).create()
# Seule méthode: .execute(prop, y, y_noise) -> {means, variance, IS_diagnostic{nb_effective_sample, effective_sample_size, qn}}
for i in range(n_observations):
    result = sampler_is.execute(mean_prop_laws[i], y_obs_noised[i], y_obs_noise[i])
    x_pred = result.mean
    y_pred = physical_model.F(x_pred)
    compute_reconstruction_error(y_pred, y_obs_noised[i])
    for center in range(1, nb_centers+1):
        result = sampler_is.execute(center_prop_laws[i], y_obs_noised[i], y_obs_noise[i])
        x_pred = result.mean
        y_pred = physical_model.F(x_pred)
        compute_reconstruction_error(y_pred, y_obs_noised[i])


# Le lien avec statModel() est utile juste pour la target_log_density (density_X_Y) qui utilise la fonctionnel F
# Donc on peut se passer de StatModel, la diff entre BasicStatModel et DependentStatModel c'est juste la covariance en vecteur ou en float. Très petite modif dans density_X_Y
# propositions/ et target/ ne servent à rien !
# proposition c'est juste une gmm_full armadillo défini par prop (GLLiM) avec vite fait une méthode associé (log_density c'est dans utils/helpers)
# target c'est juste density_X_Y de statModel() (Hapke/Shkuratov..). Il faut la fonctionnel F ! écrire une méthode target_log_density
# Faire un dossier sampling/ avec des méthodes de sampling dans un seul fichier, sans constructeur (comme Helpers) et contenant les méthodes de sampling
# ker.XSampling(prop, y, y_obs, sample_args, functional_model) -> {means, variance, IS_diagnostic{nb_effective_sample, effective_sample_size, qn}}
# faire un sampling.hpp et sampling.cpp qui fait le wrapping des méthodes. Mais chaque méthode est détaillé dans un sous répertoire

