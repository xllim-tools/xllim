import numpy as np
import kernelo as ker
import random
import json
import time
import my_python_objects as Obj

# obj = ker.ExternalModelConfig("ExternalFunctionalModelExample", "MyFunctional","/home/reverse-proxy/CLionProjects/untitled").create()
# x = np.arange(3, dtype=np.double)
# print(obj.get_D_dimension())
# print(obj.get_L_dimension())
# y = obj.F(x)
# print(x)
# print(y)

#Read geometries and photometries from file 'test_hapke.json'
with open('test_hapke.json') as json_file:
    data = json.load(json_file)
    y = np.array(data['y'])
    geom = np.array([data['eme'], data['inc'], data['phi']]).transpose()
    photom = np.array([data['omega'], [x / 30 for x in data['theta0']], data['b'], data['c'], data['b0'], data['hh']]).transpose()

mySixParamsAdapter = ker.FourParamsHapkeAdapterConfig(0.5, 0.5)
my02Model = ker.HapkeModelConfig("2002", mySixParamsAdapter, geom[3:], 30).create()
myStatModel_2 = ker.DependentGaussianStatModelConfig("sobol", my02Model, 10000, 12345).create()
x_gen, y_gen = myStatModel_2.gen_data(500)
y_test = my02Model.F(x_gen[0,:])
print(y_test)

learningConfig = ker.EMLearningConfig(30,2.0,1e-08)
initconfig = ker.MultInitConfig(123456789, 5, 3, ker.GMMLearningConfig(10,5,1e-08))
gllim = ker.GLLiM(47,4,10,"Diag", "Diag", initconfig, learningConfig)
print(x_gen[0,:])
print(y_gen[0,:])
gllim.initialize(x_gen, y_gen)
print("done")
gllim.train(x_gen, y_gen)

predictor = ker.PredictionConfig(2, 5, 0.01, gllim).create()
result1 = predictor.predict(y_test, np.zeros(47))
print(result1.centersPred.means)
print(result1.meansPred.mean)
# print(result1.centersPred.means)
# print(result1.centersPred.weights)
# print(result1.centersPred.covs)

#result2 = predictor.predict(y[1,:], np.zeros(50))
#result3 = predictor.predict(y[2,:], np.zeros(50))
# print(result.L)
# print(result.K_merged)
# print(result.means)
# print(result.variances)
# print(result.centers_means)
#print(result.centers_variances)
# series = np.array([result1.centers_means.transpose(),result2.centers_means.transpose(),result3.centers_means.transpose()])
# print(series.shape)
# print(predictor.regularize(series))

cov_is = np.zeros(47)

proposition = ker.GaussianMixturePropositionConfig(result1.meansPred.gmm_weights, result1.meansPred.gmm_means, result1.meansPred.gmm_covs).create()
#proposition = ker.GaussianRegularizedPropositionConfig(result1.centersPred.means[:,0], result1.centersPred.covs[0,:,:]).create()
sampler = ker.ImportanceSamplingConfig(1000, myStatModel_2).create()
res_is = sampler.execute(proposition, y_test, cov_is)
print(res_is.mean)
print(x_gen[0,:])







