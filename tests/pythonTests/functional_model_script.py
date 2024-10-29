#!/usr/bin/env python
# coding: utf-8

#####################
# IMPORTING PACKAGE #
#####################
# In[1]:
# import newkernelo as lib
import xllim as lib
import numpy as np
import time
import os.path
import json
import logging
logging.getLogger().setLevel(logging.INFO)



##############
# TEST MODEL #
##############
# In[4]:
model = lib.TestModel()
e = np.exp(1)
x_test = np.zeros(4)
y_test = model.F(x_test)
true_result = np.array([4+2*e, 0.5, e, 3, 0.2, -0.5, -0.2-e, 2*e-1, -0.7])*0.5
print("y_test")
print(y_test)
print("true_result")
print(true_result)

assert (y_test == true_result).all()


# In[5]:
x = np.ones(4)
z = model.F(x)
print(z)
print(x)
print(x.shape)
model.toPhysic(x) # does nothing
print(x)
print(x.shape)
w = model.toPhysic(x) # return vec size-like : w.shape() = (5,1)
print(x)
print(w)
print(w.shape)
print("=========== From physic ==========")
print(x)
model.fromPhysic(x)
print(x)
y = model.fromPhysic(x)
print(x)
print(y)
z = model.fromPhysic(y)
print(x)
print(y)
print(z)


#############
# SHKURATOV #
#############

# In[6]:

print(os.getcwd())
with open('../dataRef/Shkuratov5p_data_ref.json', 'r') as f:
    data = json.load(f)

D = 50
row_size = D
col_size = 3

scalingCoeffs = [1.0,1.5,1.5,1.5,1.5]
offset = [0,0,0.2,0,0]


# # Create JSON file with geometries
geometries = np.empty((row_size,col_size))
var_geom = ["inc", "eme", "phi"]
for j in range(3):
    i=0
    for v in data[var_geom[j]]:
        geometries[i,j] = v # geometries.shape = (D,3)
        i+=1
# print(geometries)
print(geometries.shape)

# geom = {'geometries': [[geometries[j,:].tolist()] for j in range(3)]
# }
# with open('geometries_shkuratov.json', 'w') as fp:
#     json.dump(geom, fp)

## INTEGRATION au code C++
variant = "5p"
physicalModel = lib.ShkuratovModel(geometries, variant, scalingCoeffs, offset)


### TEST
N = 100
L = 5
variables = ["an", "mu1", "nu", "m", "mu2"]
photometries = np.empty((L,N))

# Read photometries
for l in range(L):
    for n in range(N):
        photometries[l,n] = (float(data[variables[l]][n]) - offset[l]) / scalingCoeffs[l]
        n+=1


# Read expected results
expected_results = np.empty((D,N))
n=0
for n in range(N):
    for d in range(D):
        expected_results[d,n] = float(data["y"][n][d])



# compute results from the model
# result = np.empty((D,))
assert_list = []
for n in range(N):
    result = physicalModel.F(photometries[:,n])
    assert_list.append(np.allclose(expected_results[:,n], result, rtol=1e-8))

print(result.shape)
print(assert_list)
print(False in assert_list)
print(True in assert_list)

print(expected_results[:10,n])
print(result[:10])


# ## Hapke

# In[7]:


with open('../dataRef/Hapke6p_geom70_data_ref.json') as json_file:
    data = json.load(json_file)
    expected_results = np.array(data["data_ref"]["synthetic_dataset"]['Y'])
    geometries = np.array(data["data_ref"]['geometries'])
    photometries = np.array(data["data_ref"]["synthetic_dataset"]['X'])
print(geometries.shape)
print(photometries.shape)
print(expected_results.shape)


hapkeModel = lib.HapkeModel(geometries, "2002", "six", 30.0,1,0)
print(hapkeModel.getDimensionY())
print(hapkeModel.getDimensionX())

# TEST
N = 10
assert_list = []
for n in range(N):
    result = hapkeModel.F(photometries[n])
    assert_list.append(np.allclose(expected_results[n], result, rtol=1e-8))

print(result.shape)
print(assert_list)
print(False in assert_list)
print(True in assert_list)

print(expected_results[n])
print(result)


# In[8]:


with open('../dataRef/Hapke4p_geom70_data_ref.json') as json_file:
    data = json.load(json_file)
    expected_results = np.array(data["data_ref"]["synthetic_dataset"]['Y'])
    geometries = np.array(data["data_ref"]['geometries'])
    photometries = np.array(data["data_ref"]["synthetic_dataset"]['X'])
print(geometries.shape)
print(photometries.shape)
print(expected_results.shape)


hapkeModel = lib.HapkeModel(geometries, "2002", "four", 30.0,1,0)
print(hapkeModel.getDimensionY())
print(hapkeModel.getDimensionX())

# TEST
N = 10
assert_list = []
for n in range(N):
    result = hapkeModel.F(photometries[n])
    assert_list.append(np.allclose(expected_results[n], result, rtol=1e-8))

print(result.shape)
print(assert_list)
print(False in assert_list)
print(True in assert_list)

print(expected_results[n,:10])
print(result[:10])

### NOTE : le expected_results[n][2] et result[2] sont différents.... mais les autres sont égaux. à voir...


# In[9]:


print(hapkeModel.getDimensionY())
print(hapkeModel.getDimensionX())
x = np.ones(hapkeModel.getDimensionX()) / 10

print(x)
print(x.shape)
hapkeModel.toPhysic(x) # does nothing
print(x)
print(x.shape)
w = hapkeModel.toPhysic(x) # return vec size-like : w.shape() = (5,1)
print(x)
print(w)
print(w.shape)
print("=========== From physic ==========")
print(x)
hapkeModel.fromPhysic(x)
print(x)
y = hapkeModel.fromPhysic(x)
print(x)
print(y)
z = hapkeModel.fromPhysic(y)
print(x)
print(y)
print(z)


# ## External model

# In[10]:


externalPythonModel = lib.ExternalPythonModel("ShkuratovModel5p", "ShkuratovModel5pPython", "../dataRef/")

print(externalPythonModel.getDimensionY())
print(externalPythonModel.getDimensionX())


# In[11]:


x = np.ones(externalPythonModel.getDimensionX())

print(x)
print(x.shape)
print("=========== To physic ==========")
externalPythonModel.toPhysic(x) # does nothing
print(x)
print(x.shape)
w = externalPythonModel.toPhysic(x) # return vec size-like : w.shape() = (5,1)
print(x)
print(w)
print(w.shape)
print("=========== From physic ==========")
externalPythonModel.fromPhysic(x)
print(x)
y = externalPythonModel.fromPhysic(x)
print(x)
print(y)
z = externalPythonModel.fromPhysic(y)
print(x)
print(y)
print(z)


# In[12]:


with open('../dataRef/Shkuratov5p_data_ref.json', 'r') as f:
    data = json.load(f)

D = 50
row_size = D
col_size = 3

scalingCoeffs = [1.0,1.5,1.5,1.5,1.5]
offset = [0,0,0.2,0,0]



### TEST
N = 3
L = 5
variables = ["an", "mu1", "nu", "m", "mu2"]
photometries = np.empty((L,N))

# Read photometries
for l in range(L):
    for n in range(N):
        photometries[l,n] = (float(data[variables[l]][n]) - offset[l]) / scalingCoeffs[l]
        n+=1


# Read expected results
expected_results = np.empty((D,N))
n=0
for n in range(N):
    for d in range(D):
        expected_results[d,n] = float(data["y"][n][d])


# compute results from the model
# result = np.empty((D,))
assert_list = []
for n in range(N):
    result = externalPythonModel.F(photometries[:,n])
    assert_list.append(np.allclose(expected_results[:,n], result, rtol=1e-8))

print(result.shape)
print(assert_list)
print(False in assert_list)
print(True in assert_list)

print(expected_results[:10,n])
print(result[:10])


# ## dataGeneration

# ### New FunctionalModel.dataGen()

# In[13]:


x_gen, y_gen = model.genData(10, "sobol", np.ones(model.getDimensionY()), 1234)


# In[14]:


print(x_gen.shape)
print(y_gen.shape)


# In[15]:


x_gen, y_gen = model.genData(40, "sobol", 0.1, 1234)


# In[16]:


print(x_gen.shape)
print(y_gen.shape)


# ### Importance sampling

# In[17]:


K = 3
L = 4

weight = np.ones(K) * 1/K
# weight = np.array([0.2, 0.1, 0.1, 0.2, 0.4])
# mean = np.ones((4,5))*0.77
# mean[:,1] *= 3
mean = np.random.rand(L,K)
cube =  np.ones((L,L,K))*0.01
cube += np.random.rand(L,L,K) *0.1
for k in range(cube.shape[2]):
    cube[:,:,k] += np.eye(L) * 0.1
    cube[:,:,k] = np.dot(cube[:,:,k], cube[:,:,k].T) * 0.001

proposition_gmms = [(weight.T, mean, cube), (weight.T, mean*0.2, cube*0.2)] * 20
y = np.array(y_gen[:2]) # Note avec CARMA, on a l'erreur "this array cannot be borrow...." dans le cas où on met en argument un pointeur! Et donc la mémoire n'appartient pas à cette variable.
y = y_gen
y_err = y*0.001
covariance = np.ones(model.getDimensionY()) *0.003
N_0 = 1000
B = 500
J = 20


# In[18]:


print(cube[:,:,0].T)


# In[19]:


verbose = 0
tic = time.time()
results = model.importanceSampling(proposition_gmms, y, y_err, covariance, N_0, B, J, verbose)
tac = time.time()
time.sleep(1)
print("Time: {}".format(tac - tic))


# In[20]:


def compute_reconstruction_error(reconstruction, observation):
    return np.linalg.norm(observation - reconstruction) / np.linalg.norm(observation)


# In[23]:

print(results.predictions)
print(results.predictions_variance)


# In[24]:


print("New version:")
print(results.nb_effective_sample.shape)
print(results.nb_effective_sample)
print(results.effective_sample_size)
print(results.qn)
print(results.nb_effective_sample.shape)
print(results.effective_sample_size.shape)
print(results.qn.shape)

