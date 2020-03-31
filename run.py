import numpy as np
import kernelo as ker
import random
import json
import time
import my_python_objects as Obj

# Read geometries and photometries from Shkuratov test file
with open('test_shkuratov.json') as json_file:
    data = json.load(json_file)

    geom = np.array([data['inc'],
                     data['eme'],
                     data['phi']]).transpose() # column indexing

    photom = np.array([
        data['an'],
        data['mu1'],
        data['nu'],
        data['m'],
        data['mu2']]).transpose() # column indexing

scalling = [1.0,1.5,0.8,1.5,1.5]
offset = [0,0,0.2,0,0]
shkuratovModel = ker.ShkuratovModelConfig(geom, scalling, offset).create()

start_time = time.time()

# Calculate reflectances
Y = []

Y.append(shkuratovModel.F((photom[0] - offset)/scalling))

elapsed_time = (time.time() - start_time)
print("Execution time for Y = F(X) where D = 50, L= 5 and n = 10000 : " , elapsed_time)


y_test = np.array(data['y'])
print(Y)


# Create Hapke2002 Model
#myAdapter = ker.SixParamsHapkeAdapterConfig()

#myModel = ker.HapkeModelConfig("2002", myAdapter, geom, 30).create()

#myGenerator = ker.GaussianStatModelConfig("sobol", myModel, np.arange(6.0), 123456789).create()

#x_gen, y_gen = myGenerator.gen_data(1)
#print(y_gen)

# Start time
#start_time = time.time()

# Calculate reflectances
#y = myModel.F(photom[0])


#print(y)
# Calculate elapsed time for generating reflectances
#elapsed_time = (time.time() - start_time)
#print(elapsed_time)

# Write reflectances in file 'result.json'
#with open('result.json','w') as out_file:
 #   out_file.write(pd.Series({'y':y}).to_json())
