import numpy as np
import kernelo as ker
import json
import time

# Read geometries and photometries from file 'test_hapke.json'
with open('test_hapke.json') as json_file:
    data = json.load(json_file)
    geom = np.array([data['eme'], data['inc'], data['phi']]).transpose()
    photom = np.array([data['omega'], [x / 30 for x in data['theta0']], data['b'], data['c'], data['b0'], data['hh']]).transpose()



# Create Hapke2002 Model
myAdapter = ker.SixParamsHapkeAdapterConfig()

myModel = ker.HapkeModelConfig("2002", myAdapter, geom, 30).create()

myGenerator = ker.GaussianStatModelConfig("sobol", myModel, np.arange(6.0), 123456789).create()

x_gen, y_gen = myGenerator.gen_data(1)
print(y_gen)

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
