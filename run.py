import numpy as np
import pandas as pd
import kernel as ker
import json
import time

# Read geometries and photometries from file 'test_hapke.json'
with open('test_hapke.json') as json_file:
    data = json.load(json_file)
    geom = np.array([data['eme'], data['inc'], data['phi']]).transpose()
    photom = np.array([data['omega'], data['theta0'], data['b'], data['c'], data['b0'], data['hh']]).transpose()

# Create Hapke2002 Model
myModel = ker.PyFunctionnalModelFactory().getModel('hapke02', geom)

# Start time
start_time = time.time()

# Calculate reflectances
y = myModel.F(photom[0])


print(y)

# Calculate elapsed time for generating reflectances
#elapsed_time = (time.time() - start_time)
#print(elapsed_time)

# Write reflectances in file 'result.json'
#with open('result.json','w') as out_file:
 #   out_file.write(pd.Series({'y':y}).to_json())
