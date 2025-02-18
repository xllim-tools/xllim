#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Comamnd line script for xllim
# 
# Usage examples:
# 1. Use model.h5 to inverse the observations in observations.h5:
#      xllim_cli.py model.h5 observations.h5
# 2. Use xllim_cli to modify the configuration in a model.h5 file:
#      xllim_cli.py model.h5
#
# model.h5 is a hdf5 file containing the following datasets:
# /synthetic_data/functional_model_config
#        properties : model_type, version, variant, theta_bar maximum
# /synthetic_data/functional_model_config/geometries
# /synthetic_data/generator
#   propoerties : model_type, version, variant, theta_bar maximum

# Outputs are in netcdf format

import argparse
import h5py
import hashlib
import os
# import numpy as np
# import xllim

def configure_data_generator(h5_model_file):
    if "/sythetic_data/functional_model_config/data_generator" in h5_model_file:
        del h5_model_file["/sythetic_data/functional_model_config/geometries"]


def import_geometries(h5_model_file, geometries):
    if "/sythetic_data/functional_model_config/geometries" in h5_model_file:
        del h5_model_file["/sythetic_data/functional_model_config/geometries"]
    # check if geometries are in a json or hdf5 file
    if not os.path.isfile(geometries):
        raise ValueError("geometries must be a file")
    filename, file_extension = os.path.splitext(geometries)

    # all good. We can create the group
    group = h5_model_file.create_group("/sythetic_data/functional_model_config/geometries", track_order=True)
    if file_extension == '.h5':
        # copy geometetries from geometries to our model_file
        with h5py.File(geometries, 'r') as h5_geometries:
            for key, value in h5_geometries.items():
                group.create_dataset(key, data=value)
    elif file_extension == '.json':
        import json
        with open(geometries) as json_file:
            data = json.load(json_file)
            for k, v in data.items():
                group.create_dataset(k, (len(v)), dtype='f4', data=v)
    
    # wrtie configuration to the file
    h5_model_file.flush()


def configure_functional_model_config(h5_model_file, geometries=None, external_model=None):
    if "/sythetic_data/functional_model_config" not in h5_model_file:
        group = h5_model_file.create_group("/sythetic_data/functional_model_config", track_order=True)
    else:
        group = h5_model_file["/sythetic_data/functional_model_config"]
    attrs = group.attrs

    supported_models = ["Hapke 2002", "Shkuratov", "Test model", "External"]
    # Hapke model configuration
    common_hapke_config = { "variant": ("full", "reduced", "hockey_stick"),
                     "theta_bar maximum": "a positive number (0, 30)",
    }
    reduced_or_stick_config = { "B0" : "Magnitude of the opposition effect (0, XXX)",
                             "H" : "Angular width of the opposition effect (0, XXX)"}
    # Shkuratov model configuration
    shkuratov_config = { "Max An" : "Value used to normalise the An parameter into the mathematical parameter space [0,1]",
                        "Min An" : "Minimum value used to normalise the An parameter",
                        "Max mu1" : "Value used to normalise the mu1 parameter into the mathematical parameter space [0,1]",
                        "Min mu1" : "Minimum value used to normalise the mu1 parameter",
                        "Max nu" : "Value used to normalise the nu parameter into the mathematical parameter space [0,1]",
                        "Min nu" : "Minimum value used to normalise the nu parameter",
                        "Max m0" : "Value used to normalise the m0 parameter into the mathematical parameter space [0,1]",
                        "Min m0" : "Minimum value used to normalise the m0 parameter",
                        "Max mu2" : "Value used to normalise the mu2 parameter into the mathematical parameter space [0,1]",
                        "Min mu2" : "Minimum value used to normalise the mu2 parameter",
    }
    # set model type
    print("Choose model:")
    for model in supported_models:
        print(f"{supported_models.index(model)+1}. {model}")
    model_type = supported_models[int(input())-1]

    # check if required data is provided for the selected model
    if model_type == "External" and external_model is None:
        raise ValueError("External model is required")
    elif model_type == "Hapke 2002" or model_type == "Shkuratov":
        if geomtries is None:
            raise ValueError("Geometries are required")
    
    # remove existing configuration
    for key in attrs.keys():
        attrs.__delitem__(key)
        
    attrs.create("model type", model_type)

    if model_type == "External": 
        attrs.create("path to model", external_model)
        hash = hashlib.md5(open(external_model,'rb').read()).hexdigest()
        attrs.create("model file MD5 hash", hash)
        return

    if model_type == "Hapke 2002":
        for key, value in common_hapke_config.items():
            if value.__class__ == tuple:
                print(f"Choose {key} :")
                for i, variant in enumerate(value):
                    print(f"{i+1}. {variant}")
                attrs.create(key, value[int(input())-1])
                print(attrs.get(key))
            else:
                val = float(input(f"{key} ({value}) : "))
                attrs.create(key, val, dtype='f4')
        if attrs["variant"] != "full":
            for key, value in reduced_or_stick_config.items():
                val = float(input(f"{key} ({value}) : "))
                attrs.create(key, val, dtype='f4')
    elif model_type == "Shkuratov":
        for key, value in shkuratov_config.items():
            val = float(input(f"{key} ({value}) : "))
            attrs.create(key, val, dtype='f4')

    # write geometries
    import_geometries(h5_model_file, geometries)

    # wrtie configuration to the file
    h5_model_file.flush()


def print_functional_model_config(h5_model_file):
    if "/sythetic_data/functional_model_config" not in h5_model_file:
        print("No functional model configuration found")
    attrs = h5_model_file["/sythetic_data/functional_model_config"].attrs
    for key, value in attrs.items():
        print(f"{key} : {value}")
    
    # print geometries
    if "/sythetic_data/functional_model_config/geometries" in h5_model_file:
        for key, value in h5_model_file["/sythetic_data/functional_model_config/geometries"].items():
            print(f"{key} : {value}") 


def modify_model_file(args):
    h5f = h5py.File(args.model,'a')
    if "/sythetic_data" in h5f:
        # print current configuration and ask if needs to be modified
        print("Current configuration:")
        print("Functional model: XXXX")
        if input("Edit functional model configuration? (y/n) : ") == 'y':
            # modifiy the configuration
            pass
    else:
        if input("Do you wish to generate a systhetic data set? (y/n) : ") == 'y':
            h5f.create_group("/sythetic_data")
            modify_model_file(args)

def inverse_observations(args):
    model = h5py.File(args.model,'r')
    return

def main():
    parser = argparse.ArgumentParser(description='Command line interface for xllim')
    parser.add_argument('model', type=str, help='Model file in hdf5 format')
    parser.add_argument('observations', type=str, nargs='?', default=None, help='Observations file')
    # parser.add_argument('--help', action='help', help='Show this help message and exit')
    args = parser.parse_args()

    if args.observations is None:
        modify_model_file(args)
    else:
        inverse_observations(args)


    # config = xllim.Configuration()
    # config.direct_model_type = direct_model_type
    # config.write(args.write_config)
    return

if __name__ == '__main__':
    main()
