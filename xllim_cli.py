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


H5_STRING = h5py.string_dtype(encoding='utf-8')
H5_INT = 'i4'
H5_FLOAT = 'f8'


def config_dialog(h5_file: str, group_or_dataset, options):
    """Go through options, ask user input and set properties on group_ordataset in the h5_file
    
    Parameters
    ----------
    options : a tupple of tupples with format:
        (attribute name, (help string) , None or tupple of possible values)
    """
    if group_or_dataset not in h5_file:
        print(f"Creating {group_or_dataset}")
        g = h5_file.create_group(group_or_dataset, track_order=True)
    else:
        g = h5_file[group_or_dataset]
    attrs = g.attrs
    for opt in options:
        name, help, vals, type = opt
        # read current value of option if exists
        value = attrs.get(name)
        cval_string = ""
        if value is not None:
            cval_string = f"[{value}] "

        # setup help string
        if len(help):
            help = f"({help}) "
            
        if vals is None:
            prompt = f"{name} {help} {cval_string}: " 
            i = input(prompt)
            if len(i) > 0:
                value = float(i)
                print(f"converted: {value}")
        else:
            print(f"Choose {name} {help}:")
            for i, v in enumerate(vals):
                print(f"{i+1}. {v}")
            prompt = f"{cval_string} : " 
            i = input(prompt)
            if len(i) > 0:
                value = vals[int(i)-1]
        if len(cval_string):  # attribute already exists
            attrs.modify(name, value)
        else:
            attrs.create(name, value, dtype=type)
        print(f"\033[F\033[{len(prompt)}G {attrs.get(name)}")
    
    h5_file.flush()
    return attrs


def delete_attributes(h5_file, group, options):
    """ Delete attributes from a hdf5 group.

    We assume group exists in h5_file.
    Options follow the same format as configuration options.
    """
    attrs = h5_file[group].attrs
    for opt in options:
        name, _, _, _ = opt
        if name in attrs:
            del attrs[name]
    return


def configure_data_generator(h5_model_file):
    group = "/sythetic_data/functional_model_config/data_generator"
    generator_model_options = (("model","Type of Gaussian statistical model", ("basic", "dependent"), H5_STRING),
                               ("dataset size", "a positive number", None, H5_INT),
                               ("type", "Generator type",("sobol", "random", "latin hypercube"), H5_STRING),
                               ("seed", "Seed used by the random generator", None, H5_INT))
    basic_generator_options = (("variances", "Isometric fixed variance of the Gaussian noise (in %)", None, H5_FLOAT), )
    dependent_generator_options = (("noise effect", "signal to noise ratio", None, H5_FLOAT), )
    attrs = config_dialog(h5_model_file, group, generator_model_options)
    if attrs["model"] == "basic":
        config_dialog(h5_model_file, group, basic_generator_options)
        delete_attributes(h5_model_file, group, dependent_generator_options)
    else:
        config_dialog(h5_model_file, group, dependent_generator_options)
        delete_attributes(h5_model_file, group, basic_generator_options)


def configure_gllim(h5_file):
    group = "/xllim/gllim"
    gllim_options = (('K', 'Number of affine transformations', None, H5_INT),
                     ('Gamma type', 'Type of covariance matrix for the K GLLiM components', ("full", "diagonal", "isomorphic"), H5_STRING),
                     ('Sigma type', 'Type of covariance matrix for the Gaussian noise applied to each affine transformation', ("full", "diagonal", "isomorphic"), H5_STRING),
                     ('floor', 'Minimum threshold for the covariance values', None, H5_FLOAT),
                     ('init variant', 'Initialisation strategy applied to the GLLiM learning', ('fixed', 'multiple'), H5_STRING),
                     ('learning variant', 'Learning step strategy', ('GLLiM-EM', 'GMM-EM'), H5_STRING))
    multiple_init_options = (('initialisations no.', 'Number of initialization experiments', None, H5_INT),
                            ('init seed', 'The seed used by random generators', None, H5_INT),
                            ('init k-means iterations', '', None, H5_INT),
                            ('init GMM-EM iterations', '', None, H5_INT),
                            ('init GLLiM-EM iterations', '', None, H5_INT))
    fixed_init_options = (('init seed', 'The seed used by random generators', None, H5_INT),
                          ('init k-means iterations', '', None, H5_INT),
                          ('init GMM-EM iterations', '', None, H5_INT))
    gllim_em_options = (('GLLiM-EM iterations', '', None, H5_INT),
                        ('likelihood increase', '', None, H5_FLOAT))
    gmm_em_options = (('GMM-EM iterations', '', None, H5_INT),
                        ('k-means iterations', '', None, H5_FLOAT))
    attrs = config_dialog(h5_file, group, gllim_options)
    if attrs["init variant"] == 'fixed':
        config_dialog(h5_file, group, fixed_init_options)
        delete_attributes(h5_file, group, multiple_init_options)
    else:
        config_dialog(h5_file, group, multiple_init_options)
        delete_attributes(h5_file, group, fixed_init_options)
    
    if attrs['learning variant'] == 'GLLiM-EM':
        config_dialog(h5_file, group, gllim_em_options)
        delete_attributes(h5_file, group, gmm_em_options)
    else:
        config_dialog(h5_file, group, gmm_em_options)
        delete_attributes(h5_file, group, gllim_em_options)


def configure_predictions(h5_file):
    group = '/xllim/prediction_module_config'
    config = (('prediction no.', 'Number of components to retain after merging', None, H5_INT),
              ('reduced GMM size', 'Number of components to retain. The prediction is the mean of the reduced mixtures', None, H5_INT),
              'minimum component weight', 'Components which weight is lower than this threshold are discarded', None, H5_FLOAT)
    config_dialog(h5_file, group, config)


def configure_importance_sampling(h5_file):
    group = '/xllim/importance_sampling_config'
    config = (('N', 'Number of samples generated for the importance sampling of the target PDF', None, H5_INT),
              ('N_zero', 'Number of samples at initial stage (IMIS). If unspecified = N/10', None, H5_INT),
              ('B', 'Number of new samples at each step (IMIS). Default: N/20', None, H5_INT),
              ('J', 'Number of iterations (IMIS). Default: 18', None, H5_INT))
    config_dialog(h5_file, group, config)


def configure_functional_model(h5_model_file, geometries=None, external_model=None):
    group = "/sythetic_data/functional_model_config"
    supported_models = (("model", "(functional model) ", ("Hapke 2002", "Shkuratov", "Test model", "External"), H5_STRING), )
    # Hapke model configuration
    common_hapke_config = ( ("variant", "", ("full", "reduced", "hockey_stick"), H5_STRING),
                            ("theta_bar maximum", "(a positive number [0, 30]) ", None, H5_FLOAT)
    )
    reduced_or_stick_config = (("B0", "Magnitude of the opposition effect (0, XXX)", None, H5_FLOAT),
                               ("H", "Angular width of the opposition effect (0, XXX)", None, H5_FLOAT))
    # Shkuratov model configuration
    shkuratov_config = (("Max An" , "Value used to normalise the An parameter into the mathematical parameter space [0,1]", None, H5_FLOAT),
                        ("Min An" , "Minimum value used to normalise the An parameter", None, H5_FLOAT),
                        ("Max mu1",  "Value used to normalise the mu1 parameter into the mathematical parameter space [0,1]", None, H5_FLOAT),
                        ("Min mu1",  "Minimum value used to normalise the mu1 parameter", None, H5_FLOAT),
                        ("Max nu" , "Value used to normalise the nu parameter into the mathematical parameter space [0,1]", None, H5_FLOAT),
                        ("Min nu" , "Minimum value used to normalise the nu parameter", None, H5_FLOAT),
                        ("Max m0" , "Value used to normalise the m0 parameter into the mathematical parameter space [0,1]", None, H5_FLOAT),
                        ("Min m0" , "Minimum value used to normalise the m0 parameter", None, H5_FLOAT),
                        ("Max mu2",  "Value used to normalise the mu2 parameter into the mathematical parameter space [0,1]", None, H5_FLOAT),
                        ("Min mu2",  "Minimum value used to normalise the mu2 parameter", None, H5_FLOAT),
    )
    # set model type
    attrs = config_dialog(h5_model_file, group, supported_models)

    # check if required data is provided for the selected model
    model_type = attrs["model"]
    if model_type == "External" and external_model is None:
        raise ValueError("External model is required")
    elif model_type == "Hapke 2002" or model_type == "Shkuratov":
        if geometries is None:
            raise ValueError("Geometries are required")
    
    if model_type == "External": 
        attrs.create("path to model", external_model)
        hash = hashlib.md5(open(external_model,'rb').read()).hexdigest()
        attrs.create("model file MD5 hash", hash)
        return

    if model_type == "Hapke 2002":
        config_dialog(h5_model_file, group, common_hapke_config)
        if attrs["variant"] == "full":
            delete_attributes(h5_model_file, group, reduced_or_stick_config)
        else:
            config_dialog(h5_model_file, group, reduced_or_stick_config)
        # remove Shkuratov related attributes
        delete_attributes(h5_model_file, group, shkuratov_config)
    elif model_type == "Shkuratov":
        config_dialog(h5_model_file, group, shkuratov_config)
        # remove Hapke related attributes
        delete_attributes(h5_model_file, group, common_hapke_config)
        delete_attributes(h5_model_file, group, reduced_or_stick_config)

    # write geometries
    import_geometries(h5_model_file, geometries)

    # wrtie configuration to the file
    h5_model_file.flush()
    


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
                group.create_dataset(k, (len(v)), dtype=H5_FLOAT, data=v)
    
    # wrtie configuration to the file
    h5_model_file.flush()


def print_functional_model_config(h5_model_file):
    if "/sythetic_data/functional_model_config" not in h5_model_file:
        print("No functional model configuration found")
        return
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
