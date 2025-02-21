#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Comamnd line script for xllim.
#
# Usage: xllim_cli.py COMMAND H5_FILE <OTHER_FILE>
#
# Commands:
#   edit-config
#   import-direct-model
#   import-geometries
#   import-model
#   import-train
# 
#
# Usage examples:
#      xllim_cli.py print model.h5
#      xllim_cli.py edit  model.h5
#      xllim_cli.py train model.h5
#      xllim_cli.py predict -o output_file model.h5 observations.json
#      xllim_cli.py import [direct-model|geometries|train-data|model] source.[py|h5|json] target_model.h5
#      xllim_cli.py export [train-data|model|predictions] model.h5 output_file
# 2. Edit the configuration in a model.h5 file and import geometries if provided:
#      xllim_cli.py --edit  model.h5 <--geometries geometries.json>
# 3. Import functional model from an external model implementation, import configuration from a h5 file
#      xllim_cli.py --import-direct-model functional_model_file.[h5,py] target.h5
# 4. Import train data
#      xllim_cli.py --import-train-data train_data.[h5,py] target.h5
# 5. 
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
# import hashlib
import os
# import numpy as np
# import xllim


H5_STRING = h5py.string_dtype(encoding='utf-8')
H5_INT = 'i4'
H5_FLOAT = 'f8'
H5_GROUPS = {"functional": "/sythetic_data/functional_model_config",
            "generator":   "/sythetic_data/functional_model_config/data_generator",
            "gllim_model": "/xllim/gllim",
            "prediction_module": "/prediction_module_config",
            "importance_sampling": "/importance_sampling_config"}
H5_DATA_SETS = {"geometries": "/sythetic_data/functional_model_config/geometries",
                "train_data": "/sythetic_data/train_data"}


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
            if value:  # create the attribute only if value is specified
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


def configure_generator(h5_model_file):
    group = H5_GROUPS["generator"]
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


def configure_gllim_model(h5_file):
    group = H5_GROUPS["gllim_model"]
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


def configure_prediction_module(h5_file):
    group = H5_GROUPS["prediction_module"]
    config = (('prediction no.', 'Number of components to retain after merging', None, H5_INT),
              ('reduced GMM size', 'Number of components to retain. The prediction is the mean of the reduced mixtures', None, H5_INT),
              ('minimum component weight', 'Components which weight is lower than this threshold are discarded', None, H5_FLOAT))
    config_dialog(h5_file, group, config)


def configure_importance_sampling(h5_file):
    group = H5_GROUPS["importance_sampling"]
    config = (('N', 'Number of samples generated for the importance sampling of the target PDF', None, H5_INT),
              ('N_zero', 'Number of samples at initial stage (IMIS). If unspecified = N/10', None, H5_INT),
              ('B', 'Number of new samples at each step (IMIS). Default: N/20', None, H5_INT),
              ('J', 'Number of iterations (IMIS). Default: 18', None, H5_INT))
    config_dialog(h5_file, group, config)


def configure_functional(h5_model_file):
    group = H5_GROUPS["functional"]
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
    if model_type == "External":
        delete_attributes(h5_model_file, group, shkuratov_config)
        delete_attributes(h5_model_file, group, reduced_or_stick_config)
        delete_attributes(h5_model_file, group, common_hapke_config)
        print("Import external model:\n\txllim_cli.py import direct_model your_model.py")
        return
    
    # if model_type == "External": 
    #     attrs.create("path to model", external_model)
    #     hash = hashlib.md5(open(external_model,'rb').read()).hexdigest()
    #     attrs.create("model file MD5 hash", hash)
    #     return

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
    return


def print_group(h5_file, group_name):
    if H5_GROUPS[group_name] not in h5_file:
        print(f"No {group_name} configuration found")
        return
    print(f"{group_name} configuration:")
    attrs = h5_file[H5_GROUPS[group_name]].attrs
    for key, value in attrs.items():
        print(f"\t{key} : {value}")


def print_geometries(h5_model_file):
    ds = H5_DATA_SETS["geometries"] 
    if ds in h5_model_file:
        for key, value in h5_model_file[ds].items():
            print(f"    {key} : {value}") 


def print_h5(h5f):
    for short_name, group_name in H5_GROUPS.items():
        print(f"{short_name} configuration:")
        if group_name in h5f:
            for key, val in h5f[group_name].attrs.items():
                print(f"    {key}: {val}")
        print("")
    
    for short_name, group_name in H5_DATA_SETS.items():
        if group_name in h5f:
            print(f"{short_name} present")
            if short_name == "geometries":
                print_geometries(h5f)
    return True


def edit_config(h5f):
    for grp in H5_GROUPS.keys():
        print("")  # print empty line
        configure = globals()[f"configure_{grp}"]
        if H5_GROUPS[grp] in h5f:
            print_group(h5f, grp)
            if input(f"Edit {grp} configuration? (y/n) : ") == 'y':
                configure(h5f)
        else:
            if input(f"{grp} is not present. Do you want to add it? (y/n) : ") == 'y':
                configure(h5f)


def import_geometries(h5_model_file, geometries):
    if not os.path.isfile(geometries):
        raise ValueError("geometries must be a file")
    
    group = H5_DATA_SETS["geometries"]
    if group in h5_model_file:
        del h5_model_file[group]
    # check if geometries are in a json or hdf5 file
    filename, file_extension = os.path.splitext(geometries)

    # all good. We can create the group
    group = h5_model_file.create_group(group, track_order=True)
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


def import_direct_model(h5_file, python_script):
    print(f"importing {python_script}")


def import_data(what_to_import, source_file_path, h5f):
    if what_to_import == "geometries":
        import_geometries(h5f, source_file_path)
    elif what_to_import == "direct-model":
        import_direct_model(h5f, source_file_path)


def export_data(what_to_export, from_file, output_file):
    print(f"exporting {what_to_export}")


def train_model(file_path):
    print("training model")
        # check if train-condiguration is setup
        # check if train-data is available
        # do train
        # wrtie gllim model to h5f

def predict(file_path, observations_file_path, output):
    print("predicting")


def main():
    short_descr = "xllim_cli.py - Command-line interface for xllim."
    epilog = "Run 'xllim.py COMMAND --help' for more information on a command."
    parser = argparse.ArgumentParser(description=short_descr, epilog=epilog)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Print command
    print_parser = subparsers.add_parser("print", help="Print contents of the h5 file")
    print_parser.add_argument("model_file", help="Path to the model file (e.g., model.h5)")

    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Edit configuration")
    edit_parser.add_argument("model_file", help="Path to the model file (e.g., model.h5)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("model_file", help="Path to the model file (e.g., model.h5)")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions using the model")
    predict_parser.add_argument("model_file", help="Path to the model file (e.g., model.h5)")
    predict_parser.add_argument("observations_file", help="Path to the observations file (e.g., observations.json)")
    predict_parser.add_argument("-o", "--output", required=True, help="Path to the output file")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import data into the model")
    import_parser.add_argument("import_type", choices=["direct-model", "geometries", "train-data", "model"], help="Type of data to import")
    import_parser.add_argument("source_file", help="Path to the source file (e.g., source.py, source.h5, source.json)")
    import_parser.add_argument("model_file", help="Path to the target model file (e.g., target_model.h5)")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export data from the model")
    export_parser.add_argument("export_type", choices=["train-data", "model", "predictions"], help="Type of data to export")
    export_parser.add_argument("model_file", help="Path to the model file (e.g., model.h5)")
    export_parser.add_argument("output_file", help="Path to the output file")

    # Parse arguments
    args = parser.parse_args()
    
    h5_file_name = args.model_file
    if not os.path.isfile(h5_file_name):
        print(f"File {h5_file_name} not found")
        return 
    # read-only commands
    with h5py.File(h5_file_name, 'r') as h5f:
        if args.command == "print":
            print_h5(h5f)
            return
        elif args.command == "export":
            export_data(args.export_type, h5f, args.output_file)
            return
    
    # open the file for wrting
    with h5py.File(h5_file_name, 'a') as h5f:
        if args.command == "edit":
            edit_config(h5f)
        elif args.command == "train":
            train_model(h5f)
        elif args.command == "predict":
            predict(h5f, args.observations_file, args.output)
        elif args.command == "import":
            import_data(args.import_type, args.source_file, h5f)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
