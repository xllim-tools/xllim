#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Comamnd line script for xllim.
#
# Usage: xllim_cli.py COMMAND H5_FILE <OTHER_FILE>
#
# Usage examples:
#      xllim_cli.py print model.h5
#      xllim_cli.py edit  model.h5
#      xllim_cli.py train model.h5
#      xllim_cli.py predict -o output_file model.h5 observations.json
#      xllim_cli.py import [functional|geometries|train-data|model] source.[py|h5|json] target_model.h5
#      xllim_cli.py export [train-data|model|predictions] model.h5 output_file
# 2. Edit the configuration in a model.h5 file and import geometries if provided:
#      xllim_cli.py edit  model.h5 <--geometries geometries.json>
# 3. Import functional model from an external model implementation, import configuration from a h5 file
#      xllim_cli.py import functional file_with_functional_model.h5 target.h5
# 4. Import train data
#      xllim_cli.py import train-data train_data.[h5|json] target.h5
#

# Outputs are in netcdf format

import argparse
import ast
import json
import pickle
import h5py
import logging
import os
import numpy as np
from datetime import datetime
try:
    import xllim
except ImportError as e:
    from unittest.mock import Mock
    xllim = Mock()
    xllim.TestModel().genData.return_value = np.ones(100), np.ones(10)
    print(f"\nWARNING: Import error: {e}.\nSome commands my not work.\n")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xllim_cli")


H5_STRING = h5py.string_dtype(encoding='utf-8')
H5_INT = 'i4'
H5_FLOAT = 'f8'
H5_GROUPS = {"functional": "/functional_model/config",
             "generator":   "/functional_model/data_generator",
             "gllim_model": "/gllim",
             "prediction_module": "/prediction_module_config",
             "importance_sampling": "/importance_sampling_config"}
H5_DATA_SETS = {"geometries": "/functional_model/geometries",
                "train_data": "/train_data",
                "gllim_model": "/gllim/serialised_gllim"}
SUPPORTED_MODELS = (("model", "functional model", ("Hapke",
                    "Shkuratov", "External", "Test model"), H5_STRING), )
HAPKE_OPTIONS = (("variant", "", ("1993", "2002"), H5_STRING),
                 ("adapter", "Number of Hapke's model parameters",
                  ("three", "four", "six"), H5_STRING),
                 ("theta_bar maximum", "Value used to transform theta_bar between physical and mathematical spaces (eg. 30)", None, H5_INT),
                 ("b0", "The amplitude of the opposition effect", None, H5_FLOAT),
                 ("h", "Angular width of the opposition effect (0, XXX)", None, H5_FLOAT))
SHKURATOV_OPTIONS = (("variant", "Number of model parameters", ("3p", "5p"), H5_STRING),
                     ("scaling_coeffs",
                      "A set of L coefficients used in the transformation between physical and mathematical spaces ([0.1, 0.2])", None, H5_STRING),
                     ("offset", "A set of L offsets used in the transformation between physical and mathematical spaces", None, H5_STRING))
EXTERNAL_MODEL_OPTIONS = (("class name", "", None, H5_STRING),
                          ("file name", "", None, H5_STRING),
                          ("file path", "", None, H5_STRING))
GENERATOR_OPTIONS = (("N", "Dataset size; a positive number", None, H5_INT),
                     ("type", "Generator type",
                      ("sobol", "random", "latin hypercube"), H5_STRING),
                     ("covariance", "Covariances value. Same for all D.", None, H5_FLOAT),
                     ("seed", "Seed used by the random generator", None, H5_INT))  # covariance could be a list of different values
GLLIM_OPTIONS = (('K', 'Number of affine transformations', None, H5_INT),
                 ('Gamma type', 'Type of covariance matrix for the K GLLiM components',
                  ("full", "diag", "iso"), H5_STRING),
                 ('Sigma type', 'Type of covariance matrix for the Gaussian noise applied to each affine transformation',
                  ("full", "diag", "iso"), H5_STRING),
                 ('n_hidden', 'Number of hidden variables', None, H5_INT))
GLLIM_INIT_OPTIONS = (('gllim_em_iteration', 'Number of EM iterations for GLLiM', None, H5_INT),
                      ('gllim_em_floor',
                       'Floor value for EM iterations in GLLiM', None, H5_FLOAT),
                      ('gmm_kmeans_iteration',
                       'Number of k-means iterations for GMM', None, H5_INT),
                      ('gmm_em_iteration',
                       'Number of EM iterations for GMM', None, H5_INT),
                      ('gmm_floor', 'Floor value for EM iterations in GMM', None, H5_FLOAT),
                      ('nb_experiences', 'Number of experiences', None, H5_INT),
                      ('seed', 'Random numer seed', None, H5_INT))
GLLIM_TRAIN_VARIANTS = (('train_variant', 'Which training method to apply',
                        ('GLLiM', 'JGMM'), H5_STRING), )
GLLIM_TRAIN_OPTIONS = (('train_max_iteration', 'Maximum number of iterations', None, H5_INT),
                       ('train_ratio_ll',
                        'Ratio for log-likelihood convergence', None, H5_FLOAT),
                       ('train_floor', 'Floor value for the training process', None, H5_FLOAT))
JGMM_TRAIN_OPTIONS = (('jgmm_train_kmeans_iteration', 'The number of iterations of the k-means algorithm', None, H5_INT),
                      ('jgmm_train_em_iteration',
                       'The number of iterations of the EM algorithm', None, H5_INT),
                      ('jgmm_train_floor', 'The variance floor (smallest allowed value) for the diagonal covariances', None, H5_FLOAT))
PREDICTION_OPTIONS = (('K_merged', 'Merged the full GMM (K components) into K_merged gaussian components', None, H5_INT),
              ('merging_threshold', 'Threshold on the merged GMM weights. Gaussian component with a weight below this threshold are ignored.', None, H5_FLOAT))
IS_OPTIONS = (('N_zero', 'Number of samples at initial stage (IMIS). If unspecified = N/10', None, H5_INT),
              ('B', 'Number of new samples at each step (IMIS). Default: N/20', None, H5_INT),
              ('J', 'Number of iterations (IMIS). Default: 18', None, H5_INT))


def config_dialog(h5_file: str, group_or_dataset, options) -> list:
    """Go through options, ask user input and set properties on group_ordataset in the h5_file

    Parameters
    ----------
    options : a tupple of tupples with format:
        (attribute name, (help string) , None or tupple of possible values, type)
    """
    modified = []
    if group_or_dataset not in h5_file:
        print(f"Creating {group_or_dataset}")
        g = h5_file.create_group(group_or_dataset, track_order=True)
    else:
        g = h5_file[group_or_dataset]
    attrs = g.attrs
    for opt in options:
        name, help, vals, dtype = opt
        # read current value of option if exists
        value = attrs.get(name)
        cval_string = ""
        if value is not None:
            cval_string = f"[{value}] "

        # setup help string
        if len(help):
            help = f"({help}) "

        if type(vals) is tuple:
            print(f"Choose {name} {help}:")
            for i, v in enumerate(vals):
                print(f"{i+1}. {v}")
            prompt = f"{cval_string} : "
            i = input(prompt)
            if len(i) > 0:
                value = vals[int(i)-1]
        else:
            prompt = f"{name} {help} {cval_string}: "
            i = input(prompt)
            if len(i) > 0:
                if dtype == H5_FLOAT:
                    value = float(i)
                else:
                    value = i

        # modify or create an attribute
        if len(cval_string):  # attribute already exists
            attrs.modify(name, value)
        else:
            if value:  # create the attribute only if value is specified
                attrs.create(name, value, dtype=dtype)
        print(f"\033[F\033[{len(prompt)}G {attrs.get(name)}")
        modified.append(name)

    h5_file.flush()
    return modified


def delete_all_attributes_except(h5_file, group, attrs_to_keep):
    attrs = h5_file[group].attrs
    for at in attrs:
        if at not in attrs_to_keep:
            del attrs[at]
    return


def configure_generator(h5_model_file):
    group = H5_GROUPS["generator"]
    changed = config_dialog(h5_model_file, group, GENERATOR_OPTIONS)
    delete_all_attributes_except(h5_model_file, group, changed)
    return


def configure_gllim_model(h5_file):
    group = H5_GROUPS["gllim_model"]

    changed = config_dialog(h5_file, group, GLLIM_OPTIONS)
    changed += config_dialog(h5_file, group, GLLIM_INIT_OPTIONS)

    changed += config_dialog(h5_file, group, GLLIM_TRAIN_VARIANTS)
    train_variant = h5_file[group].attrs['train_variant']

    if train_variant == 'GLLiM' or train_variant == 'GLLiM and JGMM':
        changed += config_dialog(h5_file, group, GLLIM_TRAIN_OPTIONS)
    if train_variant == 'JGMM' or train_variant == 'GLLiM and JGMM':
        changed += config_dialog(h5_file, group, JGMM_TRAIN_OPTIONS)

    delete_all_attributes_except(h5_file, group, changed)
    return


def configure_prediction_module(h5_file):
    group = H5_GROUPS["prediction_module"]
    changed = config_dialog(h5_file, group, PREDICTION_OPTIONS)
    delete_all_attributes_except(h5_file, group, changed)
    return


def configure_importance_sampling(h5_file):
    group = H5_GROUPS["importance_sampling"]
    changed = config_dialog(h5_file, group, IS_OPTIONS)
    delete_all_attributes_except(h5_file, group, changed)
    return


def configure_functional(h5_model_file):
    group = H5_GROUPS["functional"]

    # set model type
    changed_attrs = config_dialog(h5_model_file, group, SUPPORTED_MODELS)

    # check if required data is provided for the selected model
    model_type = h5_model_file[group].attrs["model"]
    if model_type == "External":
        changed_attrs += config_dialog(h5_model_file,
                                       group, EXTERNAL_MODEL_OPTIONS)
    elif model_type == "Hapke":
        changed_attrs += config_dialog(h5_model_file, group, HAPKE_OPTIONS)
    elif model_type == "Shkuratov":
        changed_attrs += config_dialog(h5_model_file, group, SHKURATOV_OPTIONS)
    elif model_type == "Test model":
        pass
    else:
        logger.error("Invalid functional model type")

    delete_all_attributes_except(h5_model_file, group, changed_attrs)
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
            print(f"    {key} : {value[()]}")


def print_h5(h5f, verbose):
    if H5_GROUPS["functional"] in h5f and H5_GROUPS["generator"] in h5f:
        attrs = h5f[H5_GROUPS["functional"]].attrs
        model_type = attrs["model"]
        if model_type == "Hapke" or model_type == "Shkuratov":
            try:
                g = _geometries(h5f)
            except ValueError:
                print(f"Geometries missing for mode: {model_type}")
            print(f"Direct model:\tYES ({model_type}, geometries{g.shape})")
        else:
            print(f"Direct model:\tYES ({model_type})")
    else:
        print(f"Direct model:\tNO")

    if H5_DATA_SETS["train_data"] in h5f:
        x, y = _train_data(h5f)
        print(f"Train data:\tYES (X{x.shape}  Y{y.shape})")
    else:
        print("Train data:\tNO")

    if H5_DATA_SETS["gllim_model"] in h5f:
        print("Trained model:\tYES")
    else:
        print("Trained model:\tNO")

    if verbose:
        print("\nConfiguration:\n")
        for short_name, group_name in H5_GROUPS.items():
            print(f"{short_name} configuration:")
            if group_name in h5f:
                for key, val in h5f[group_name].attrs.items():
                    print(f"    {key}: {val}")
            print("")

        if H5_DATA_SETS["geometries"] in h5f:
            print("geometries:")
            print_geometries(h5f)
    return


def edit_config(h5f):
    for grp in H5_GROUPS.keys():
        print("")

        # get the configure function for the group
        configure = globals()[f"configure_{grp}"]

        prompt = f"{grp} is not present. Do you want to add it? (y/n/q) : "
        if H5_GROUPS[grp] in h5f:
            print_group(h5f, grp)
            prompt = f"Edit {grp} configuration? (y/n/q) : "

        k = input(prompt)
        if k == 'y':
            configure(h5f)
        elif k == 'q':
            exit(0)
        elif k == 'n':
            pass
        else:
            logger.warning(f'User entered {k}. Assuming it is no')


def import_geometries(source_path: str, dest_h5f: h5py.File) -> None:
    """
    Import geometries from a JSON file into an HDF5 file.

    Args:
        source_path (str): Path to the source JSON file.
        dest_h5f (h5py.File): Destination HDF5 file object.

    Raises:
        ValueError: If the source file is not a JSON file.
        IOError: If there are issues reading or writing files.
    """
    group = H5_DATA_SETS["geometries"]
    _, file_extension = os.path.splitext(source_path)

    try:
        if file_extension == '.json':
            if group in dest_h5f:
                logger.info(f"Removing exisitng {group}")
                del dest_h5f[group]

            logger.info(f'Importing geometries from {source_path}')
            group = dest_h5f.create_group(group, track_order=True)

            with open(source_path, 'r') as json_file:
                data = json.load(json_file)
                for k, v in data.items():
                    group.create_dataset(k, (len(v)), dtype=H5_FLOAT, data=v)

            # set source attribute to filename
            attrs = group.attrs
            attrs.create("source", source_path, dtype=H5_STRING)

        else:
            raise ValueError("geometries must be a json file")

        # write configuration to the file
        dest_h5f.flush()
        logger.info('Geometries import done')

    except (IOError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"An error occurred: {e}")
        raise


def _h5_copy(group_name: str, src_h5: str, dst_h5: h5py.File):
    if not h5py.is_hdf5(src_h5):
        raise ValueError(f'{src_h5} is not a valid hdf5 file.')
    with h5py.File(src_h5, 'r') as src_h5f:
        if group_name not in src_h5f:
            raise ValueError(f"{group_name} not found in the source file")
        if group_name in dst_h5:  # remove it and replace it
            print(f"Removing exisitng {group_name}")
            del dst_h5[group_name]
        print(f"Copying {group_name} from {src_h5}")
        dst_h5.copy(src_h5f[group_name], group_name)


def import_data(what_to_import, source_file_path, h5f):
    if what_to_import == "geometries":
        import_geometries(source_file_path, h5f)
    elif what_to_import == "train-data":
        # TODO implement import from other sources (JSON?)
        _h5_copy(H5_DATA_SETS["train_data"], source_file_path, h5f)
    else:
        logger.error(f"Invalid option: {what_to_import}")


def copy_h5_section_dialog(src_h5: str, dst_h5):
    """List all group and datas set names present in src and ask user what to copy to dst."""
    # construct dict with data present in source file
    data = {}
    i = 1
    print(src_h5)
    with h5py.File(src_h5, 'r') as src_h5f:
        for k, v in H5_GROUPS.items():
            if v in src_h5f:
                data[i] = k
                i += 1
        for k, v in H5_DATA_SETS.items():
            if v in src_h5f:
                data[i] = k
                i += 1
    print(f"Data present in {src_h5}:")
    for k, v in data.items():
        print(f"{k}. {v}")

    input_str = input("What to copy (coma seprated list ex. 1,3): ")
    for i in input_str.split(","):
        obj_id = data[int(i)]
        if obj_id in H5_GROUPS.keys():
            _h5_copy(H5_GROUPS[obj_id], src_h5, dst_h5)
        else:
            _h5_copy(H5_DATA_SETS[obj_id], src_h5, dst_h5)
    return


def _shkuratov_config(attrs):
    """Returns: variant, scalingCoeffs, offset"""

    variant, s, o = _option_values(attrs, SHKURATOV_OPTIONS)
    # parse scalingCoeffs and offsets
    scaling = ast.literal_eval(s)
    offsets = ast.literal_eval(o)
    if len(scaling) != len(offsets):
        logger.error(
            "Scaling and offset vectors must hae the same length in Shkuratov config. Aborting.")
        exit(1)
    return variant, scaling, offsets


def _geometries(h5f):
    """Reads geometries from the h5f file."""

    if H5_DATA_SETS["geometries"] not in h5f:
        raise ValueError("No Geometries")
     
    group = h5f[H5_DATA_SETS["geometries"]]
    d = group["sza"].shape[0]  # len of the first geometries vector
    data = np.zeros((d, 3), dtype=np.float64)
    for i, ds in enumerate(["sza", "vza", "phi"]):
        data[:, i] = group.get(ds)
    return data


def _physical_model(h5f):
    attrs = h5f[H5_GROUPS["functional"]].attrs
    model_type = attrs["model"]
    if model_type == "Test model":
        f_model = xllim.TestModel()
    elif model_type == "Hapke":
        f_model = xllim.HapkeModel(_geometries(h5f), *_option_values(attrs, HAPKE_OPTIONS))
    elif model_type == "Shkuratov":
        variant, scalingCoeffs, offset = _shkuratov_config(attrs)
        f_model = xllim.ShkuratovModel(_geometries(h5f), variant, scalingCoeffs, offset)
    elif model_type == "External":
        f_model = xllim.ExternalPythonModel(*_option_values(attrs, EXTERNAL_MODEL_OPTIONS))
    return f_model


def generate_data(h5f):
    """Generates sythetic train data, if functional model and generator configurations are present."""

    if H5_GROUPS["functional"] in h5f and H5_GROUPS["generator"] in h5f:
        logger.info("Generating train dataset")
        f_model = _physical_model(h5f)

        attrs = h5f[H5_GROUPS["generator"]].attrs
        N, generator_type, covariance, seed = _option_values(attrs, GENERATOR_OPTIONS)
        D = f_model.getDimensionY()
        covariance = np.ones(D) * covariance

        x_gen, y_gen = f_model.genData(N, generator_type, covariance, seed)

        # store x_gen and y_gen in a dataset
        if H5_DATA_SETS["train_data"] in h5f:
            del h5f[H5_DATA_SETS["train_data"]]
        group = h5f.create_group(H5_DATA_SETS["train_data"], track_order=True)
        group.create_dataset("X", x_gen.shape, dtype=H5_FLOAT, data=x_gen)
        group.create_dataset("Y", y_gen.shape, dtype=H5_FLOAT, data=y_gen)
        # write source of the train dataset
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        attrs = group.attrs
        info_text = f"generated using {model_type} on {date_time}"
        attrs.create("source", info_text, dtype=H5_STRING)

        h5f.flush()
    else:
        logger.error("Configuration is not complete. Cannot generate data.")
        exit(1)
    return


def export_data(what_to_export, from_file, output_file):
    print(f"TODO exporting {what_to_export}")


def _option_values(attrs, options):
    """Returns option values stored in attrs.

    Args:
        attrs : hdf5 attributes object
        options (tuple of tuples): A tuple of tuples where the first element is the key to look up in attrs.

    Returns:
        list: A list of values corresponding to the keys in options.
    """
    return [attrs[opt[0]] for opt in options]


def _train_data(h5f):
    # check if train-data is available
    train_group_name = H5_DATA_SETS["train_data"]
    if train_group_name not in h5f:
        logger.info("Train data not available. Generating")
        generate_data(h5f)

    train_group = h5f[train_group_name]
    X = np.array(train_group.get("X"), dtype=np.float64)
    Y = np.array(train_group.get("Y"), dtype=np.float64)
    return X, Y


def _gllim(h5f):
    X, Y = _train_data(h5f) 
    L = X.shape[0]
    D = Y.shape[0]
    
    gllim_attrs = h5f[H5_GROUPS["gllim_model"]].attrs
    K, gamma, sigma, n_hidden = _option_values(gllim_attrs, GLLIM_OPTIONS)
    return xllim.GLLiM(K, D, L, gamma, sigma, n_hidden)


def _prediction(h5f):
    pred_attrs = h5f[H5_GROUPS["prediction_module"]].attrs
    K_merged, merging_threshold = _option_values(pred_attrs, PREDICTION_OPTIONS)
    return K_merged, merging_threshold


def train_model(h5f):
    verbose = 1

    X, Y = _train_data(h5f)

    gllim = _gllim(h5f)

    gllim_attrs = h5f[H5_GROUPS["gllim_model"]].attrs
    gllim.initialize(X, Y, *_option_values(gllim_attrs, GLLIM_INIT_OPTIONS), verbose)

    # train
    train_variant = _option_values(gllim_attrs, GLLIM_TRAIN_VARIANTS)[0]  # cause it is a list
    if train_variant == 'GLLiM':
        gllim.train(X, Y, *_option_values(gllim_attrs, GLLIM_TRAIN_OPTIONS), verbose)
    else:
        gllim.trainJGMM(X, Y, *_option_values(gllim_attrs, JGMM_TRAIN_OPTIONS), verbose)

    # wrtie gllim model to h5f
    gllim_serialised = pickle.dumps(gllim.getParams())
    group_name = H5_GROUPS["gllim_model"]
    h5f[group_name].create_dataset("serialised_gllim", data=np.void(gllim_serialised))
    h5f.flush()
    
    return


def _load(observations_file_path):
    relative_uncertainty = 0.1  # TODO which config should have it?
    if observations_file_path.endswith('.json'):
        with open(observations_file_path) as f:
            observations_data = json.load(f)
        observations_title = list(observations_data.keys())[0]
        data = observations_data[observations_title]

        reflectances = []
        for key in data.keys():
            if key.startswith('reflectance'):
                observations = []
                for wavelength in data[key]:
                    if len(wavelength) == 2:
                        observations.append({
                            'reflectances': wavelength[0],
                            'incertitudes': wavelength[1]
                        })
                    else:
                        observations.append({
                            'reflectances': wavelength[0],
                            'incertitudes': [reflectance * relative_uncertainty for reflectance in wavelength[0]]
                        })
                reflectances.append(observations)
    elif observations_file_path.endswith('.nc'):  # TODO NetCDF or a .zip file with 2 files?
        pass
    else:
        raise ValueError("Observations file must be a json file")

    #return np.array(reflectances)
    return reflectances


def importance_sampling(h5f, all_reflectances, all_incertitudes, predictions, nb_pred):
    nb_centers = predictions.mergedGMM.weights.shape[1]
    is_attrs = h5f[H5_GROUPS["importance_sampling"]].attrs
    is_results = []
    logging.info("[Sampling] Starting Incremental Mixture Importance Sampling for the {} predictions by the mean and by the {} centers".format(nb_pred, nb_centers))
    logging.info("[Sampling] Processsing ...")
    
    logging.info("[Sampling] Processsing merged_mean ...")
    ph_model = _physical_model(h5f)
    is_results.append(
        ph_model.importanceSampling(
            predictions.mergedGMM,
            all_reflectances,
            all_incertitudes,
            *_option_values(is_attrs, IS_OPTIONS),
            covariance=np.zeros(ph_model.getDimensionY()),
            verbose=0
        )
    )

    for id_center in range(nb_centers):
        logging.info("[Sampling] Processsing center_{} ...".format(id_center))
        is_results.append(
            ph_model.importanceSampling(
                predictions.mergedGMM,
                all_reflectances, 
                all_incertitudes,
                *_option_values(is_attrs, IS_OPTIONS),
                covariance=np.zeros(ph_model.getDimensionY()),
                idx_gaussian=id_center,
                verbose=0
            )
        )

    return is_results


def predict(h5f, observations_file_path, output_dir):
    # get gllim model from file
    ds_name = H5_DATA_SETS["gllim_model"]
    if ds_name not in h5f:
        train_model(h5f)

    gllim = _gllim(h5f)

    ds = h5f.get(ds_name)
    gllim_params = pickle.loads(np.void(ds))
    gllim.setParams(gllim_params)

    # load observations ------------------
    observations = _load(observations_file_path)
    # load prediction configuration

    # predict ----------------------------
    nb_samples = len(observations)
    nb_wavelengths = len(observations[0])
    nb_geometries = len(observations[0][0]['reflectances'])
    nb_pred = nb_samples * nb_wavelengths
    # predictions = np.empty((nb_samples, nb_wavelengths), dtype=object)
    logging.info("[Prediction] Starting estimation of {} predictions composed of {} samples/scenes and {} wavelengths".format(nb_pred, nb_samples, nb_wavelengths))
    logging.info("[Prediction] Processsing ...")
    
    all_reflectances = np.empty((len(observations[0,0]['reflectances']), nb_pred))
    all_incertitudes = np.empty((len(observations[0,0]['incertitudes']), nb_pred))
    for id_sample, sample in enumerate(observations):
        for id_wavelength, wavelength in enumerate(sample):
            n_obs = id_wavelength + id_sample * nb_wavelengths
            all_reflectances[:,n_obs] = wavelength['reflectances']
            all_incertitudes[:,n_obs] = wavelength['incertitudes']

    predictions = gllim.inverseDensities(all_reflectances, all_incertitudes, *_prediction(h5f), 0) # PredictionResult(N_obs, L, K)
    logging.info("  Predictions step is done")

    if H5_GROUPS["importance_sampling"] in h5f:
        is_results = importance_sampling(h5f, all_reflectances, all_incertitudes, predictions, nb_pred)
        logging.info("  Important sampling step is done")

    # write to output folder ----------------

    # results = write_results(predictions, is_results, observations, physical_model)
    # logging.info("  Results are saved in files")


def main():
    short_descr = "xllim_cli.py - Command-line interface for xllim."
    epilog = "Run 'xllim.py COMMAND --help' for more information on a command."
    parser = argparse.ArgumentParser(description=short_descr, epilog=epilog)
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Print command
    print_parser = subparsers.add_parser(
        "print", help="Print contents of the h5 file")
    print_parser.add_argument(
        "target_file", help="Path to the model file (e.g., model.h5)")
    print_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output")

    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Edit configuration")
    edit_parser.add_argument(
        "target_file", help="Path to the model file (e.g., model.h5)")

    # Copy command
    import_parser = subparsers.add_parser(
        "cp", help="Copy data from one hfd5 file to another")
    import_parser.add_argument(
        "source_file", help="Path to the source file (e.g. experiment1.h5)")
    import_parser.add_argument(
        "target_file", help="Path to the target file (e.g., experiment2.h5)")

    # Generate command
    train_parser = subparsers.add_parser(
        "generate", help="Generate sythetic data")
    train_parser.add_argument(
        "target_file", help="Path to the model file (e.g., model.h5)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "target_file", help="Path to the model file (e.g., model.h5)")

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Make predictions using the model")
    predict_parser.add_argument(
        "model_file", help="Path to the model file (e.g., model.h5)")
    predict_parser.add_argument(
        "observations_file", help="Path to the observations file (e.g., observations.json)")
    predict_parser.add_argument(
        "output_directory", help="Path to the output directory. It will be created if needed.")

    # Import command
    import_parser = subparsers.add_parser(
        "import", help="Import data into the model")
    import_parser.add_argument("import_type", choices=[
                               "geometries", "train-data"], help="Type of data to import")
    import_parser.add_argument(
        "source_file", help="Path to the source file (e.g. geometries.json)")
    import_parser.add_argument(
        "target_file", help="Path to the target model file (e.g., target_model.h5)")

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export data from the model")
    export_parser.add_argument(
        "source_file", help="Path to the model file (e.g., experiment_1.h5)")
    export_parser.add_argument(
        "target_file", help="Path to the output file (e.g. predictions.json)")

    # Parse arguments
    args = parser.parse_args()

    if hasattr(args, 'target_file'):
        h5_file_name = args.target_file
    else:
        h5_file_name = args.model_file

    if hasattr(args, 'source_file'):
        if not os.path.isfile(args.source_file):
            print(f"source_file must exist. Aborting")
            return

    # process read-only commands
    if args.command in ('print', 'export'):
        if not os.path.isfile(h5_file_name):
            print(f"File {h5_file_name} not found")
            return
        with h5py.File(h5_file_name, 'r') as h5f:
            if args.command == "print":
                print_h5(h5f, args.verbose)
                return
            elif args.command == "export":
                export_data(args.export_type, h5f, args.output_file)
                return

    # process read-write commands
    # open the file for wrting
    if not os.path.isfile(h5_file_name):
        print(f"Creating new file: {h5_file_name}")

    with h5py.File(h5_file_name, 'a') as h5f:
        if args.command == "edit":
            edit_config(h5f)
        elif args.command == "cp":
            copy_h5_section_dialog(args.source_file, h5f)
        elif args.command == "generate":
            generate_data(h5f)
        elif args.command == "train":
            train_model(h5f)
        elif args.command == "predict":
            if not os.path.exists(args.output_directory):
                os.makedirs(args.output_directory)
            predict(h5f, args.observations_file, args.output_directory)
        elif args.command == "import":
            import_data(args.import_type, args.source_file, h5f)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
