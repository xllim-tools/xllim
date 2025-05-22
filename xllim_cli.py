#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Command line script for xllim.
#
# Copyright (C) 2025 Inria

import argparse
import ast
import json
import pickle
import h5py
from zipfile import ZipFile
import logging
import os
# import fnmatch
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional, Union

# Attempt Gdal import (try common namespaces)
try:
    from osgeo import gdal
except ImportError:
    try:
        import gdal
    except ImportError:
        from unittest.mock import Mock
        gdal = Mock()
        print("\nWARNING: GDAL not found. ENVI file operations will not work.\n")

# Attempt xllim import, provide a basic mock if not found
try:
    import xllim
except ImportError as e:
    from unittest.mock import Mock
    print(f"\nWARNING: Import error: {e}.\nxllim library not found. Commands generate, train, and predict will not work.\n")
    xllim = Mock()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("xllim_cli")


# --- HDF5 Constants ---
H5_STRING = h5py.string_dtype(encoding='utf-8')
H5_INT = np.int32
H5_FLOAT = np.float64
H5_BOOL = bool

class H5Group():
    def __init__(self, name: str, h5_path: str, conf_function=None, datasets=None):
        self.h5_path = h5_path
        self.name = name
        self.config_function = conf_function
        self.datasets = datasets

    def exist(self, h5f: h5py.File):
        return self.h5_path in h5f
    
    def ensure_exist(self, h5f: h5py.File):
        if self.h5_path not in h5f:
            h5f.create_group(self.h5_path, track_order=True)
    
    def get_value(self, h5f: h5py.File, key: str):
        if self.h5_path in h5f:
            attrs = h5f[self.h5_path].attrs
            if key not in attrs:
                return None
            return attrs[key]
    
    def set_value(self, h5f: h5py.File, key: str, value, dtype):
        self.ensure_exist(h5f)
        attrs = h5f[self.h5_path].attrs
        if key in attrs:
            attrs.modify(key, value)
        else:
            attrs.create(key, value, dtype=dtype)
        return
    
    def get_data_set(self, h5f: h5py.File, ds_name: str):
        dataset_path = self.h5_path + "/" + ds_name
        if dataset_path not in h5f:
            return None
        return h5f[dataset_path][()] # [()] reads data into memory
    
    def set_data_set(self, h5f: h5py.File, name: str, data, dtype=H5_FLOAT):
        dataset_path = self.h5_path + "/" + name
        if dataset_path in h5f:
            logger.debug(f"Replacing existing HDF5 dataset: {dataset_path}")
            del h5f[dataset_path]
        
        logger.debug(f"Writing HDF5 dataset: {dataset_path} with shape {getattr(data, 'shape', 'N/A')}")
        h5f.create_dataset(dataset_path, data=data, dtype=dtype)
        return
    
    def print_datasets(self, h5f: h5py.File, verbose=False):
        if self.h5_path in h5f:
            for key, value in h5f[self.h5_path].items():
                if verbose:
                    print(f"    {key} : {value[()]}")
                else:
                    print(f"    {key} : {value[()].shape}")
        return
    
    def option_values(self, h5f: h5py.File, options):
        """Extracts values for specified option keys from HDF5 attributes.

        Args:
            attrs: h5py AttributeManager object.
            options: Tuple of tuples where the first element is the attribute key.

        Returns:
            List of attribute values in the order specified by options.

        Raises:
            KeyError: If a required attribute is missing.
        """
        values = []
        for opt in options:
            key = opt[0]
            values.append(self.get_value(h5f, key))
        return values
    
    def print(self, h5f: h5py.File):
        """Prints attributes of a configured HDF5 group."""
        if not self.exist(h5f):
            print(f"  {self.name}: Not configured")
            return 

        print(f"  {self.name} configuration:")
        try:
            attrs = h5f[self.h5_path].attrs
            if not attrs:
                print("\t(No attributes)")
            else:
                for key, value in attrs.items():
                    print(f"\t{key} : {value}")
        except Exception as e:
            print(f"\tError reading attributes: {e}")
    

H5_DATA_SETS = {"geometries": "/functional_model/geometries",
                "train_data": "/train_data",
                "gllim_model": "/gllim/serialised_gllim"}

# --- HDF5 datasets --------
GEOMETRIES = H5Group("geometries", "/functional_model/geometries", datasets=["sza", "vsa", "zsa"])
TRAIN_DATA = H5Group("train_data", "/train_data", datasets=["X", "Y"])
TRAINED_MODEL = H5Group("trained_gllim", "/gllim", datasets=["serialised_gllim"])

# --- HDF5 configurable sections --------
FUNCTIONAL = H5Group("functional", "/functional_model/config")
GENERATOR = H5Group("generator", "/functional_model/data_generator")
GLLIM_MODEL = H5Group("gllim_model", "/gllim")
PREDICTION = H5Group("prediction_module", "/prediction_module_config")
IMPORTANCE_SAMPLING = H5Group("importance_sampling", "/importance_sampling_config")
OUTPUT = H5Group("output", "/output_config")

H5_SECTIONS = [FUNCTIONAL, GENERATOR, GLLIM_MODEL, IMPORTANCE_SAMPLING, PREDICTION, OUTPUT]

# --- Configuration Options ---
# Format: (attribute_name, help_string, allowed_values_tuple_or_None, hdf5_dtype)
SUPPORTED_MODELS = (("model", "functional model", ("Hapke", "Shkuratov", "External", "Test model"), H5_STRING),)
HAPKE_OPTIONS = (("variant", "", ("1993", "2002"), H5_STRING),
                 ("adapter", "Number of Hapke's model parameters", ("three", "four", "six"), H5_STRING),
                 ("theta_bar_maximum", "Value used to transform theta_bar (e.g. 30)", None, H5_INT),
                 ("b0", "Amplitude of the opposition effect", None, H5_FLOAT),
                 ("h", "Angular width of the opposition effect (0, XXX)", None, H5_FLOAT))
SHKURATOV_OPTIONS = (("variant", "Number of model parameters", ("3p", "5p"), H5_STRING),
                     ("scaling_coeffs", "List of L scaling coeffs (e.g., '[0.1, 0.2]')", None, H5_STRING),
                     ("offset", "List of L offsets (e.g., '[0.0, 0.0]')", None, H5_STRING))
EXTERNAL_MODEL_OPTIONS = (("class_name", "Python class name", None, H5_STRING),
                          ("file_name", "Python file name (.py)", None, H5_STRING),
                          ("file_path", "Path to the Python file", None, H5_STRING))
GENERATOR_OPTIONS = (("N", "Dataset size (positive integer)", None, H5_INT),
                     ("type", "Generator type", ("sobol", "random", "latin hypercube"), H5_STRING),
                     ("covariance", "Scalar covariance value (applied to all D dims)", None, H5_FLOAT),
                     ("seed", "Seed for random generator (integer)", None, H5_INT))
GLLIM_OPTIONS = (('K', 'Number of GLLiM components (integer)', None, H5_INT),
                 ('Gamma_type', 'Covariance type for K GLLiM components', ("full", "diag", "iso"), H5_STRING),
                 ('Sigma_type', 'Covariance type for Gaussian noise', ("full", "diag", "iso"), H5_STRING),
                 ('n_hidden', 'Number of hidden variables (integer, optional)', None, H5_INT))
GLLIM_INIT_OPTIONS = (('gllim_em_iteration', 'Number of EM iterations for GLLiM', None, H5_INT),
                      ('gllim_em_floor', 'Floor value for EM iterations in GLLiM', None, H5_FLOAT),
                      ('gmm_kmeans_iteration', 'Number of k-means iterations for GMM', None, H5_INT),
                      ('gmm_em_iteration', 'Number of EM iterations for GMM', None, H5_INT),
                      ('gmm_floor', 'Floor value for EM iterations in GMM', None, H5_FLOAT),
                      ('nb_experiences', 'Number of initialization experiences', None, H5_INT),
                      ('seed', 'Random number seed for initialization', None, H5_INT))
GLLIM_TRAIN_VARIANTS = (('train_variant', 'Training method', ('GLLiM', 'JGMM'), H5_STRING),) # Simplified options if 'GLLiM and JGMM' isn't a direct choice
GLLIM_TRAIN_OPTIONS = (('train_max_iteration', 'Maximum number of iterations', None, H5_INT),
                       ('train_ratio_ll', 'Ratio for log-likelihood convergence', None, H5_FLOAT),
                       ('train_floor', 'Floor value for the training process', None, H5_FLOAT))
JGMM_TRAIN_OPTIONS = (('jgmm_train_kmeans_iteration', 'Number of k-means iterations', None, H5_INT),
                      ('jgmm_train_em_iteration', 'Number of EM iterations', None, H5_INT),
                      ('jgmm_train_floor', 'Variance floor for diagonal covariances', None, H5_FLOAT))
PREDICTION_OPTIONS = (('K_merged', 'Merge GMM to K_merged components', None, H5_INT),
                      ('merging_threshold', 'Threshold on merged GMM weights', None, H5_FLOAT),
                      ('relative_uncertainty', 'Relative uncertainty if not provided as Y_u in observations (e.g., 0.1 for 10%)', None, H5_FLOAT))
IS_OPTIONS = (('N_zero', 'Initial samples (IMIS, default: N/10)', None, H5_INT),
              ('B', 'New samples per step (IMIS, default: N/20)', None, H5_INT),
              ('J', 'Number of iterations (IMIS, default: 18)', None, H5_INT))
IS_ON_FULL_OPTIONS = (('IS_on_fullGMM', 'Do importance sampling on full GMM', None, H5_BOOL),)
IS_ON_X_OPTIONS = (('IS_on_mergedGMM', 'Do importance sampling on merged GMM', None, H5_BOOL),
                  ('IS_on_centers', 'Do importance sampling on each center', None, H5_BOOL))
OUTPUT_OPTIONS = (('gllim_fullGMM', 'Write mean prediction', None, H5_BOOL),
                  ('gllim_mergedGMM', 'Write mean merged prediction', None, H5_BOOL),
                  ('IS_fullGMM', 'Write IS on fullGMM prediction', None, H5_BOOL),
                  ('IS_mergedGMM', 'Write IS on mergedGMM prediction', None, H5_BOOL),
                  ('IS_centers', 'Write IS on merged centers prediction', None, H5_BOOL),)


# --- Configuration Functions ---

def config_dialog(h5f: h5py.File, section: H5Group, options: Tuple[Tuple[str, str, Optional[Tuple[str, ...]], Any], ...]) -> List[str]:
    """Interactively configure attributes for an HDF5 group.

    Args:
        h5f: Open HDF5 file object (write mode).
        group_path: Path to the HDF5 group.
        options: Tuple of tuples defining configuration options.
                 Format: (attr_name, help_text, choices_tuple | None, h5_dtype)

    Returns:
        List of attribute names that were set or modified.
    """
    modified = []
    section.ensure_exist(h5f)
    try:
        for name, help_text, vals, dtype in options:
            current_value = section.get_value(h5f, name)
            cval_string = f" [{current_value}]" if current_value is not None else ""
            help_string = f" ({help_text})" if help_text else ""

            user_input = None
            prompt_base = f"{name}{help_string}{cval_string}"

            if dtype == H5_BOOL:  # boolean option "yes/no"
                while user_input is None:
                    prompt = f"{prompt_base} yes/no (y/n): "
                    raw_input = input(prompt).strip()
                    if not raw_input and current_value is not None: # User hit enter, keep current
                        user_input = current_value
                        break
                    if raw_input == 'y' or raw_input == 'yes':
                        user_input = True
                    elif raw_input == 'n' or raw_input == 'no':
                        user_input = False
                    else:
                        print("Please enter 'y' or 'n'.")
            elif isinstance(vals, tuple):
                print(f"\nChoose {name}{help_string}:")
                for i, v in enumerate(vals):
                    print(f"  {i+1}. {v}")
                prompt = f"Enter choice number{cval_string}: "
                while user_input is None:
                    raw_input = input(prompt).strip()
                    if not raw_input and current_value is not None: # User hit enter, keep current
                        user_input = current_value
                        break
                    try:
                        choice_idx = int(raw_input) - 1
                        if 0 <= choice_idx < len(vals):
                            user_input = vals[choice_idx]
                        else:
                            print("Invalid choice number.")
                    except ValueError:
                        print("Please enter a number.")
            else: # Free text input
                prompt = f"{prompt_base}: "
                raw_input = input(prompt).strip()
                if not raw_input and current_value is not None:
                     user_input = current_value
                elif raw_input:
                    try:
                        if dtype == H5_FLOAT:
                             user_input = float(raw_input)
                        elif dtype == H5_INT:
                             user_input = int(raw_input)
                        else: # String or other
                             user_input = raw_input
                    except ValueError:
                        print(f"Invalid input. Expected type compatible with {dtype}.")
                        user_input = None # Force re-ask or handle as error? For now, just mark unset.

            # Modify or create attribute if a valid value was provided or kept
            if user_input is not None:
                section.set_value(h5f, name, user_input, dtype)
            print(f"\033[F\033[{len(prompt)}G {user_input}")
            modified.append(name)

        h5f.flush()
        return modified
    except Exception as e:
        logger.error(f"Error during configuration dialog for {section.h5_path}: {e}")
        raise # Re-raise after logging


def delete_all_attributes_except(h5f: h5py.File, group_path: str, attrs_to_keep: List[str]):
    """Deletes attributes from an HDF5 group except those specified."""
    if group_path not in h5f:
        logger.warning(f"Group {group_path} not found for attribute cleanup.")
        return
    try:
        attrs = h5f[group_path].attrs
        attrs_to_delete = [a for a in attrs if a not in attrs_to_keep]
        for at in attrs_to_delete:
            del attrs[at]
            logger.info(f"Removed unused attribute '{at}' from {group_path}")
    except Exception as e:
        logger.error(f"Error cleaning up attributes in {group_path}: {e}")


def configure_generator(h5f: h5py.File):
    changed = config_dialog(h5f, GENERATOR, GENERATOR_OPTIONS)
    delete_all_attributes_except(h5f, GENERATOR.h5_path, changed)

def configure_gllim_model(h5f: h5py.File):
    changed = config_dialog(h5f, GLLIM_MODEL, GLLIM_OPTIONS)
    changed += config_dialog(h5f, GLLIM_MODEL, GLLIM_INIT_OPTIONS)
    changed += config_dialog(h5f, GLLIM_MODEL, GLLIM_TRAIN_VARIANTS)

    train_variant = GLLIM_MODEL.get_value('train_variant')
    if train_variant == 'GLLiM':
        changed += config_dialog(h5f, GLLIM_MODEL, GLLIM_TRAIN_OPTIONS)
    elif train_variant == 'JGMM':
        changed += config_dialog(h5f, GLLIM_MODEL, JGMM_TRAIN_OPTIONS)

    delete_all_attributes_except(h5f, GLLIM_MODEL.h5_path, changed)

def configure_prediction_module(h5f: h5py.File):
    # print("\nConfiguring prediction options:")
    changed = config_dialog(h5f, PREDICTION, PREDICTION_OPTIONS)
    changed += config_dialog(h5f, PREDICTION, IS_ON_FULL_OPTIONS)
    K_merged = PREDICTION.get_value(h5f, "K_merged")
    if K_merged > 0:
        changed += config_dialog(h5f, PREDICTION, IS_ON_X_OPTIONS)
    delete_all_attributes_except(h5f, PREDICTION.h5_path, changed)

def configure_importance_sampling(h5f: h5py.File):
    changed = config_dialog(h5f, IMPORTANCE_SAMPLING, IS_OPTIONS)
    delete_all_attributes_except(h5f, IMPORTANCE_SAMPLING.h5_path, changed)

def configure_output(h5f: h5py.File):
    changed = config_dialog(h5f, OUTPUT, OUTPUT_OPTIONS)
    is_full = PREDICTION.get_value(h5f, "IS_on_fullGMM")
    K_merged = PREDICTION.get_value(h5f, "K_merged")
    is_merged, is_centers = PREDICTION.option_values(h5f, IS_ON_X_OPTIONS)
    w_full, w_merged, w_is_full, w_is_merges, w_is_centers = OUTPUT.option_values(h5f, OUTPUT_OPTIONS)
    if K_merged == 0 and (w_merged or w_is_merges or w_is_centers):
        logger.warning("K_merged is 0. Results based on mergedGMM will not be produced.")
    if w_is_full and not is_full:
        logger.warning("Importance Sampling on fullGMM is not set in prediction step. The IS_fullGMM results won't be written.")
    if w_is_merges and not is_merged:
        logger.warning("Importance Sampling on mergedGMM is not set in prediction step. The IS_mergedGMM result won't be written.")
    if w_is_centers and not is_centers:
        logger.warning("Importance Sampling on centers is not set in prediction step. The IS_centers result won't be written.")
    delete_all_attributes_except(h5f, OUTPUT.h5_path, changed)

def configure_functional(h5f: h5py.File):
    changed_attrs = config_dialog(h5f, FUNCTIONAL, SUPPORTED_MODELS)

    model_type = FUNCTIONAL.get_value(h5f, "model")
    if model_type == "External":
        changed_attrs += config_dialog(h5f, FUNCTIONAL, EXTERNAL_MODEL_OPTIONS)
    elif model_type == "Hapke":
        changed_attrs += config_dialog(h5f, FUNCTIONAL, HAPKE_OPTIONS)
    elif model_type == "Shkuratov":
        changed_attrs += config_dialog(h5f, FUNCTIONAL, SHKURATOV_OPTIONS)
    elif model_type == "Test model":
        pass # No specific options
    else:
        logger.error(f"Invalid functional model type configured: {model_type}")
        # Don't delete attributes if config is potentially broken
        return

    delete_all_attributes_except(h5f, FUNCTIONAL.h5_path, changed_attrs)


# --- Utility Functions ---

def _geometries(h5f: h5py.File) -> np.ndarray:
    """Reads geometries (sza, vza, phi) from the HDF5 file."""
    if not GEOMETRIES.exist(h5f):
        raise ValueError(f"HDF5 dataset '{GEOMETRIES.name}' not found.")

    required_datasets = ["sza", "vza", "phi"]
    data_list = []

    for ds_name in required_datasets:
        ds = GEOMETRIES.get_data_set(h5f, ds_name)
        if ds is None:
            raise ValueError(f"Dataset '{ds_name}' missing in '{GEOMETRIES.name}'.")
        data_list.append(ds)

    # Stack horizontally: N x 3
    return np.stack(data_list, axis=-1).astype(H5_FLOAT)

def _shkuratov_config(h5f) -> Tuple[str, List[float], List[float]]:
    """Parses Shkuratov model configuration from attributes."""
    variant, s_str, o_str = FUNCTIONAL.option_values(h5f, SHKURATOV_OPTIONS)
    try:
        scaling = ast.literal_eval(s_str)
        offsets = ast.literal_eval(o_str)
        if not isinstance(scaling, list) or not isinstance(offsets, list):
             raise TypeError("Scaling coefficients and offsets must be lists.")
        if len(scaling) != len(offsets):
            raise ValueError("Scaling and offset lists must have the same length.")
        # Further check if elements are numbers?
        scaling = [float(x) for x in scaling]
        offsets = [float(x) for x in offsets]
        return variant, scaling, offsets
    except (ValueError, SyntaxError, TypeError) as e:
        logger.error(f"Error parsing Shkuratov 'scaling_coeffs' or 'offset': {e}. Ensure they are valid Python list strings (e.g., '[0.1, 0.2]').")
        raise ValueError("Invalid Shkuratov configuration.") from e

def _physical_model(h5f: h5py.File) -> Any:
    """Instantiates the physical model based on HDF5 configuration."""
    if not FUNCTIONAL.exist(h5f):
        return None
    
    model_type = FUNCTIONAL.get_value(h5f, "model")

    try:
        if model_type == "Test model":
            # Assumes TestModel needs no extra config from HDF5
            f_model = xllim.TestModel()
        elif model_type == "Hapke":
            geoms = _geometries(h5f) # Raises ValueError if missing
            hapke_params = FUNCTIONAL.option_values(h5f, HAPKE_OPTIONS) # Raises KeyError if missing
            f_model = xllim.HapkeModel(geoms, *hapke_params)
        elif model_type == "Shkuratov":
            geoms = _geometries(h5f)
            variant, scalingCoeffs, offset = _shkuratov_config(h5f) # Raises ValueError/KeyError
            logger.info(f"Instantiating ShkuratovModel(geoms, variant={variant}, scaling={scalingCoeffs}, offset={offset})")
            f_model = xllim.ShkuratovModel(geoms, variant, scalingCoeffs, offset)
        elif model_type == "External":
            ext_params = FUNCTIONAL.option_values(h5f, EXTERNAL_MODEL_OPTIONS)
            # Add validation for file existence if needed
            f_model = xllim.ExternalPythonModel(*ext_params)
        else:
            raise ValueError(f"Unsupported functional model type: {model_type}")

        logger.info(f"Successfully instantiated physical model: {model_type}")
        return f_model
    except (ValueError, KeyError, ImportError, AttributeError) as e:
        logger.error(f"Failed to instantiate physical model '{model_type}': {e}")
        raise RuntimeError(f"Could not create physical model '{model_type}'.") from e


def _train_data(h5f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    """Loads training data (X, Y) from HDF5 file, or generates it if missing."""
    if not TRAIN_DATA.exist():
        return generate_data(h5f)

    X = TRAIN_DATA.get_data_set(h5f, "X")
    Y = TRAIN_DATA.get_data_set(h5f, "Y")
    if X is None or Y is None:
        raise ValueError(f"Datasets 'X' or 'Y' missing in '{TRAIN_DATA.name}'.")

    # Basic dimension check might be useful here
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        logger.warning(f"Training data dimensions seem unusual: X{X.shape}, Y{Y.shape}. Expected (L, N) and (D, N).")
    logger.info(f"Loaded training data: X{X.shape}, Y{Y.shape}")
    return X, Y


def _prediction_config(h5f: h5py.File) -> Tuple[int, float]:
    """Loads prediction module configuration."""
    if not PREDICTION.exist(h5f):
        raise ValueError(f"Prediction module configuration group '{PREDICTION.h5_path}' not found.")

    try:
        return PREDICTION.option_values(h5f, PREDICTION_OPTIONS)
    except KeyError as e:
        logger.error(f"Missing prediction configuration: {e}")
        raise ValueError("Incomplete prediction configuration.") from e

def _load_gllim_model(h5f: h5py.File) -> Any:
    """Loads serialized GLLiM parameters from HDF5 and sets them."""
    gllim_pickle = TRAINED_MODEL.get_data_set(h5f, "serialised_gllim")
    if gllim_pickle is None:
        logger.error(f"Serialized GLLiM model not found.")
        return train_model(h5f)

    try:
        D = TRAIN_DATA.get_value(h5f, "D")
        L = TRAIN_DATA.get_value(h5f, "L")
        K, gamma_type, sigma_type, n_hidden = GLLIM_MODEL.option_values(h5f, GLLIM_OPTIONS)
        logger.info(f"Instantiating GLLiM(K={K}, D={D}, L={L}, gamma={gamma_type}, sigma={sigma_type}, n_hidden={n_hidden})")
        gllim_instance = xllim.GLLiM(K, D, L, gamma_type, sigma_type, n_hidden)

        gllim_params = pickle.loads(gllim_pickle.tobytes())
        gllim_instance.setParams(gllim_params)
        logger.info(f"Successfully loaded and set GLLiM parameters")
        return gllim_instance
    except (pickle.UnpicklingError, TypeError, AttributeError, KeyError) as e:
        logger.error(f"Failed to load or set GLLiM parameters: {e}")
        raise RuntimeError("Could not load trained GLLiM model.") from e

# --- Core Command Functions ---

def print_h5(h5f: h5py.File, verbose: bool):
    """Prints a summary or detailed view of the HDF5 file contents."""
    print(f"--- Contents of {h5f.filename} ---")

    # Check physical model setup
    model_type = "N/A"
    geom_shape = None
    has_functional = FUNCTIONAL.exist(h5f)
    has_generator = GENERATOR.exist(h5f)
    if has_functional:
        try:
            model_type = FUNCTIONAL.get_value(h5f, "model")
            if model_type in ("Hapke", "Shkuratov") and GEOMETRIES.exist(h5f):
                geom_shape = GEOMETRIES.get_data_set(h5f, "sza").shape # Get shape from one dataset
        except Exception as e:
            logger.warning(f"Could not fully read functional model config: {e}")

    if has_functional and has_generator:
         geom_str = f", geometries{geom_shape}" if geom_shape else ""
         print(f"Direct model setup: YES ({model_type}{geom_str})")
    elif has_functional:
         print(f"Direct model setup: PARTIAL (functional config present, generator missing)")
    else:
         print(f"Direct model setup: NO")

    # Check train data
    train_data_shape_str = "NO"
    X = TRAIN_DATA.get_data_set(h5f, "X")
    Y = TRAIN_DATA.get_data_set(h5f, "Y")
    if X is not None and Y is not None:
        train_data_shape_str = f"YES (X{X.shape}  Y{Y.shape})"
    print(f"Train data:       {train_data_shape_str}")

    # Check trained GLLiM model
    print(f"Trained model:    {'NO' if TRAINED_MODEL.get_data_set(h5f, "serialised_gllim") is None else 'YES'}")

    if verbose:
        print("\n--- Detailed Configuration ---")
        for section in H5_SECTIONS:
            section.print(h5f)
            print("") # Spacer

    GEOMETRIES.print_datasets(h5f, verbose=True)

def edit_config(h5f: h5py.File):
    """Walks through configuration groups for editing."""
    print("\n--- Interactive Configuration Editor ---")
    # Define the mapping from group key to configure function

    for section in H5_SECTIONS:
        print("-" * 20)
        prompt_action = "Edit" if section.exist(h5f) else "Add"

        if section.exist(h5f):
            section.print(h5f)

        while True:
            k = input(f"{prompt_action} {section.name} configuration? (y/n/q): ").lower().strip()
            if k == 'y':
                try:
                    section.config_function(h5f)
                    break # Move to next group
                except Exception as e:
                    logger.error(f"Configuration failed for {section.name}: {e}. Skipping.")
                    break # Skip this group on error
            elif k == 'n':
                break # Move to next group
            elif k == 'q':
                print("Exiting configuration.")
                return # Exit the entire edit process
            else:
                print("Invalid input. Please enter 'y', 'n', or 'q'.")
    print("--- Configuration finished ---")


def import_geometries(source_path: str, dest_h5f: h5py.File):
    """Imports geometries from a JSON file into the HDF5 file."""
    _, file_extension = os.path.splitext(source_path)

    if file_extension != '.json':
        raise ValueError("Geometries source file must be a JSON file.")

    try:
        logger.info(f"Importing geometries from {source_path} into {GEOMETRIES.h5_path}")

        GEOMETRIES.ensure_exist(dest_h5f)

        with open(source_path, 'r') as json_file:
            data = json.load(json_file)
            if not isinstance(data, dict):
                raise ValueError("JSON root must be an object (dictionary).")
            # Basic validation: check for expected keys?
            expected_keys = ["sza", "vza", "phi"] # Adjust if needed
            data_length = -1
            for k, v in data.items():
                if not isinstance(v, list):
                     raise ValueError(f"Value for key '{k}' in JSON must be a list.")
                if not all(isinstance(x, (int, float)) for x in v):
                     raise ValueError(f"All elements in list '{k}' must be numbers.")
                if data_length == -1:
                     data_length = len(v)
                elif data_length != len(v):
                     raise ValueError("All geometry lists in JSON must have the same length.")

                # Check if key is one of the expected ones (optional but good practice)
                # if k not in expected_keys:
                #     logger.warning(f"Importing unexpected geometry key '{k}' from JSON.")

                GEOMETRIES.set_data_set(dest_h5f, k, np.array(v, dtype=H5_FLOAT))
                logger.debug(f"Created dataset '{k}' with shape {GEOMETRIES.get_data_set(dest_h5f, k).shape}")

        # Add source attribute
        GEOMETRIES.set_value(dest_h5f, 'source', os.path.basename(source_path), dtype=H5_STRING)
        dest_h5f.flush()
        logger.info(f"Geometries import successful.")

    except FileNotFoundError:
        logger.error(f"Source JSON file not found: {source_path}")
        raise
    except (IOError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error importing geometries: {e}")
        raise


def _h5_copy(group, dst_h5f: h5py.File):
    """Copies a group from a source HDF5 file to the destination."""
    # TODO
    try:
        with h5py.File(src_h5_path, 'r') as src_h5f:
            if group_or_ds_name not in src_h5f:
                raise ValueError(f"Object '{group_or_ds_name}' not found in source file {src_h5_path}")

            if group_or_ds_name in dst_h5f:
                logger.warning(f"Replacing existing object '{group_or_ds_name}' in destination file.")
                del dst_h5f[group_or_ds_name]

            logger.info(f"Copying '{group_or_ds_name}' from {src_h5_path} to {dst_h5f.filename}")
            src_obj = src_h5f[group_or_ds_name]
            # Copy object (group or dataset)
            dst_h5f.copy(src_obj, group_or_ds_name)
            dst_h5f.flush()
            logger.debug(f"Copy successful for '{group_or_ds_name}'.")

    except (IOError, ValueError, KeyError) as e:
        logger.error(f"Error copying '{group_or_ds_name}' from {src_h5_path}: {e}")
        raise


def import_data(what_to_import: str, source_file_path: str, h5f: h5py.File):
    """Imports specified data type from source file into the HDF5 file."""
    logger.info(f"Attempting to import '{what_to_import}' from '{source_file_path}'")
    try:
        if what_to_import == "geometries":
            import_geometries(source_file_path, h5f)
        elif what_to_import == "train-data":
            if not source_file_path.lower().endswith('.npz'):
                raise ValueError("Train data source file must be an NPZ file.")
            logger.info(f"Importing training data from NPZ: {source_file_path}")
            npz_data = np.load(source_file_path)
            if 'X' not in npz_data or 'Y' not in npz_data:
                raise ValueError("NPZ file for training data must contain 'X' and 'Y' arrays.")
            
            X_data = npz_data['X'].astype(H5_FLOAT)
            Y_data = npz_data['Y'].astype(H5_FLOAT)

            if X_data.ndim != 2 or Y_data.ndim != 2 or X_data.shape[1] != Y_data.shape[1]:
                logger.warning(f"Imported training data dimensions unusual: X{X_data.shape}, Y{Y_data.shape}. Expected (L, N) and (D, N).")

            TRAIN_DATA.set_data_set(h5f, "X", X_data, H5_FLOAT)
            TRAIN_DATA.set_data_set(h5f, "Y", Y_data, H5_FLOAT)
            TRAIN_DATA.set_value(h5f, "L", X_data.shape[0], dtype=H5_INT)
            TRAIN_DATA.set_value(h5f, "D", Y_data.shape[0], dtype=H5_INT)
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_text = f"Imported from {os.path.basename(source_file_path)} by xllim_cli on {now}"
            TRAIN_DATA.set_value(h5f, "source", info_text, dtype=H5_STRING)
            h5f.flush()
            logger.info(f"Successfully imported training data: X{X_data.shape}, Y{Y_data.shape}")
        else:
            # This case should ideally be caught by argparse choices
            raise ValueError(f"Invalid import type specified: {what_to_import}")
    except (FileNotFoundError, ValueError, IOError, RuntimeError) as e:
        # Errors already logged in lower functions, just indicate failure
        logger.error(f"Import failed: {e}")
        # Re-raise specific error types if needed by caller
        raise RuntimeError(f"Import of '{what_to_import}' failed.") from e


def copy_h5_section_dialog(src_h5_path: str, dst_h5f: h5py.File):
    """Interactively copies sections from a source HDF5 file to the destination."""
    print(f"\n--- Copy Sections from {src_h5_path} ---")
    available_groups = []
    for group in H5_SECTIONS:
        if group.exist(src_h5_path):
            available_groups.append(group)
    if TRAIN_DATA.exist():  # has it's own non-configurable group
        available_groups.append(TRAIN_DATA)

    if len(available_groups) == 0:
        print("Nothing to copy in the source file.")
        return

    print("Sections available to copy:")
    for idx, group in enumerate(available_groups):
        print(f"  {idx}. {group.name}")
    idxses = range(len(available_groups))

    while True:
        input_str = input("Enter numbers of sections to copy (comma-separated, e.g., 1,3) or 'q' to cancel: ").strip()
        if input_str.lower() == 'q':
            print("Copy operation cancelled.")
            return

        selected_indices = []
        try:
            selected_indices = [int(x.strip()) for x in input_str.split(',') if x.strip()]
            if not all(idx in idxses for idx in selected_indices):
                raise ValueError("Invalid selection number.")
            break # Valid input
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers corresponding to the list above.")

    copy_count = 0
    fail_count = 0
    for idx in selected_indices:
        try:
            _h5_copy(available_groups[idx], dst_h5f)
            copy_count += 1
        except Exception:
            # Error logged in _h5_copy
            fail_count += 1

    print(f"--- Copy finished: {copy_count} sections copied, {fail_count} failed. ---")

def generate_data(h5f: h5py.File):
    """Generates synthetic training data using the configured physical model and generator."""
    logger.info("Attempting to generate training data...")
    try:
        # Ensure prerequisites are met
        if not FUNCTIONAL.exist(h5f) or not GENERATOR.exist(h5f):
            raise ValueError("Functional model and/or generator configuration missing.")

        f_model = _physical_model(h5f) # Can raise exceptions
        if not f_model:
            raise RuntimeError("Cannot generate train data wihtout direct model functional.")

        N, generator_type, covariance_scalar, seed = GENERATOR.option_values(h5f, GENERATOR_OPTIONS) # Can raise KeyError

        D = f_model.getDimensionY()
        # Create covariance vector
        covariance_vector = np.ones(D, dtype=H5_FLOAT) * covariance_scalar

        logger.info(f"Calling genData(N={N}, type='{generator_type}', cov={covariance_vector}, seed={seed}) using {type(f_model).__name__}")
        x_gen, y_gen = f_model.genData(N, generator_type, covariance_vector, seed)

        # Validate generated data shapes (simple check)
        L_model = f_model.getDimensionX()
        if x_gen.shape != (L_model, N) or y_gen.shape != (D, N):
             logger.warning(f"Generated data shape mismatch: X_gen{x_gen.shape} (expected {(L_model, N)}), Y_gen{y_gen.shape} (expected {(D, N)})")

        # Store data
        TRAIN_DATA.set_data_set(h5f, "X", x_gen, H5_FLOAT)
        TRAIN_DATA.set_data_set(h5f, "Y", y_gen, H5_FLOAT)
        TRAIN_DATA.set_value(h5f, "D", D, H5_INT)
        TRAIN_DATA.set_value(h5f, "L", L_model, H5_INT)

        # Add metadata
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = f"Generated by xllim_cli using {type(f_model).__name__} on {now}"
        TRAIN_DATA.set_value(h5f, "source", info_text, H5_STRING)

        h5f.flush()
        logger.info(f"Successfully generated and saved training data: X{x_gen.shape}, Y{y_gen.shape}")
        return x_gen, y_gen

    except (ValueError, KeyError, RuntimeError, AttributeError, TypeError) as e:
        logger.error(f"Failed to generate data: {e}")
        # Don't raise further, just log the failure


def train_model(h5f: h5py.File):
    """Trains the GLLiM model using data and configuration from the HDF5 file."""
    logger.info("Attempting to train GLLiM model...")
    verbose_level = 1 # Or make configurable

    try:
        # 1. Load or Generate Training Data
        X, Y = _train_data(h5f)
        L = X.shape[0]
        D = Y.shape[0]

        # 2. Instantiate GLLiM model
        K, gamma_type, sigma_type, n_hidden = GLLIM_MODEL.option_values(h5f, GLLIM_OPTIONS)
        logger.info(f"Instantiating GLLiM(K={K}, D={D}, L={L}, gamma={gamma_type}, sigma={sigma_type}, n_hidden={n_hidden})")
        gllim_instance = xllim.GLLiM(K, D, L, gamma_type, sigma_type, n_hidden)

        # 3. Initialize GLLiM
        init_params = GLLIM_MODEL.option_values(h5f, GLLIM_INIT_OPTIONS) # Can raise KeyError
        logger.info(f"Initializing GLLiM with params: {init_params}")
        gllim_instance.initialize(X, Y, *init_params, verbose_level)
        logger.info("GLLiM initialization complete.")

        # 4. Train GLLiM
        train_variant = GLLIM_MODEL.option_values(h5f, GLLIM_TRAIN_VARIANTS)[0] # Can raise KeyError

        if train_variant == 'GLLiM':
            train_params = GLLIM_MODEL.option_values(h5f, GLLIM_TRAIN_OPTIONS) # Can raise KeyError
            logger.info(f"Starting GLLiM training with params: {train_params}")
            gllim_instance.train(X, Y, *train_params, verbose_level)
        elif train_variant == 'JGMM':
            train_params = GLLIM_MODEL.option_values(h5f, JGMM_TRAIN_OPTIONS) # Can raise KeyError
            logger.info(f"Starting JGMM training with params: {train_params}")
            gllim_instance.trainJGMM(X, Y, *train_params, verbose_level)
        else:
            # Should be caught during config, but double-check
            raise ValueError(f"Unsupported training variant: {train_variant}")

        logger.info(f"GLLiM training ({train_variant}) complete.")

        # 5. Save Trained Model
        gllim_params_obj = gllim_instance.getParams()
        # Optionally log some parameters
        # logger.debug(f"Trained Pi: {gllim_params_obj.Pi}")
        gllim_serialised = pickle.dumps(gllim_params_obj, pickle.HIGHEST_PROTOCOL)

        # Store pickled object as numpy void object in dataset
        TRAINED_MODEL.set_data_set(h5f, "serialised_gllim", np.void(gllim_serialised), np.void)
        h5f.flush()
        logger.info(f"Trained GLLiM model parameters saved")
        return gllim_instance

    except (FileNotFoundError, ValueError, KeyError, RuntimeError, pickle.PicklingError, IOError, AttributeError, TypeError) as e:
        logger.error(f"GLLiM training failed: {e}")
        # Don't raise further, failure is logged.


# --- Prediction Related Functions ---

def get_wavelengths_remote_sensing(data_rho: gdal.Dataset) -> List[str]:
    """Extracts sorted wavelengths from GDAL dataset metadata."""
    if not gdal: raise RuntimeError("GDAL is required for remote sensing data.")
    try:
        # Metadata might not be present or in expected format
        header = data_rho.GetMetadata_Dict() # e.g., {'Band_1': '450.0', 'Band_10': '900.0', ...}
        # Filter keys that look like band definitions and extract numeric part for sorting
        band_wl = {}
        for key, wl_str in header.items():
            if key.lower().startswith('band_'):
                try:
                    # Attempt to convert wavelength to float for sorting, keep original string
                    band_wl[float(wl_str)] = wl_str
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse wavelength '{wl_str}' for metadata key '{key}'. Skipping.")
        # Sort by float key, return original strings
        sorted_wavelengths = [band_wl[k] for k in sorted(band_wl.keys())]
        if not sorted_wavelengths:
             logger.warning("Could not extract wavelengths from GDAL metadata. Returning empty list.")
        return sorted_wavelengths
    except AttributeError:
         logger.warning("Could not read metadata using GetMetadata_Dict(). No wavelengths extracted.")
         return []


def get_reflectances_laboratory(lab_data: Dict[str, Any], relative_uncertainty: float) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parses laboratory reflectance data from a dictionary structure."""
    reflectances_list = []
    keys = []
    # Find keys containing reflectance data
    reflectance_keys = sorted([key for key in lab_data.keys() if key.startswith('reflectance')]) # Sort for consistent order

    for key in reflectance_keys:
        keys.append(key)
        sample_observations = []
        wavelength_data = lab_data[key] # Should be list of lists/tuples per wavelength

        # Infer number of wavelengths from the first sample's data structure
        if not wavelength_data:
             logger.warning(f"Empty data for sample key '{key}'. Skipping.")
             continue

        num_wavelengths = len(wavelength_data)

        # Data structure assumptions:
        # wavelength_data = [ wl1_data, wl2_data, ... ]
        # wl1_data = [reflectance_value(s)] or [reflectance_value(s), uncertainty_value(s)]
        # Values can be single numbers or lists (for multiple measurements?) - Assuming single values for now.

        current_wl_reflectances = []
        current_wl_uncertainties = []

        # This part needs clarification based on the *exact* JSON structure.
        # Let's assume a simplified structure for now:
        # { "reflectance_sample1": [ [wl1_refl, wl1_unc], [wl2_refl, wl2_unc], ... ],
        #   "reflectance_sample2": [ [wl1_refl], [wl2_refl], ... ] } -> needs rel_unc

        for i, wl_entry in enumerate(wavelength_data):
             if isinstance(wl_entry, (list, tuple)) and len(wl_entry) >= 1:
                 refl = wl_entry[0]
                 if len(wl_entry) == 2:
                     unc = wl_entry[1]
                 else: # Only reflectance provided, calculate uncertainty
                     unc = refl * relative_uncertainty
                 current_wl_reflectances.append(float(refl))
                 current_wl_uncertainties.append(float(unc))
             else:
                 # Handle unexpected format for this wavelength
                 logger.warning(f"Unexpected data format for wavelength {i+1} in sample '{key}': {wl_entry}. Skipping.")
                 # Need to decide how to handle gaps: fill with NaN? Abort? For now, skip.
                 # To maintain structure, might need to append NaN if skipping
                 # current_wl_reflectances.append(np.nan)
                 # current_wl_uncertainties.append(np.nan)
                 # Or raise error:
                 raise ValueError(f"Invalid data format for wavelength {i+1} in sample '{key}'. Expected [refl] or [refl, unc].")

        # Store as dict expected by downstream code
        # This assumes downstream expects separate dicts per wavelength *within* a sample?
        # The original code structure seemed to group by wavelength across samples later.
        # Let's adapt to the structure needed by compile_results: list of samples, each sample is list of wavelength dicts.
        sample_wl_dicts = []
        for i in range(len(current_wl_reflectances)):
             sample_wl_dicts.append({
                 'reflectances': np.array([current_wl_reflectances[i]], dtype=H5_FLOAT), # Array of 1 element
                 'incertitudes': np.array([current_wl_uncertainties[i]], dtype=H5_FLOAT) # Array of 1 element
             })
        reflectances_list.append(sample_wl_dicts)

    # The original return structure was List[List[Dict]], matching the remote sensing case.
    return reflectances_list, keys # keys are 'reflectance_sample1', etc.

def get_reflectances_remote_sensing(data_rho: gdal.Dataset, data_drho: gdal.Dataset) -> List[List[Dict[str, np.ndarray]]]:
    """Parses remote sensing reflectance data from GDAL datasets."""
    if not gdal: raise RuntimeError("GDAL is required for remote sensing data.")
    try:
        nb_geometries = data_rho.RasterXSize    # Samples per scene/line
        nb_scenes = data_rho.RasterYSize        # Lines/scenes
        nb_wavelengths = data_rho.RasterCount   # Bands

        # Validate dimensions match between rho and drho
        if (data_drho.RasterXSize != nb_geometries or
            data_drho.RasterYSize != nb_scenes or
            data_drho.RasterCount != nb_wavelengths):
            raise ValueError("Reflectance (rho) and uncertainty (drho) datasets have different dimensions.")

        # Read data: Returns (bands, lines, samples) -> (nb_wavelengths, nb_scenes, nb_geometries)
        reflectances_raster = data_rho.ReadAsArray()
        incertitude_raster = data_drho.ReadAsArray()

        # Reshape/Organize into the target structure: List[scenes], each scene is List[wavelengths], each wavelength is Dict
        reflectances_list = []
        for scene_idx in range(nb_scenes):
            scene_observations = []
            for wl_idx in range(nb_wavelengths):
                 # Extract data for this scene and wavelength
                 # Array shape: (nb_geometries,)
                 refl_data = reflectances_raster[wl_idx, scene_idx, :].astype(H5_FLOAT)
                 unc_data = incertitude_raster[wl_idx, scene_idx, :].astype(H5_FLOAT)
                 scene_observations.append({
                     'reflectances': refl_data,
                     'incertitudes': unc_data
                 })
            reflectances_list.append(scene_observations)

        return reflectances_list

    except AttributeError as e:
        logger.error(f"Error accessing GDAL dataset properties or reading data: {e}")
        raise RuntimeError("Failed to read remote sensing data using GDAL.") from e
    except ValueError as e:
        logger.error(f"Data inconsistency: {e}")
        raise


def _load_observations(observations_file_path: str, relative_uncertainty: float) -> Tuple[np.ndarray, np.ndarray]:
    """Loads observations from JSON, ENVI (via GDAL), or NPZ files.

    Returns:
        Tuple containing:
        - observations: np.ndarray (structured like List[samples/scenes], List[wavelengths], Dict['reflectances'|'incertitudes'])
        - wavelengths: List of wavelength strings or None.
        - keys: List of sample keys (e.g., 'reflectance_sample1') or None for ENVI/NPZ.
    """
    logger.info(f"Loading observations from: {observations_file_path}")
    relative_uncertainty = 0.1  # TODO: Make this configurable (e.g., in prediction_module config?)

    wavelengths = None
    keys = None
    observations_list = None

    if not os.path.exists(observations_file_path):
        raise FileNotFoundError(f"Observations file/directory not found: {observations_file_path}")

    try:
        if observations_file_path.lower().endswith('.json'):
            with open(observations_file_path, 'r') as f:
                observations_data = json.load(f)
            # Assume JSON structure: { "some_title": { "wavelengths": [...], "reflectance_sample1": [...], ... } }
            if not isinstance(observations_data, dict) or len(observations_data) != 1:
                 raise ValueError("JSON file should have a single root key containing the observation data.")
            observations_title = list(observations_data.keys())[0]
            observations_content = observations_data[observations_title]

            if "wavelengths" not in observations_content:
                 raise ValueError("JSON data must contain a 'wavelengths' list.")
            wavelengths = [str(wl) for wl in observations_content['wavelengths']] # Ensure strings

            observations_list, keys = get_reflectances_laboratory(observations_content, relative_uncertainty)
            if not observations_list:
                 raise ValueError("No valid reflectance data found in JSON file.")
            num_wl_json = len(observations_list[0]) # Wavelengths per sample
            if num_wl_json != len(wavelengths):
                 raise ValueError(f"Number of wavelengths in 'wavelengths' list ({len(wavelengths)}) does not match data found ({num_wl_json}).")


        elif os.path.isdir(observations_file_path): # Assuming ENVI directory structure (requires GDAL)
            if not gdal: raise RuntimeError("GDAL is required to load observations from a directory (ENVI format).")
            # Infer base name (assuming dir name matches base name of files inside)
            base_name = os.path.basename(observations_file_path)
            rho_path = os.path.join(observations_file_path, f'{base_name}_rho_mod')
            drho_path = os.path.join(observations_file_path, f'{base_name}_drho')

            if not os.path.isfile(rho_path) or not os.path.isfile(drho_path):
                 # Try extracting from zip if dir name ends with .zip (legacy?)
                 if observations_file_path.lower().endswith('.zip'):
                      logger.info("Input is a directory ending in .zip, trying to extract first.")
                      try:
                           with ZipFile(observations_file_path, 'r') as zip_ref:
                               zip_ref.extractall(os.path.dirname(observations_file_path)) # Extract to parent dir
                           # Update paths to potentially extracted files (assuming standard naming)
                           extracted_base = os.path.splitext(base_name)[0]
                           rho_path = os.path.join(os.path.dirname(observations_file_path), f'{extracted_base}_rho_mod')
                           drho_path = os.path.join(os.path.dirname(observations_file_path), f'{extracted_base}_drho')
                      except Exception as e:
                           raise IOError(f"Failed to extract zip file '{observations_file_path}': {e}")

            if not os.path.isfile(rho_path): raise FileNotFoundError(f"Reflectance file not found: {rho_path}")
            if not os.path.isfile(drho_path): raise FileNotFoundError(f"Uncertainty file not found: {drho_path}")

            data_rho = gdal.Open(rho_path, gdal.GA_ReadOnly)
            data_drho = gdal.Open(drho_path, gdal.GA_ReadOnly)
            if data_rho is None or data_drho is None:
                raise IOError("Could not open GDAL datasets for reflectance or uncertainty.")

            wavelengths = get_wavelengths_remote_sensing(data_rho) # Extracts from metadata
            observations_list = get_reflectances_remote_sensing(data_rho, data_drho)
            keys = None # No specific sample keys for ENVI format

            # GDAL datasets should be closed implicitly when they go out of scope,
            # but explicit closing/dereferencing is safer.
            del data_rho
            del data_drho


        elif observations_file_path.lower().endswith('.npz'):
            # Expects npz file with 'Y' (D, N_obs) and 'Y_u' (D, N_obs) arrays
            try:
                npzfile = np.load(observations_file_path)
                if 'Y' not in npzfile or 'Y_u' not in npzfile:
                    raise ValueError("NPZ file must contain 'Y' and 'Y_u' arrays.")
                y_obs = npzfile['Y'].astype(H5_FLOAT)
                y_unc = npzfile['Y_u'].astype(H5_FLOAT)
                if y_obs.shape != y_unc.shape or y_obs.ndim != 2:
                    raise ValueError("'Y' and 'Y_u' arrays must be 2D and have the same shape (D, N_obs).")

            except Exception as e:
                logger.error(f"Error loading data from NPZ file {observations_file_path}: {e}")
                raise IOError("Could not read data from NPZ file.") from e

        else:
            raise ValueError(f"Unsupported observations file type: {observations_file_path}. Must be .json, .npz, or a directory (for ENVI).")

        # Convert the list structure to a numpy array of objects for potentially easier handling?
        # Or keep as list of lists? Let's keep the list structure as compile_results expects it.
        # observations_array = np.array(observations_list, dtype=object)
        logger.info(f"Observations loaded successfully: Y({y_obs.shape}).")
        return y_obs, y_unc

    except (FileNotFoundError, IOError, ValueError, json.JSONDecodeError, RuntimeError) as e:
        logger.error(f"Failed to load observations: {e}")
        raise


def _importance_sampling(h5f: h5py.File, physical_model: Any, predictions: Any, Y: np.ndarray, Y_u: np.ndarray) -> List[Any]:
    """Performs Importance Sampling based on configuration and prediction results."""
    if not IMPORTANCE_SAMPLING.exist(h5f):
        logger.info("Importance sampling configuration not found. Skipping IS step.")
        return {} # Return empty dict if IS is not configured

    logger.info("Starting Importance Sampling...")
    try:
        N_zero, B, J = IMPORTANCE_SAMPLING.option_values(h5f, IS_OPTIONS) # N_zero, B, J
        on_means = IMPORTANCE_SAMPLING.get_value(h5f, "is_on_fullGMM")
        on_merged, on_centers = IMPORTANCE_SAMPLING.option_values(h5f, IS_ON_X_OPTIONS)
        nb_centers = predictions.mergedGMM.weights.shape[1] # K_merged
        D_model = physical_model.getDimensionY()
        covariance = np.ones(D_model) * 0.001

        is_results = {}

        if on_means == "yes":
            logger.info(f"Sampling fullGMM...")
            is_results["is_on_fullGMM"] = physical_model.importanceSampling(predictions.fullGMM, Y, Y_u, N_zero,
                                                                            B, J, covariance)
        if on_merged == "yes":
            if nb_centers == 0:
                logger.warning("IS requested on mergedGMM but K_merged=0. Skipping IS.")
            else:
                logger.info(f"Sampling mergedGMM...")
                is_results["is_on_mergedGMM"] = physical_model.importanceSampling(predictions.mergedGMM, Y, Y_u,
                                                                                  N_zero, B, J, covariance)
        if on_centers == "yes":
            if nb_centers == 0:
                logger.warning("IS requested on each merged GMM (centers) but K_merged=0. Skipping IS.")
            else:
                is_centers_res = []
                for id_center in range(nb_centers):
                    logger.info(f"Sampling center {id_center}...")
                    is_centers_res.append(
                        physical_model.importanceSampling(predictions.mergedGMM, Y, Y_u, N_zero, B, J,
                                                          np.ones(D_model) * 0.001,
                                                          idx_gaussian=id_center,
                                                          verbose = 1)
                    )
                is_results["is_on_centers"] = is_centers_res

        logger.info("Importance Sampling finished successfully.")
        return is_results

    except (KeyError, AttributeError, TypeError, ValueError) as e:
        logger.error(f"Importance Sampling failed: {e}")
        logger.warning("Continuing prediction process without Importance Sampling results.")
        return {} # Return empty dict on failure


# --- Output Writing Functions ---

def _write_results_to_npz(h5f, predictions, is_predicitions, output_dir):
    """Writes results to a output_dir.npz file. Results are predictions by mean, by centers, best prediction for both predictions and is_predictions."""
    write_gllim_preds = OUTPUT.get_value(h5f, "gllim_predictions")

    if write_gllim_preds == "fullGMM" or write_gllim_preds == "both":
        a = predictions.fullGMM.mean
        b = predictions.fullGMM.variance
        c = predictions.fullGMM.weights
        d = predictions.fullGMM.means
        e = predictions.fullGMM.covs
        np.savez(output_dir + "_fullGMM", mean=a, variance=b, weights=c, means=d, covs=e)
    
    if write_gllim_preds == "mergedGMM" or write_gllim_preds == "both":
        a = predictions.mergedGMM.mean
        b = predictions.mergedGMM.variance
        c = predictions.mergedGMM.weights
        d = predictions.mergedGMM.means
        e = predictions.mergedGMM.covs
        np.savez(output_dir + "_mergedGMM", mean=a, variance=b, weights=c, means=d, covs=e)
    
    for key, is_res in is_predicitions.items():
        if key == "fullGMM" or key == "mergedGMM":
            a = is_res.predictions
            b = is_res.predictions_variance
            c = is_res.nb_effective_sample
            d = is_res.effective_sample_size
            e = is_res.qn
            np.savez(output_dir + "_" + key, predictions=a, pred_variance=b, nb_effective_sample=c,
                     effective_sample_size=d, qn=e)
        elif key == "centers":
            for i, is_center_res in is_res:
                name = f"_center{i}"
                a = is_center_res.predictions
                b = is_center_res.predictions_variance
                c = is_center_res.nb_effective_sample
                d = is_center_res.effective_sample_size
                e = is_center_res.qn
                np.savez(output_dir + name, predictions=a, pred_variance=b, nb_effective_sample=c,
                        effective_sample_size=d, qn=e)

    

def predict(h5f: h5py.File, observations_file_path: str, output_dir: str, output_format: str):
    """Performs the prediction workflow: load model, load obs, predict, IS, compile, write."""
    logger.info("--- Starting Prediction Workflow ---")
    try:
        k_merged, merging_th, rel_uncertainity = _prediction_config(h5f) # K_merged, merging_threshold
        gllim_instance = _load_gllim_model(h5f)

        Y, Y_u = _load_observations(observations_file_path, rel_uncertainity)

        logger.info(f"Running GLLiM inverseDensities for {Y.shape[1]} observations...")
        predictions = gllim_instance.inverseDensities(Y, Y_u, k_merged, merging_th, 0)
        logger.info("GLLiM prediction step finished.")

        # 6. Run Importance Sampling (Optional)
        physical_model = _physical_model(h5f)
        if physical_model:
            is_predicitions = _importance_sampling(h5f, physical_model, predictions, Y, Y_u)
        else:
            is_predicitions = None

        _write_results_to_npz(h5f, predictions, is_predicitions, output_dir)

        logger.info("--- Prediction Workflow Finished Successfully ---")

    except (FileNotFoundError, ValueError, KeyError, RuntimeError, IOError, pickle.UnpicklingError, AttributeError, TypeError) as e:
        logger.critical(f"--- Prediction Workflow FAILED: {e} ---")
        # Potentially print traceback for debugging if needed
        # import traceback
        # traceback.print_exc()


# --- Main Execution ---

def main():
    short_descr = "xllim_cli.py - Command-line interface for xllim."
    epilog = "Run 'xllim_cli.py COMMAND --help' for more information on a command."
    parser = argparse.ArgumentParser(description=short_descr, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True) # Make command required

    # Common argument parsing logic
    def add_target_file_arg(p):
        p.add_argument("target_file", help="Path to the target HDF5 model file (e.g., model.h5)")
    def add_source_file_arg(p):
        p.add_argument("source_file", help="Path to the source file (format depends on command)")

    # Print command
    print_parser = subparsers.add_parser("print", help="Print contents of the HDF5 file")
    add_target_file_arg(print_parser)
    print_parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed configuration output")

    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Interactively edit HDF5 configuration")
    add_target_file_arg(edit_parser)

    # Copy command (renamed from 'cp' for clarity)
    copy_parser = subparsers.add_parser("copy", help="Interactively copy sections from one HDF5 file to another")
    add_source_file_arg(copy_parser)
    add_target_file_arg(copy_parser)

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate synthetic training data")
    add_target_file_arg(generate_parser)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the GLLiM model")
    add_target_file_arg(train_parser)

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions using a trained model")
    predict_parser.add_argument("model_file", help="Path to the HDF5 model file containing configuration and trained GLLiM")
    predict_parser.add_argument("observations_file", help="Path to observations file (JSON, NPZ, or ENVI dir)")
    predict_parser.add_argument("output", help="Output file name for results")
    predict_parser.add_argument(
        "-f", "--output-format", choices=["npz","json", "envi"], default="npz",
        help="Output format (default: npz)")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import data into the HDF5 file")
    import_parser.add_argument("import_type", choices=["geometries", "train-data"], help="Type of data to import")
    add_source_file_arg(import_parser) # Source format depends on import_type
    add_target_file_arg(import_parser)

    # Parse arguments
    args = parser.parse_args()

    # Determine HDF5 file path based on command
    h5_file_path = None
    if hasattr(args, 'target_file'):
        h5_file_path = args.target_file
    elif hasattr(args, 'model_file'): # For predict command
        h5_file_path = args.model_file

    # Check source file existence if applicable
    if hasattr(args, 'source_file'):
        if not os.path.exists(args.source_file):
            parser.error(f"Source file/directory not found: {args.source_file}")
            # No return here, parser.error exits

    # --- Execute Command ---

    # Read-only commands
    if args.command == "print":
        if not os.path.isfile(h5_file_path):
             print(f"Error: Target HDF5 file not found: {h5_file_path}")
             return # Exit gracefully
        try:
             with h5py.File(h5_file_path, 'r') as h5f:
                 print_h5(h5f, args.verbose)
        except Exception as e:
             print(f"Error opening or reading HDF5 file {h5_file_path}: {e}")
        return # Command finished

    # Commands requiring write access ('a' mode: read/write/create)
    if args.command in ["edit", "copy", "generate", "train", "predict", "import"]:

        # Open HDF5 file in append mode (creates if not exists)
        if h5_file_path: # Ensure path is defined (should be unless parser logic error)
             if not os.path.exists(h5_file_path):
                  logger.info(f"Creating new HDF5 file: {h5_file_path}")
             try:
                  with h5py.File(h5_file_path, 'a') as h5f:
                       # Dispatch to the correct function
                       if args.command == "edit":
                           edit_config(h5f)
                       elif args.command == "copy":
                           copy_h5_section_dialog(args.source_file, h5f)
                       elif args.command == "generate":
                           generate_data(h5f)
                       elif args.command == "train":
                           train_model(h5f)
                       elif args.command == "predict":
                           # Pass the open file handle to predict
                           predict(h5f, args.observations_file, args.output, args.output_format)
                       elif args.command == "import":
                            import_data(args.import_type, args.source_file, h5f)
             except Exception as e:
                  # Catch broad exceptions during file operations or command execution
                  print(f"An unexpected error occurred during command '{args.command}': {e}")
                  # Optionally log traceback here for debugging
        else:
             print("Error: HDF5 file path could not be determined for the command.")


# Init config functions -----------
FUNCTIONAL.config_function = configure_functional
GENERATOR.config_function = configure_generator
GLLIM_MODEL.config_function = configure_gllim_model
PREDICTION.config_function = configure_prediction_module
IMPORTANCE_SAMPLING.config_function = configure_importance_sampling
OUTPUT.config_function = configure_output

if __name__ == "__main__":
    main()
