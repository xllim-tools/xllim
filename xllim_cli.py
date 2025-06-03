#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Command line script for xllim.
#
# Copyright (C) 2025 Inria

import argparse
import ast
import json
import h5py
from zipfile import ZipFile
import logging
import os
# import fnmatch
import numpy as np
from pathlib import Path
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
    print(f"\nWARNING: Import error: {e}.\nxllim library not found. Commands generate, train, and predict will not work.\n")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("xllim_cli")


# --- HDF5 Constants ---
H5_STRING = h5py.string_dtype(encoding='utf-8')
H5_INT = np.int32
H5_FLOAT = np.float64
H5_BOOL = bool

class H5Group():
    def __init__(self, name: str, h5_path: str, conf_function=None, datasets=[]):
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
    
    def get_data_set(self, h5f: h5py.File, ds_name: str) -> Union[np.ndarray, None]:
        dataset_path = self.h5_path + "/" + ds_name
        if dataset_path not in h5f:
            return None
        ds = h5f[dataset_path][()] # type: ignore
        if ds.dtype != H5_FLOAT: # type: ignore
            raise TypeError(f"Dataset '{dataset_path}' has unexpected dtype: {ds.dtype}. Expected {H5_FLOAT}.") # type: ignore
        return ds # type: ignore
    
    def set_data_set(self, h5f: h5py.File, name: str, data, dtype = H5_FLOAT):
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not in the list of expected datasets: {self.datasets}")
        dataset_path = self.h5_path + "/" + name
        if dataset_path in h5f:
            logger.debug(f"Replacing existing HDF5 dataset: {dataset_path}")
            del h5f[dataset_path]
        
        logger.debug(f"Writing HDF5 dataset: {dataset_path} with shape {getattr(data, 'shape', 'N/A')}")
        h5f.create_dataset(dataset_path, data=data, dtype=dtype)
        return
    
    def print_datasets(self, h5f: h5py.File, verbose=False):
        if not self.datasets:
            return
        if self.h5_path + "/" + self.datasets[0] not in h5f and not verbose:
            return
        print(f"  {self.name} datasets:")
        for ds in self.datasets:
            path = self.h5_path + "/" + ds
            if path in h5f:
                data = h5f[path][()] # type: ignore
                if verbose:
                    print(f"\t{ds}: {data}")
                else:
                    print(f"\t{ds}: {data.shape}") # type: ignore
            else:
                if verbose:
                    print(f"\t{ds}: (NO)")
        print("")
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
    
    def print_attributes(self, h5f: h5py.File, verbose: bool):
        """Prints attributes of a configured HDF5 group."""
        if not self.exist(h5f):
            if verbose:
                print(f"  {self.name}: Not configured")
            return 

        attrs = h5f[self.h5_path].attrs
        if len(attrs) > 0:
            if verbose:
                print(f"  {self.name} configuration:")
                for key, value in attrs.items():
                    print(f"\t{key} : {value}")
                print("")
            else:
                print(f"  {self.name} config: YES")
                if self.name == "functional":
                    model = FUNCTIONAL.get_value(h5f, "model")
                    print(f"\t\t({model})")
        else:
            if verbose and not self.datasets:
                print("\t(No attributes)")
    

# --- HDF5 datasets --------
GEOMETRIES = H5Group("geometries", "/functional_model/geometries", datasets=["sza", "vza", "phi"])
TRAIN_DATA = H5Group("train_data", "/train_data", datasets=["X", "Y"])
TRAINED_MODEL = H5Group("trained_gllim", "/gllim/trained_gllim", datasets=["Pi", "A", "B", "C", "Gamma", "Sigma"])

# --- HDF5 configurable sections --------
FUNCTIONAL = H5Group("functional", "/functional_model/config")
GENERATOR = H5Group("generator", "/functional_model/data_generator")
GLLIM_MODEL = H5Group("gllim_config", "/gllim/config")
PREDICTION = H5Group("prediction_module", "/prediction_module_config")
IMPORTANCE_SAMPLING = H5Group("importance_sampling", "/importance_sampling_config")
OUTPUT = H5Group("output", "/output_config")

H5_CONFIGURABLE_GROUPS = [FUNCTIONAL, GENERATOR, GLLIM_MODEL, IMPORTANCE_SAMPLING, PREDICTION, OUTPUT]
H5_ALL_ORDERED = [FUNCTIONAL, GEOMETRIES, GENERATOR, TRAIN_DATA,
                  GLLIM_MODEL, TRAINED_MODEL,
                  IMPORTANCE_SAMPLING, PREDICTION,
                  OUTPUT]

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
                      ('relative_uncertainty', 'Relative uncertainty. Used if no Y_u in observations (Optional, 0.001)', None, H5_FLOAT))
IS_OPTIONS = (('N_zero', 'Initial samples (IMIS, default: N/10)', None, H5_INT),
              ('B', 'New samples per step (IMIS, default: N/20)', None, H5_INT),
              ('J', 'Number of iterations (IMIS, default: 18)', None, H5_INT),
              ('covariance', 'Optional covariance', None, H5_FLOAT),
              ('seed', 'Seed for random generator (integer)', None, H5_INT))
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

    train_variant = GLLIM_MODEL.get_value(h5f, 'train_variant')
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
    if K_merged > 0: # type: ignore
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
    if not TRAIN_DATA.exist(h5f):
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


def _load_gllim_model(h5f: h5py.File) -> Any:
    """Loads serialized GLLiM parameters from HDF5 and sets them."""
    try:
        gllim_Pi = TRAINED_MODEL.get_data_set(h5f, "Pi")
    except:
        logger.error(f"Serialized GLLiM model not found.")
        return train_model(h5f)

    try:
        D = TRAIN_DATA.get_value(h5f, "D")
        L = TRAIN_DATA.get_value(h5f, "L")
        K, gamma_type, sigma_type, n_hidden = GLLIM_MODEL.option_values(h5f, GLLIM_OPTIONS)
        logger.info(f"Instantiating GLLiM(K={K}, D={D}, L={L}, gamma={gamma_type}, sigma={sigma_type}, n_hidden={n_hidden})")
        gllim_instance = xllim.GLLiM(K, D, L, gamma_type, sigma_type, n_hidden)

        gllim_instance.setParamPi(TRAINED_MODEL.get_data_set(h5f, "Pi"))
        gllim_instance.setParamA(TRAINED_MODEL.get_data_set(h5f, "A"))
        gllim_instance.setParamB(TRAINED_MODEL.get_data_set(h5f, "B"))
        gllim_instance.setParamC(TRAINED_MODEL.get_data_set(h5f, "C"))
        gllim_instance.setParamGamma(TRAINED_MODEL.get_data_set(h5f, "Gamma"))
        gllim_instance.setParamSigma(TRAINED_MODEL.get_data_set(h5f, "Sigma"))
        logger.info(f"Successfully loaded and set GLLiM parameters")
        return gllim_instance
    except (TypeError, AttributeError, KeyError) as e:
        logger.error(f"Failed to load or set GLLiM parameters: {e}")
        raise RuntimeError("Could not load trained GLLiM model.") from e

# --- Core Command Functions ---

def print_h5(h5f: h5py.File, verbose: bool):
    """Prints a summary or detailed view of the HDF5 file contents."""
    print(f"\n--- Contents of {h5f.filename} ---")
    
    for section in H5_ALL_ORDERED:
        section.print_attributes(h5f, verbose=verbose)
        if section.name == "geometries":
            section.print_datasets(h5f, verbose=verbose)
        else:
            section.print_datasets(h5f, verbose=False)

    groups = []
    for gr in H5_ALL_ORDERED:
        if gr.exist(h5f):
            groups.append(gr.name)

    if set(["train_data", "trained_gllim", "prediction_module", "output"]) <= set(groups):
        print(f"Ready to run predict") 
    

def edit_config(h5f: h5py.File):
    """Walks through configuration groups for editing."""
    print("\n--- Interactive Configuration Editor ---")
    # Define the mapping from group key to configure function

    for section in H5_CONFIGURABLE_GROUPS:
        print("-" * 20)
        prompt_action = "Edit" if section.exist(h5f) else "Add"

        if section.exist(h5f):
            section.print_attributes(h5f, verbose=True)

        while True:
            k = input(f"{prompt_action} {section.name} configuration? (y/n/q): ").lower().strip()
            if k == 'y':
                try:
                    section.config_function(h5f) # type: ignore
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
    """Imports geometries from a JSON or NPZ file into the HDF5 file."""

    logger.info(f"Importing geometries from {source_path} into {GEOMETRIES.h5_path}")

    GEOMETRIES.ensure_exist(dest_h5f)

    try:
        if source_path.endswith('.json'):
            with open(source_path, 'r') as json_file:
                geoms = json.load(json_file)
        elif source_path.endswith('.npz'):
            geoms = np.load(source_path)
        else:
            raise ValueError("Geometries source file must be a JSON file.")

        for k in GEOMETRIES.datasets:
            data = geoms[k]
            GEOMETRIES.set_data_set(dest_h5f, k, data)

        # Add source attribute
        GEOMETRIES.set_value(dest_h5f, 'source', os.path.basename(source_path), dtype=H5_STRING)
        dest_h5f.flush()
        logger.info(f"Geometries import successful.")

    except FileNotFoundError:
        logger.error(f"Source file not found: {source_path}")
        raise
    except (IOError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error importing geometries: {e}")
        raise


def _h5_copy(group: H5Group, src_h5f: h5py.File, dst_h5f: h5py.File):
    """Copies a group from a source HDF5 file to the destination."""
    if group.exist(dst_h5f):
        logger.warning(f"Replacing existing object '{group.name}' in destination file.")
        del dst_h5f[group.h5_path]

    logger.info(f"Copying '{group.name}' from {src_h5f.filename} to {dst_h5f.filename}")
    src_obj = src_h5f[group.h5_path]
    # Copy object (group or dataset)
    dst_h5f.copy(src_obj, group.h5_path)
    dst_h5f.flush()
    logger.debug(f"Copy successful for '{group.name}'.")


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


def copy_h5_section_dialog(src_h5f: h5py.File, dst_h5f: h5py.File):
    """Interactively copies sections from a source HDF5 file to the destination."""
    available_groups = []
    for group in H5_ALL_ORDERED:
        if group.exist(src_h5f):
            available_groups.append(group)

    if len(available_groups) == 0:
        print("Nothing to copy in the source file.")
        return

    print("Objects available to copy:")
    for idx, group in enumerate(available_groups):
        print(f"  {idx+1}. {group.name}")
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
            _h5_copy(available_groups[idx-1], src_h5f, dst_h5f)
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

        logger.info(f"Calling genData(N={N}, type='{generator_type}', cov={covariance_vector.shape}, seed={seed}) using {type(f_model).__name__}")
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
        raise # Re-raise to indicate failure


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
        TRAINED_MODEL.set_data_set(h5f, "Pi", gllim_params_obj.Pi)
        TRAINED_MODEL.set_data_set(h5f, "A", gllim_params_obj.A)
        TRAINED_MODEL.set_data_set(h5f, "B", gllim_params_obj.B)
        TRAINED_MODEL.set_data_set(h5f, "C", gllim_params_obj.C)
        TRAINED_MODEL.set_data_set(h5f, "Gamma", gllim_params_obj.Gamma)
        TRAINED_MODEL.set_data_set(h5f, "Sigma", gllim_params_obj.Sigma)
        h5f.flush()
        logger.info(f"Trained GLLiM model parameters saved")
        return gllim_instance

    except (FileNotFoundError, ValueError, KeyError, RuntimeError, IOError, AttributeError, TypeError) as e:
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


def _load_observations(observations_file_path: str, relative_uncertainty: Union[float, None]) -> list:
    """Loads observations from JSON, ENVI (via GDAL), or NPZ files.

    Returns:
        List of tupples containing (name, Y, Y_u, labels)
        "name": "" or name of the dataset
        "Y", "Y_u" : observations and observations uncertainities (D, N)
        "labels": list of labels (e.g. waveleghts) len(labels) is D
    """
    logger.info(f"Loading observations from: {observations_file_path}")

    if not os.path.exists(observations_file_path):
        raise FileNotFoundError(f"Observations file/directory not found: {observations_file_path}")

    observations = []
    try:
        if observations_file_path.lower().endswith('.json'):
            with open(observations_file_path, 'r') as f:
                observations_data = json.load(f)
            # Assume JSON structure: { "some_title": { "wavelengths": [...], "reflectance_sample1": [...], ... } }
            if not isinstance(observations_data, dict) or len(observations_data) != 1:
                 raise ValueError("JSON file should have a single root key containing the observation data.")
            observations_title = list(observations_data.keys())[0]
            content = observations_data[observations_title]
            
            if "wavelengths" in content:
                wavelengths = content["wavelengths"]
            else:
                wavelengths = []

            for k in content.keys():
                if k.startswith('reflectance'):
                    obs = np.array(content[k], dtype=np.float64)
                    if obs.shape[1] == 2:
                        Y = np.array(obs[:,0,:].T)
                        Y_u = np.array(obs[:,1,:].T)
                    else:
                        raise ValueError("TODO: can lab data not contain uncertainities?")
                    name = k[12:] if len(k) > 11 else k
                    observations.append((name, Y, Y_u, wavelengths))
                
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

            # GDAL datasets should be closed implicitly when they go out of scope,
            # but explicit closing/dereferencing is safer.
            del data_rho
            del data_drho


        elif observations_file_path.lower().endswith('.npz'):
            # Expects npz file with 'Y' (D, N_obs) and 'Y_u' (D, N_obs) arrays
            npzfile = np.load(observations_file_path)
            if 'Y' in npzfile:
                Y = npzfile['Y'].astype(H5_FLOAT)
            else:
                raise ValueError("NPZ file must contain 'Y'")
            
            if 'Y_u' in npzfile:
                Y_u = npzfile['Y_u'].astype(H5_FLOAT)
            else:
                if relative_uncertainty is None:
                    raise ValueError("NPZ file must contain 'Y_u' or relative_uncertainity must be set")
                else:
                    Y_u = np.ones(Y.shape) * relative_uncertainty
            if Y.shape != Y_u.shape or Y.ndim != 2:
                raise ValueError("'Y' and 'Y_u' arrays must be 2D and have the same shape (D, N_obs).")
            observations.append(("", Y, Y_u, []))

        else:
            raise ValueError(f"Unsupported observations file type: {observations_file_path}. Must be .json, .npz, or a directory (for ENVI).")

        logger.info(f"Observations loaded successfully: Y({Y.shape}).")
        return observations

    except (FileNotFoundError, IOError, ValueError, json.JSONDecodeError, RuntimeError) as e:
        logger.error(f"Failed to load observations: {e}")
        raise


def _importance_sampling(h5f: h5py.File, physical_model: Any, predictions: Any, Y: np.ndarray, Y_u: np.ndarray) -> Dict:
    """Performs Importance Sampling based on configuration and prediction results."""
    if not IMPORTANCE_SAMPLING.exist(h5f):
        logger.info("Importance sampling configuration not found. Skipping IS step.")
        return {} # Return empty dict if IS is not configured

    logger.info("Starting Importance Sampling...")
    try:
        nb_centers = predictions.mergedGMM.weights.shape[1] # K_merged
        D_model = physical_model.getDimensionY()
        N_zero, B, J, cov, seed = IMPORTANCE_SAMPLING.option_values(h5f, IS_OPTIONS)
        # handle optional configuration
        B = 0 if B is None else B
        J = 0 if J is None else J
        covariance = 0 if cov is None else np.ones(D_model) * cov

        is_results = {}

        if PREDICTION.get_value(h5f, "IS_on_fullGMM"):
            logger.info(f"Sampling fullGMM...")
            is_results["is_on_fullGMM"] = physical_model.importanceSampling(predictions.fullGMM, Y, Y_u, N_zero,
                                                                            B, J, covariance, seed=seed)
        if PREDICTION.get_value(h5f, "IS_on_mergedGMM"):
            if nb_centers == 0:
                logger.warning("IS requested on mergedGMM but K_merged=0. Skipping IS.")
            else:
                logger.info(f"Sampling mergedGMM...")
                is_results["is_on_mergedGMM"] = physical_model.importanceSampling(predictions.mergedGMM, Y, Y_u,
                                                                                  N_zero, B, J, covariance, seed=seed)
        if PREDICTION.get_value(h5f, "IS_on_centers"):
            if nb_centers == 0:
                logger.warning("IS requested on each merged GMM (centers) but K_merged=0. Skipping IS.")
            else:
                is_centers_res = []
                for id_center in range(nb_centers):
                    logger.info(f"Sampling center {id_center}...")
                    is_centers_res.append(
                        physical_model.importanceSampling(predictions.mergedGMM, Y, Y_u, N_zero, B, J,
                                                          covariance,
                                                          idx_gaussian=id_center,
                                                          verbose = 1, seed=seed)
                    )
                is_results["is_on_centers"] = is_centers_res

        logger.info("Importance Sampling finished successfully.")
        return is_results

    except (KeyError, AttributeError, TypeError, ValueError) as e:
        logger.error(f"Importance Sampling failed: {e}")
        logger.warning("Continuing prediction process without Importance Sampling results.")
        return {} # Return empty dict on failure


# --- Output Writing Functions ---

def _write_results_to_npz(h5f, predictions, is_predicitions, name: str, output_dir: str):
    """Writes results to a output_dir.npz file. Results are predictions by mean, by centers, best prediction for both predictions and is_predictions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if len(name) > 0:
        name += "_"

    if OUTPUT.get_value(h5f, "gllim_fullGMM"):
        a = predictions.fullGMM.mean
        b = predictions.fullGMM.variance
        c = predictions.fullGMM.weights
        d = predictions.fullGMM.means
        e = predictions.fullGMM.covs
        out_file = os.path.join(output_dir, name + "fullGMM")
        logger.info(f"Writing results: {out_file}")
        np.savez(out_file, mean=a, variance=b, weights=c, means=d, covs=e)
    
    if OUTPUT.get_value(h5f, "gllim_mergedGMM"):
        a = predictions.mergedGMM.mean
        b = predictions.mergedGMM.variance
        c = predictions.mergedGMM.weights
        d = predictions.mergedGMM.means
        e = predictions.mergedGMM.covs
        if c.shape[1] > 0:
            out_file = os.path.join(output_dir, name + "mergedGMM")
            logger.info(f"Writing results: {out_file}")
            np.savez(out_file, mean=a, variance=b, weights=c, means=d, covs=e)
        else:
            logger.info("No data in gllim_mergedGMM results. Skipping")

    for variant in ("is_on_fullGMM", "is_on_mergedGMM"): 
        if variant in is_predicitions:
            is_res = is_predicitions[variant]
            a = is_res.predictions
            b = is_res.predictions_variance
            c = is_res.nb_effective_sample
            d = is_res.effective_sample_size
            e = is_res.qn
            out_file = os.path.join(output_dir, name + variant)
            logger.info(f"Writing results: {out_file}")
            np.savez(out_file, predictions=a, pred_variance=b, nb_effective_sample=c,
                        effective_sample_size=d, qn=e)
    
    if "is_on_centers" in is_predicitions:
        for i, is_center_res in enumerate(is_predicitions["is_on_centers"]):
            a = is_center_res.predictions
            b = is_center_res.predictions_variance
            c = is_center_res.nb_effective_sample
            d = is_center_res.effective_sample_size
            e = is_center_res.qn
            out_file = os.path.join(output_dir, name + f"center{i}")
            logger.info(f"Writing results: {out_file}")
            np.savez(out_file, predictions=a, pred_variance=b, nb_effective_sample=c,
                    effective_sample_size=d, qn=e)

    

def predict(h5f: h5py.File, observations_file_path: str, output_dir: str, output_format: str):
    """Performs the prediction workflow: load model, load obs, predict, IS, compile, write."""
    logger.info("--- Starting Prediction Workflow ---")
    try:
        k_merged, merging_th, rel_uncertainity = PREDICTION.option_values(h5f, PREDICTION_OPTIONS)
        gllim_instance = _load_gllim_model(h5f)

        observations = _load_observations(observations_file_path, rel_uncertainity)

        for item in observations:
            name, Y, Y_u, labels = item
            logger.info(f"Running GLLiM inverseDensities for {Y.shape[1]} observations...")
            predictions = gllim_instance.inverseDensities(Y, Y_u, k_merged, merging_th, 0)
            logger.info("GLLiM prediction step finished.")

            # 6. Run Importance Sampling (Optional)
            physical_model = _physical_model(h5f)
            if physical_model:
                is_predicitions = _importance_sampling(h5f, physical_model, predictions, Y, Y_u)
            else:
                is_predicitions = None

            if output_format == 'npz':
                _write_results_to_npz(h5f, predictions, is_predicitions, name, output_dir)

        logger.info("--- Prediction Workflow Finished Successfully ---")

    except (ValueError, KeyError, RuntimeError, IOError, AttributeError, TypeError) as e:
        logger.critical(f"--- Prediction Workflow FAILED: {e} ---")

# --- Main Execution ---

def main():
    short_descr = "xllim_cli.py - Command-line interface for xllim."
    epilog = "Run 'xllim_cli.py COMMAND --help' for more information on a command."
    parser = argparse.ArgumentParser(description=short_descr, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True) # Make command required

    # Common argument parsing logic
    def add_model_file_arg(p):
        p.add_argument("model_file", help="Path to the HDF5 model file (e.g., model.h5)")
    def add_source_file_arg(p):  # geometries, observations
        p.add_argument("source_file", help="Path to the source file (format depends on command)")

    # Print command
    print_parser = subparsers.add_parser("print", help="Print contents of the HDF5 file")
    add_model_file_arg(print_parser)
    print_parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed configuration output")

    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Interactively edit HDF5 configuration")
    add_model_file_arg(edit_parser)

    # Copy command (renamed from 'cp' for clarity)
    copy_parser = subparsers.add_parser("copy", help="Interactively copy sections from one HDF5 file to another")
    add_model_file_arg(copy_parser)
    copy_parser.add_argument("target_file", help="Path to the file sections will be copied to")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate synthetic training data")
    add_model_file_arg(generate_parser)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the GLLiM model")
    add_model_file_arg(train_parser)

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
    add_model_file_arg(import_parser)

    # Parse arguments
    args = parser.parse_args()

    # Check source file existence if applicable
    for arg in ('model_file', 'source_file'):
        if hasattr(args, arg):
            h5_file_path = vars(args)[arg]
            if not os.path.isfile(h5_file_path) and args.command != 'edit':
                print(f"Error: File not found: {h5_file_path}")
                return # Exit gracefully

    # --- Execute Command ---

    if args.command in ["print", "copy"]:
        mode = 'r'
    else:
        mode = 'a'

    with h5py.File(h5_file_path, mode) as h5f:
        # Dispatch to the correct function
        if args.command == "print":
            print_h5(h5f, args.verbose)
        elif args.command == "edit":
            edit_config(h5f)
        elif args.command == "copy":
            with h5py.File(args.target_file, 'a') as target_h5f:
                copy_h5_section_dialog(h5f, target_h5f)
        elif args.command == "generate":
            generate_data(h5f)
        elif args.command == "train":
            train_model(h5f)
        elif args.command == "predict":
            # Pass the open file handle to predict
            predict(h5f, args.observations_file, args.output, args.output_format)
        elif args.command == "import":
            import_data(args.import_type, args.source_file, h5f)
    return # end of main()


# Init config functions -----------
FUNCTIONAL.config_function = configure_functional
GENERATOR.config_function = configure_generator
GLLIM_MODEL.config_function = configure_gllim_model
PREDICTION.config_function = configure_prediction_module
IMPORTANCE_SAMPLING.config_function = configure_importance_sampling
OUTPUT.config_function = configure_output

if __name__ == "__main__":
    main()
