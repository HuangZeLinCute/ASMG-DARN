#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7

    Attributes
    ----------
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):
        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.SESSION = 'SR'
        self._C.MODEL.INPUT = 'input'
        self._C.MODEL.TARGET = 'target'

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 1
        self._C.OPTIM.SEED = 3407
        self._C.OPTIM.NUM_EPOCHS = 300
        self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.000001
        self._C.OPTIM.BETA1 = 0.5
        self._C.OPTIM.WANDB = False

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = ""
        self._C.TRAINING.TRAIN_DIR = 'images_dir/train'
        self._C.TRAINING.VAL_DIR = 'images_dir/val'
        self._C.TRAINING.SAVE_DIR = 'checkpoints'
        self._C.TRAINING.PS_W = 512
        self._C.TRAINING.PS_H = 512
        self._C.TRAINING.ORI = False
        self._C.TRAINING.NUM_WORKERS = 4
        self._C.TRAINING.VAL_BATCH_SIZE = 4
        self._C.TRAINING.VAL_NUM_WORKERS = 4
        self._C.TRAINING.PATIENCE = 20

        # Loss function configuration
        self._C.LOSS = CN()
        self._C.LOSS.TYPE = 'STRONG_EDGE'  # Loss configuration type: DEFAULT, STRONG_EDGE, SMOOTH_PRIORITY, LIGHTWEIGHT
        
        # Various loss weight configurations
        self._C.LOSS.CONFIGS = CN()
        
        # DEFAULT configuration
        self._C.LOSS.CONFIGS.DEFAULT = CN()
        self._C.LOSS.CONFIGS.DEFAULT.MSE_WEIGHT = 1.0
        self._C.LOSS.CONFIGS.DEFAULT.SSIM_WEIGHT = 0.2
        
        # CUSTOM configuration
        self._C.LOSS.CONFIGS.CUSTOM = CN()
        self._C.LOSS.CONFIGS.CUSTOM.MSE_WEIGHT = 1.0
        self._C.LOSS.CONFIGS.CUSTOM.SSIM_WEIGHT = 0.3
        
        self._C.TESTING = CN()
        self._C.TESTING.WEIGHT = None
        self._C.TESTING.SAVE_IMAGES = True
        self._C.TESTING.BATCH_SIZE = 4
        self._C.TESTING.NUM_WORKERS = 4


        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def get_loss_config(self):
        """
        Get the currently selected loss configuration
        
        Returns:
            dict: dictionary containing loss weights
        """
        loss_type = self._C.LOSS.TYPE
        if hasattr(self._C.LOSS.CONFIGS, loss_type):
            config = getattr(self._C.LOSS.CONFIGS, loss_type)
            loss_config = {
                'mse_weight': getattr(config, 'MSE_WEIGHT', 1.0),
                'ssim_weight': config.SSIM_WEIGHT
            }
                
            return loss_config
        else:
            # If the specified configuration does not exist, return default configuration
            print(f"Warning: Loss configuration '{loss_type}' does not exist, using default configuration")
            config = self._C.LOSS.CONFIGS.DEFAULT
            return {
                'mse_weight': getattr(config, 'MSE_WEIGHT', 1.0),
                'ssim_weight': config.SSIM_WEIGHT
            }

    def __repr__(self):
        return self._C.__repr__()
