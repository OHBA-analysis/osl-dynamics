"""
Functions for batch training

In some use cases, e.g. compare and evaluate different models or hyperparameters,
we need to train many models and submit them to cluster using batch
This module contains useful functions to initialise proper batch training
See ./config_train_prototype.yaml for an example batch file.
"""
import os
import random
import time

import yaml
import numpy as np
import pandas as pd
from .pipeline import run_pipeline_from_file
class IndexParser:

    """
    Parse the training config file with index for batch training.
    Typically, a root config YAML file looks like the following, where
    batch_variable contains list of variables for different configurations
    non_batch_variable contains all other hyperparameters for training
    Given a training index, we need to find the specific batch_variable
    and combine that with other non_batch variables
    header:
      # We assign a time stamp for each batch training
      time: 2024-02-02T16:05:00.000Z
      # Where to save the model training results
      save_dir: './results_yaml_test/'
      # where to read the data
      data_dir: './data/node_timeseries/simulation_202402/sigma_0.1/'
      # where to load the spatial map and spatial surface map
      spatial_map:
      # Add custom notes, which will also be saved
      note: "test whether your yaml file works"

    # The following variables have lists for batch training
    batch_variable:
      models:
        - 'HMM'
        - 'Dynemo'
        - 'mDyenmo'
        - 'SWC'
      channels: [15,25,50,100]
      states: [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
      # Mode can be: train, repeat, split, cross_validation
      mode:
        - train
        - repeat_1
        - repeat_2
        - repeat_3
        - repeat_4
        - repeat_5
        - split_1
        - split_2
        - split_3
        - split_4
        - split_5
        - cv_1
        - cv_2
        - cv_3
        - cv_4
        - cv_5

    non_batch_variable:
      learn_means: false
      learn_covariances: true
      learn_trans_prob: true
      z_score_data: false
      learning_rate: 0.01
      split_strategy: random



    """
    def __init__(self,config:dict):

        # Sleep for random seconds, otherwise the batch job might contradicts
        time.sleep(random.uniform(0.,2.))

        self.save_dir = config['save_dir']
        self.batch_variable = config['batch_variable']
        self.non_batch_variable = config['non_batch_variable']

        # Check whether the save_dir exists, make directory if not
        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        # Check whether the root configuration file exists, save if not
        if not os.path.exists(f'{self.save_dir}config_root.yaml'):
            with open(f'{self.save_dir}config_root.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

        # Check if the config list file exists, create if not
        if not os.path.exists(f'{self.save_dir}config_list.csv'):
            self._make_list()
    def parse(self,index:int=0):
        """
        Given the index, parse the correct configuration file
        Parameters
        ----------
        index: the index passed in from batch.

        Returns
        -------
        config: dict
          the configuration file given the index
        """
        # Read in the list
        config_list = pd.read_csv(f'{self.save_dir}config_list.csv', index_col=0)

        # sv represents batch_variable given specific row
        bv = config_list.iloc[index].to_dict()

        # concatenate three parts of the dictionary
        new_config = {}
        new_config['save_dir'] = self.save_dir
        new_config.update(bv)
        new_config.update(self.non_batch_variable)

        return new_config


    def _make_list(self):
        """
        Make the list of batch variables with respect to index,
        and save them to f'{self.header["save_dir"]}config_list.xlsx'
        Returns
        -------
        """
        from itertools import product
        combinations = list(product(*self.batch_variable.values()))
        # Create a DataFrame
        df = pd.DataFrame(combinations, columns=self.batch_variable.keys())
        df.to_csv(f'{self.save_dir}config_list.csv', index=True)

class BatchTrain:
    """
    Convert a batch training configuration file to another config
    for training pipeline
    """
    train_keys_default = ['n_channels',
                          'n_states',
                          'sequence_length',
                          'learn_means',
                          'learn_covariances',
                          'learn_trans_prob',
                          'learning_rate',
                          'n_epochs'
                          ]
    def __init__(self,train_keys=None):
        self.train_keys = self.train_keys_default if train_keys is None else train_keys

    def model_train(self, config:dict):
        '''
        Batch model train method
        Parameters
        ----------
        config: dict
            the original configuration from the batch.

        Returns
        -------

        '''
        # data_dir need to be specified in the config
        if 'inputs' not in config:
            raise ValueError('No data directory specified!')
        # if prepare is not in the config, add 'prepare':{} to config
        if 'prepare' not in config:
            config['prepare'] = {}
        train_config = {'load_data':self.copy_key_value(['inputs','prepare'],config)}
        config_kwargs = self.copy_key_value(self.train_keys,config)
        init_kwargs = self.copy_key_value(["init_kwargs"],config)

        train_config[f'train_{config["model"]}'] = {'config_kwargs':config_kwargs,
                                                    'init_kwargs':init_kwargs}

        save_dir = f'{config["save_dir"]}{config["model"]}_ICA' \
                   f'_{config["n_channels"]}_state_{config["n_states"]}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(f'{save_dir}general_config.yaml', 'w') as file:
            yaml.dump(config, file)



        save_dir = f'{save_dir}{config["mode"]}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(f'{save_dir}train_config.yaml', 'w') as file:
            yaml.dump(train_config, file)
        if "split" in config["mode"]:
            pass
        elif "cv" in config["mode"]:
            pass
        else:
            run_pipeline_from_file(f'{save_dir}train_config.yaml',save_dir)







    def copy_key_value(self,keys:list,source:dict,dest:dict=None):
        """
        Copy the (key,value) pair from source to dest. dest can be none at the beginning.
        Parameters
        ----------
        source: dict
            source dictionary
        dest: dict
            destination dictionary
        keys: list
            keys to copy
        Returns
        -------
        dest: dict
            updated configuration
        """
        if dest is None:
            dest = {}
        for key in keys:
            dest[key] = source[key]
        return dest
