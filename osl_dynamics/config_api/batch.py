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
import json
from itertools import product

import yaml
import numpy as np
import pandas as pd
from .pipeline import run_pipeline_from_file
from ..data.base import Data
from ..utils.misc import override_dict_defaults
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
  # Add custom notes, which will also be saved
  note: "test whether your yaml file works"
# where to read the data
load_data:
  inputs: './data/node_timeseries/simulation_202402/sigma_0.1/'
  prepare:
    select:
      timepoints:
        - 0
        - 1200
    standardize: {}
# Where to save model training results
save_dir: './results_yaml_test/'
# where to load the spatial map and spatial surface map
spatial_map: './data/spatial_maps/'
non_batch_variable:
  n_channels: 25
  sequence_length: 600
  learn_means: false
  learn_covariances: true
  learn_trans_prob: true
  learning_rate: 0.01
  n_epochs: 30
  split_strategy: random
  init_kwargs:
    n_init: 10
    n_epochs: 2
# The following variables have lists for batch training
batch_variable:
  model:
    - 'hmm'
    - 'dynemo'
    - 'mdyenmo'
    - 'swc'
  n_states: [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
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
    """
    def __init__(self,config:dict):

        # Sleep for random seconds, otherwise the batch job might contradicts
        time.sleep(random.uniform(0.,2.))

        self.save_dir = config['save_dir']
        self.batch_variable = config['batch_variable']
        self.non_batch_variable = config['non_batch_variable']
        self.other_keys =  {key: value for key, value in config.items()
                                if key not in ['batch_variable','non_batch_variable']}

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
        new_config.update(self.other_keys)
        new_config.update(bv)
        new_config.update(self.non_batch_variable)

        new_config['save_dir'] = f'{new_config["save_dir"]}{new_config["model"]}' \
                             f'_ICA_{new_config["n_channels"]}_state_{new_config["n_states"]}/{new_config["mode"]}/'

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
    mode_key_default = 'mode'

    train_keys_default = ['n_channels',
                          'n_states',
                          'learn_means',
                          'learn_covariances',
                          'learn_trans_prob',
                          'initial_means',
                          'initial_covariances',
                          'initial_trans_prob',
                          'sequence_length',
                          'batch_size',
                          'learning_rate',
                          'n_epochs',
                          ]
    def __init__(self,config:dict,train_keys=None):
        self.train_keys = self.train_keys_default if train_keys is None else train_keys

        # Validate the configuration file
        if 'load_data' not in config:
            raise ValueError('No data directory specified!')
        # The default mode of 'mode' is train
        if 'mode' not in config:
            config['mode'] = 'train'
        if 'init_kwargs' not in config:
            config['init_kwargs'] = {}
        # Check whether save directory is specified
        if 'save_dir' not in config:
            raise ValueError('Saving directory not specified!')

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        if not os.path.isfile(f'{config["save_dir"]}batch_config.yaml'):
            with open(f'{config["save_dir"]}batch_config.yaml','w') as file:
                yaml.safe_dump(config,file, default_flow_style=False)
        self.config = config

    def model_train(self,cv_ratio=0.8):
        '''
        Batch model train method
        cv_ration: float,optional
           the proportion of sessions to use as the training data
        Returns
        -------
        '''
        prepare_config = {}
        prepare_config['load_data'] = self.config['load_data']

        prepare_config[f'train_{self.config["model"]}'] = {
            'config_kwargs':
                {key: self.config[key] for key in self.train_keys if key in self.config},
            'init_kwargs':
                self.config['init_kwargs']
        }

        if "split" in self.config["mode"]:
            # We need to know how many sessions in advance
            indice_1, indice_2 = self.select_indice()

            # Save the selected and remaining indices to JSON files
            with open(f'{self.config["save_dir"]}indices_1.json', 'w') as json_file:
                json.dump(indice_1, json_file)
            with open(f'{self.config["save_dir"]}indices_2.json', 'w') as json_file:
                json.dump(indice_2, json_file)

            for i in range(0,2):
                temp_save_dir = f'{self.config["save_dir"]}half_{i+1}/'
                if not os.path.exists(temp_save_dir):
                    os.makedirs(temp_save_dir)
                prepare_config['keep_list'] = f'{self.config["save_dir"]}indices_{i+1}.json'
                with open(f'{temp_save_dir}prepared_config.yaml', 'w') as file:
                    yaml.safe_dump(prepare_config, file, default_flow_style=False)
                run_pipeline_from_file(f'{temp_save_dir}prepared_config.yaml',
                                      temp_save_dir)


        elif "cv" in self.config["mode"]:
            indice_1, indice_2 = self.select_indice(ratio=cv_ratio)

            # Save the selected and remaining indices to JSON files
            with open(f'{self.config["save_dir"]}indices_train.json', 'w') as json_file:
                json.dump(indice_1, json_file)
            with open(f'{self.config["save_dir"]}indices_validate.json', 'w') as json_file:
                json.dump(indice_2, json_file)

            prepare_config['keep_list'] = f'{self.config["save_dir"]}indices_train.json'
            with open(f'{self.config["save_dir"]}prepared_config.yaml', 'w') as file:
                yaml.safe_dump(prepare_config, file, default_flow_style=False)
            run_pipeline_from_file(f'{self.config["save_dir"]}prepared_config.yaml',
                                   self.config['save_dir'])


        else:
            with open(f'{self.config["save_dir"]}prepared_config.yaml', 'w') as file:
                yaml.safe_dump(prepare_config, file, default_flow_style=False)
            run_pipeline_from_file(f'{self.config["save_dir"]}prepared_config.yaml',
                                   self.config["save_dir"])


    def select_indice(self,ratio=0.5):
        if "n_sessions" not in self.config:
            data = Data(self.config["load_data"]["inputs"])
            n_sessions = len(data.arrays)
        else:
            n_sessions = self.config["n_sessions"]

        all_indices = list(range(n_sessions))
        # Calculate the number of indices to select (half of the total)
        n_selected_sessions = int(n_sessions * ratio)

        # Randomly select indices without replacement
        selected_indices = random.sample(all_indices, n_selected_sessions)

        # Calculate the remaining indices
        remaining_indices = list(set(all_indices) - set(selected_indices))
        return selected_indices,remaining_indices

def batch_check(config:dict):
    '''
    Check whether the batch training is successful, raise value Error
    and save the list if some batch training is not successful.
    Parameters
    ----------
    config: str
        configuration file of batch training
    '''
    # check the bad directories
    bad_dirs = []

    # Check whether the fail training list exists, delete if so.
    bad_dirs_save_path = f'{config["save_dir"]}failure_list.yaml'

    # Check if the file exists, delete if so
    if os.path.exists(bad_dirs_save_path):
        os.remove(bad_dirs_save_path)

    for values in product(*config['batch_variable'].values()):
        combination = dict(zip(config['batch_variable'].keys(), values))
        vars = override_dict_defaults(config['non_batch_variable'],combination)
        check_dir = f'{config["save_dir"]}{vars["model"]}_ICA' \
                    f'_{vars["n_channels"]}_state_{vars["n_states"]}/{vars["mode"]}/'

        # Check whether batch_config exists
        if not os.path.isfile(f'{check_dir}batch_config.yaml'):
            bad_dirs.append(check_dir)
        # Check whether prepared_config exists
        try:
            if "split" in vars['mode']:
                check_dir = f'{check_dir}half_2/'
            assert os.path.isfile(f'{check_dir}prepared_config.yaml')
            # Check whether model training is successful
            assert os.path.exists(f'{check_dir}model')
            # Check if the covs.npy file exists
            assert os.path.isfile(f'{check_dir}inf_params/covs.npy')
            # Check if the means.npy file exists
            assert os.path.isfile(f'{check_dir}inf_params/means.npy')
            # Check whether the alp.pkl exists
            assert os.path.isfile(f'{check_dir}inf_params/alp.pkl')
        except AssertionError:
            bad_dirs.append(check_dir)

    if len(bad_dirs)>0:
        # Serialize and save the list to a file
        with open(bad_dirs_save_path, 'w') as file:
            yaml.safe_dump(bad_dirs, file,default_flow_style=False)
        raise ValueError(f'Some training cases failed, check {bad_dirs_save_path} for the list')
    else:
        print('All model training successful!')
