import os
import time
import random
import copy
import yaml
import pandas as pd
from typing import Union
from itertools import product

from ..evaluate.cross_validation import CrossValidationSplit
class IndexParser:
    """
    Parse the training configuration file for batch training.
    Typically, a root config YAML file looks like the following.

    header:
    # We assign a time stamp for each batch training
    time: 2024-11-18T17:00:00.000Z
    # Add custom notes, which will also be saved
    note: "Training configuration for HCP data (ICA=50), bi-cross-validation(bcv),repeat,split and naive cross-validation (ncv)"
    # where to read the data
    load_data:
      inputs: './data/node_timeseries/3T_HCP1200_MSMAll_d50_ts2/'
      prepare:
        select:
          timepoints:
            - 0
            - 4800
        standardize:
          session_length: 1200
      kwargs:
        load_memmaps: True
    # Where to save model training results
    save_dir: './results_final/real/ICA_50/'
    # where to load the spatial map and spatial surface map
    spatial_map: './data/spatial_maps/groupICA_3T_HCP1200_MSMAll_d50.ica'
    model:
      hmm:
        n_channels: 50
        sequence_length: 400
        batch_size: 256
        learn_means: false
        learn_covariances: true
        learn_trans_prob: true
        learning_rate: 0.001
        n_epochs: 30
        init_kwargs:
          n_init: 10
          n_epochs: 2
      dynemo:
        n_channels: 50
        sequence_length: 100
        inference_n_units: 64
        inference_normalization: layer
        model_n_units: 64
        model_normalization: layer
        learn_alpha_temperature: True
        initial_alpha_temperature: 1.0
        learn_means: False
        learn_covariances: True
        do_kl_annealing: True
        kl_annealing_curve: tanh
        kl_annealing_sharpness: 5
        n_kl_annealing_epochs: 15
        batch_size: 64
        learning_rate: 0.001
        n_epochs: 30
        init_kwargs:
          n_init: 10
          n_epochs: 2
      swc:
        n_channels: 50
        learn_means: False
        learn_covariances: True
        window_length: 100
        window_offset: 100
    n_states: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # Mode can be: bi-cross-validatioin(bcv),repeat,naive cross validation (ncv) and split
    mode:
      bcv:
        split_row:
          n_samples: 1003
          method: ShuffleSplit
          method_kwargs:
            n_splits: 100
            train_size: 0.8
        split_column:
          n_samples: 50
          method: ShuffleSplit
          method_kwargs:
            n_splits: 100
            train_size: 0.5
        strategy: pairing
      repeat:
        n_realizations: 3
      ncv:
        split_row:
          n_samples: 1003
          method: KFold
          method_kwargs:
            n_splits: 5
      split:
        split_row:
          n_samples: 1003
          method: ShuffleSplit
          method_kwargs:
            n_splits: 5
            train_size: 0.5
    """

    def __init__(self, config: Union[dict, str]):
        time.sleep(random.uniform(0., 2.)) # Prevent job conflicts

        # If config is a string, assume it's a file path and load YAML
        if isinstance(config, str):
            with open(config, 'r') as file:
                config = yaml.safe_load(file)

        self.config = config
        self.save_dir = config['save_dir']

        # Check whether the save_dir exists, make directory if not
        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])

        # Check whether the root configuration file exists, save if not
        if not os.path.exists(f'{self.save_dir}config_root.yaml'):
            with open(f'{self.save_dir}config_root.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

    def parse(self, index: int = 0):
        """
        Parse the configuration file and train the model given the index

        Parameters
        ----------
        index: int
            the index passed in from batch.
        """
        # Read in the list
        config_list = pd.read_csv(os.path.join(self.save_dir,'config_list.csv'), index_col=0)

        model, n_states, mode = config_list.iloc[index]

        # concatenate three parts of the dictionary
        new_config = copy.deepcopy(self.config)
        # Preserve the correct model used here
        value = new_config['model'].get(model)
        new_config['model'] = {model: value} if value is not None else {}
        if model!='dynemo':
            new_config['n_states'] = int(n_states)
        else:
            new_config['n_modes'] = int(n_states)
            new_config.pop('n_states', None)
        new_config['mode'] = mode

        ### Deal with the cross validation split
        mode_name,mode_index = mode.rsplit('_',1)
        # Deal with cross validation case
        if mode_name!='repeat':
            # Update the new_config['cv_kwargs']
            new_config['indices'] = f'{new_config["save_dir"]}/{mode_name}_partition/fold_indices_{mode_index}.json'
        ### Update the save_dir
        new_config['save_dir'] = f'{new_config["save_dir"]}/{model}_state_{n_states}/{mode}/'

        batch_train = BatchTrain(new_config)
        batch_train.model_train()

    def make_list(self):
        """
        Make the list of batch variables with respect to index,
        and save them to f'{self.header["save_dir"]}config_list.csv'
        Returns
        -------
        """

        mode = self.config['mode']
        mode_index = self._generate_mode_indices(mode)

        model_index = self.config['model'].keys()
        n_states = self.config['n_states']

        combinations = list(product(model_index, n_states,mode_index))
        df = pd.DataFrame(combinations, columns=['model_index', 'n_states','mode_index'])
        df.to_csv(os.path.join(self.save_dir,'config_list.csv'), index=True)

    def _generate_mode_indices(self,mode):
        mode_index = []
        for key in ['bcv', 'ncv', 'split']:
            if key in mode:
                cv_split = CrossValidationSplit(**mode[key])
                cv_split.save(os.path.join(self.save_dir, f'{key}_partition/'))
                mode_index.extend([f'{key}_{i}' for i in range(1, cv_split.get_n_splits() + 1)])

        if 'repeat' in mode:
            n_realizations = mode['repeat'].get('n_realizations', 3)
            mode_index.extend([f'repeat_{i}' for i in range(1, n_realizations + 1)])

        return mode_index

class BatchTrain():
    def __init__(self,config):
        pass

    def model_train(self):
        pass