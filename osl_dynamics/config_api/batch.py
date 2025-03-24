import os
import pickle
import time
import random
import json
import yaml
import shutil
import logging

import numpy as np
import pandas as pd
from typing import Union
from itertools import product

from ..evaluate.cross_validation import CrossValidationSplit, BiCrossValidation
from ..config_api.wrappers import train_model

from .pipeline import run_pipeline_from_file
from ..data.base import Data
from ..inference.metrics import twopair_riemannian_distance
from ..inference.modes import (argmax_time_courses, fractional_occupancies,
                               mean_lifetimes,mean_intervals,reweight_alphas,hungarian_pair)
from ..analysis.power import independent_components_to_surface_maps as ic2surface
from ..analysis.workbench import render
from ..utils.misc import override_dict_defaults
from ..utils.plotting import plot_box, plot_alpha, plot_violin, plot_mode_pairing, plot_matrices, plot_brain_surface
from ..array_ops import cov2corr, first_eigenvector

_logger = logging.getLogger("osl-dynamics")


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
        time.sleep(random.uniform(0., 2.))  # Prevent job conflicts

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
        config_list = pd.read_csv(os.path.join(self.save_dir, 'config_list.csv'), index_col=0)

        model, n_states, mode = config_list.iloc[index]

        _logger.info(f"Configuration: {self.config}")
        _logger.info(f"Model: {model}, n_states: {n_states}, mode: {mode}")
        '''
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
        '''
        batch_train = BatchTrain(self.config, model, n_states, mode)
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

        combinations = list(product(model_index, n_states, mode_index))
        df = pd.DataFrame(combinations, columns=['model_index', 'n_states', 'mode_index'])
        df.to_csv(os.path.join(self.save_dir, 'config_list.csv'), index=True)

    def _generate_mode_indices(self, mode):
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


class BatchTrain:
    """
    Converts a batch training configuration file to another config
    for the training pipeline.
    """

    def __init__(self, config: dict, model: str, n_states: int, mode: str):
        self.config = self._prepare_config(config, model, n_states, mode)
        self.save_dir = self.config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        self._save_batch_config()

    def _prepare_config(self, config: dict, model: str, n_states: int, mode: str) -> dict:
        """Prepares the training configuration."""
        new_config = config.copy()

        # Handle cross-validation splits
        mode_name, mode_index = mode.rsplit('_', 1)
        if mode_name != 'repeat':
            new_config['indices'] = os.path.join(new_config['save_dir'],
                                                 f'{mode_name}_partition/fold_indices_{mode_index}.json')

        # Preserve only the relevant model
        model_config = new_config['model'].get(model, {})
        new_config['model'] = {model: model_config}

        if model != 'dynemo':
            new_config['model'][model]['config_kwargs']['n_states'] = int(n_states)
        else:
            new_config['model'][model]['config_kwargs']['n_modes'] = int(n_states)
            new_config['model'][model]['config_kwargs'].pop('n_states', None)

        new_config['mode'] = mode
        new_config['save_dir'] = os.path.join(config['save_dir'], f'{model}_state_{n_states}/{mode}/')

        new_config.pop('n_states')

        return new_config

    def _save_batch_config(self):
        """Saves the batch configuration file."""
        config_path = os.path.join(self.save_dir, 'batch_config.yaml')
        if not os.path.isfile(config_path):
            with open(config_path, 'w') as file:
                yaml.safe_dump(self.config, file, default_flow_style=False)

    def model_train(self):
        """
        Trains the model based on the configuration mode.
        """
        mode = self.config['mode']

        if 'split' in mode:
            self._handle_split()
        elif 'ncv' in mode:
            self._handle_ncv()
        elif 'bcv' in mode:
            self._handle_bcv()
        elif 'repeat' in mode:
            self._handle_repeat()
        else:
            raise ValueError("Invalid mode. Must contain 'bcv', 'ncv', 'split', or 'repeat'.")

    def _handle_split(self):
        """Handles split-based training."""
        with open(self.config['indices'], 'r') as file:
            indices_list = list(json.load(file).values())

        for i, indices in enumerate(indices_list):
            temp_save_dir = os.path.join(self.config['save_dir'], f'partition_{i + 1}/')
            os.makedirs(temp_save_dir, exist_ok=True)

            indices_path = os.path.join(temp_save_dir, f'indices_{i + 1}.json')
            with open(indices_path, 'w') as file:
                json.dump(indices, file)

            self._run_pipeline(temp_save_dir, indices_path)

    def _handle_ncv(self):
        """Handles naive cross-validation training."""
        '''
        ncv = NCV(self.config)
        ncv.validate()
        '''

    def _handle_bcv(self):
        """Handles bi-cross-validation training."""
        bcv = BiCrossValidation(self.config)
        bcv.validate()

    def _handle_repeat(self):
        """Handles repeated training mode."""
        #temp_save_dir = os.path.join(self.config['save_dir'], 'tmp/')
        #os.makedirs(temp_save_dir, exist_ok=True)
        self._run_pipeline(self.config['save_dir'])

    def _run_pipeline(self, save_dir: str, indices_path: str = None):
        """Runs the training pipeline with the given save directory and optional indices."""
        # Get model and model_kwargs
        model, model_kwargs = next(iter(self.config['model'].items()))

        prepare_config = {'model_type':model,
                           'data':self.config['load_data'],
                           'output_dir':save_dir}

        # Add keys only if they exist in self.model_kwargs
        for key in ["config_kwargs", "init_kwargs", "fit_kwargs"]:
            if key in model_kwargs:
                prepare_config[key] = model_kwargs[key]
        prepare_config['data'].setdefault('kwargs', {})['store_dir'] = f'{save_dir}/tmp/'
        if indices_path:
            prepare_config['keep_list'] = indices_path

        config_path = os.path.join(save_dir, 'prepared_config.yaml')
        with open(config_path, 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)

        train_model(**prepare_config)


class BatchAnalysis:
    '''
    Analysis code after batch training. The config path in initialisation should contain
    :code:`config_root.yaml` and :code:`config_list.csv`
    '''

    def __init__(self, config_path):
        self.config_path = config_path
        with open(os.path.join(config_path, 'config_root.yaml'), 'r') as file:
            self.config_root = yaml.safe_load(file)
        self.indexparser = IndexParser(self.config_root)
        self.config_list = pd.read_csv(os.path.join(config_path, 'config_list.csv'), index_col=0)
        self.analysis_path = os.path.join(config_path, 'analysis')
        if not os.path.exists(self.analysis_path):
            os.makedirs(self.analysis_path)

    def compare(self, demean_index=-1, inset_start_index=None, plot_end_index=None, fig_kwargs=None, folder='Y_test/',
                object='log_likelihood'):
        '''
        By default of bi-cross validation, we should compare the final log_likelihood on the Y_test.
        But for sanity check, and potentiall understand how the method work, we are also interested in
        the folder Y_train/metrics, X_test/metrics.
        '''
        models = self.config_root['model'].keys()
        n_states_list = self.config_root['n_states']

        metrics = {model: {str(int(num)): [] for num in n_states_list} for model in models}
        for i in range(len(self.config_list)):

            config = self.indexparser.parse(i)
            model = next(iter(config['model']))

            if model != 'dynemo':
                n_states = config['n_states']
            else:
                n_states = config['n_modes']

            save_dir = config['save_dir']
            mode = config['mode']
            if 'bcv' in mode:
                try:
                    with open(os.path.join(save_dir, folder, 'metrics.json'), 'r') as file:
                        metric = json.load(file)[object]
                    metrics[model][str(int(n_states))].append(metric)
                except Exception:
                    print(f'save_dir {save_dir} fails!')
                    metrics[model][str(int(n_states))].append(np.nan)

        # Plot
        for model in models:
            if plot_end_index is not None:
                n_states_list = n_states_list[:plot_end_index]
            temp_values = [metrics[model][str(key)] for key in n_states_list]
            plot_box(data=temp_values,
                     labels=n_states_list,
                     demean=True,
                     demean_index=demean_index,
                     x_label=r'$N_{states}$',
                     y_label='Bi-cross validated log likelihood',
                     inset_start_index=inset_start_index,
                     fig_kwargs=fig_kwargs,
                     filename=os.path.join(self.analysis_path, f'{model}_{folder.split("/")[0]}_{object}_demean.pdf')
                     )
            plot_box(data=temp_values,
                     labels=n_states_list,
                     demean=True,
                     demean_index=demean_index,
                     x_label=r'$N_{states}$',
                     y_label='Bi-cross validated log likelihood',
                     inset_start_index=inset_start_index,
                     fig_kwargs=fig_kwargs,
                     filename=os.path.join(self.analysis_path, f'{model}_{folder.split("/")[0]}_{object}_demean.svg')
                     )
            plot_box(data=temp_values,
                     labels=n_states_list,
                     demean=False,
                     demean_index=demean_index,
                     x_label=r'$N_{states}$',
                     y_label='Bi-cross validated log likelihood',
                     inset_start_index=inset_start_index,
                     fig_kwargs=fig_kwargs,
                     filename=os.path.join(self.analysis_path, f'{model}_{folder.split("/")[0]}_{object}.pdf')
                     )
            plot_box(data=temp_values,
                     labels=n_states_list,
                     demean=False,
                     demean_index=demean_index,
                     x_label=r'$N_{states}$',
                     y_label='Bi-cross validated log likelihood',
                     inset_start_index=inset_start_index,
                     fig_kwargs=fig_kwargs,
                     filename=os.path.join(self.analysis_path, f'{model}_{folder.split("/")[0]}_{object}.svg')
                     )

    def temporal_analysis(self, demean=False, inset_start_index=None, theme='reproducibility', normalisation=False):
        if theme == 'reproducibility':
            directory_list = [['fold_1_1/X_train/inf_params/alp.pkl', 'fold_1_2/X_train/inf_params/alp.pkl'],
                              ['fold_2_1/X_train/inf_params/alp.pkl', 'fold_2_2/X_train/inf_params/alp.pkl']]
        elif theme == 'compromise':
            directory_list = [['fold_1_1/X_train/inf_params/alp.pkl', 'fold_2_1/Y_test/inf_params/alp.pkl'],
                              ['fold_1_2/X_train/inf_params/alp.pkl', 'fold_2_2/Y_test/inf_params/alp.pkl'],
                              ['fold_2_1/X_train/inf_params/alp.pkl', 'fold_1_1/Y_test/inf_params/alp.pkl'],
                              ['fold_2_2/X_train/inf_params/alp.pkl', 'fold_1_2/Y_test/inf_params/alp.pkl']]
        elif theme == 'fixed':
            directory_list = [['fold_1_1/X_train/inf_params/alp.pkl', 'fold_2_2/Y_test/inf_params/alp.pkl'],
                              ['fold_1_2/X_train/inf_params/alp.pkl', 'fold_2_1/Y_test/inf_params/alp.pkl'],
                              ['fold_2_1/X_train/inf_params/alp.pkl', 'fold_1_2/Y_test/inf_params/alp.pkl'],
                              ['fold_2_2/X_train/inf_params/alp.pkl', 'fold_1_1/Y_test/inf_params/alp.pkl']]
        else:
            raise ValueError('Invalid theme presented!')

        models = self.config_root['batch_variable']['model']
        n_states = self.config_root['batch_variable']['n_states']
        modes = self.config_root['batch_variable']['mode']
        modes = [mode for mode in modes if 'cv' in mode]
        metrics = {model: {str(int(num)): [] for num in n_states} for model in models}
        temporal_directory = os.path.join(self.analysis_path, 'temporal_analysis')
        if not os.path.exists(temporal_directory):
            os.makedirs(temporal_directory)
        for model in models:
            for n_state in n_states:
                for mode in modes:
                    save_dir = (f"{self.config_root['save_dir']}/"
                                f"{model}_ICA_{self.config_root['non_batch_variable']['n_channels']}_state_{n_state}/"
                                f"{mode}/")
                    count = 1
                    for directory in directory_list:
                        try:
                            temp = self._temporal_reproducibility(
                                os.path.join(save_dir, directory[0]),
                                os.path.join(save_dir, directory[1]),
                                n_states=n_state,
                                normalisation=normalisation,
                                filename=os.path.join(temporal_directory,
                                                      f"{model}_{n_state}_{mode}_{theme}_{count}.jpg"))
                            count += 1
                            metrics[model][str(int(n_state))].append(temp)
                        except Exception:
                            print(f'Case {model} {n_state} {mode} {theme} fails!')
                            metrics[model][str(int(n_state))].append(np.nan)
        # Save the dictionary to a file
        with open(os.path.join(self.analysis_path, f'{model}_temporal_analysis_{theme}.pkl'), 'wb') as file:
            pickle.dump(metrics, file)
        for model in models:
            temp_keys = list(metrics[model].keys())
            temp_values = [metrics[model][key] for key in temp_keys]
            plot_box(data=temp_values,
                     labels=temp_keys,
                     demean=demean,
                     inset_start_index=inset_start_index,
                     filename=os.path.join(self.analysis_path,
                                           f'{model}_temporal_analysis_{theme}{"_norm" if normalisation else ""}.jpg')
                     )

    def spatial_analysis(self, demean=False, inset_start_index=None, theme='reproducibility', normalisation=False):
        if theme == 'reproducibility':
            directory_list = [['fold_1_1/Y_train/inf_params/covs.npy', 'fold_2_1/Y_train/inf_params/covs.npy'],
                              ['fold_1_2/Y_train/inf_params/covs.npy', 'fold_2_2/Y_train/inf_params/covs.npy']]
        elif theme == 'fixed':
            directory_list = [['fold_1_1/Y_train/inf_params/covs.npy', 'fold_1_2/X_train/dual_estimates/covs.npy'],
                              ['fold_1_2/Y_train/inf_params/covs.npy', 'fold_1_1/X_train/dual_estimates/covs.npy'],
                              ['fold_2_1/Y_train/inf_params/covs.npy', 'fold_2_2/X_train/dual_estimates/covs.npy'],
                              ['fold_2_2/Y_train/inf_params/covs.npy', 'fold_2_1/X_train/dual_estimates/covs.npy']]
        else:
            raise ValueError('Invalid theme presented!')

        models = self.config_root['batch_variable']['model']
        n_states = self.config_root['batch_variable']['n_states']
        modes = self.config_root['batch_variable']['mode']
        modes = [mode for mode in modes if 'cv' in mode]
        metrics = {model: {str(int(num)): [] for num in n_states} for model in models}
        spatial_directory = os.path.join(self.analysis_path, 'spatial_analysis')
        if not os.path.exists(spatial_directory):
            os.makedirs(spatial_directory)
        for model in models:
            for n_state in n_states:
                for mode in modes:
                    save_dir = (f"{self.config_root['save_dir']}/"
                                f"{model}_ICA_{self.config_root['non_batch_variable']['n_channels']}_state_{n_state}/"
                                f"{mode}/")
                    count = 1
                    for directory in directory_list:
                        try:
                            temp = self._spatial_reproducibility(
                                os.path.join(save_dir, directory[0]),
                                os.path.join(save_dir, directory[1]),
                                normalisation=normalisation,
                                filename=os.path.join(spatial_directory,
                                                      f"{model}_{n_state}_{mode}_{theme}_{count}.jpg"))
                            count += 1
                            metrics[model][str(int(n_state))].append(temp)
                        except Exception:
                            print(f'Case {model} {n_state} {mode} {theme} fails!')
                            metrics[model][str(int(n_state))].append(np.nan)
        for model in models:
            temp_keys = list(metrics[model].keys())
            temp_values = [metrics[model][key] for key in temp_keys]
            plot_box(data=temp_values,
                     labels=temp_keys,
                     demean=demean,
                     inset_start_index=inset_start_index,
                     filename=os.path.join(self.analysis_path,
                                           f'{model}_spatial_analysis_{theme}{"_norm" if normalisation else ""}.jpg')
                     )

    def plot_training_loss(self, metrics=['free_energy']):

        models = self.config_root['model'].keys()
        n_states_list = self.config_root['n_states'].copy()
        # Remove the case where n_states = 1 because no dFC model
        if 1 in n_states_list:
            n_states_list.remove(1)

        loss = {metric: {model: {str(int(num)): [] for num in n_states_list} for model in models} for metric in metrics}
        for i in range(len(self.config_list)):
            config = self.indexparser.parse(i)
            model = next(iter(config['model']))
            if model != 'dynemo':
                n_states = config['n_states']
            else:
                n_states = config['n_modes']
            save_dir = config['save_dir']
            mode = config['mode']

            ### SWC does not have an explicit model, sFC does not either.
            if 'repeat' in mode and int(n_states) > 1 and model != 'swc':
                try:
                    with open(f'{save_dir}/metrics/metrics.json', "r") as file:
                        data = json.load(file)
                    for metric in metrics:
                        loss[metric][model][str(int(n_states))].append(data[metric])
                except Exception:
                    print(f'save_dir {save_dir} fails!')
                    for metric in metrics:
                        loss[metric][model][str(int(n_states))].append(np.nan)

        # Plot
        for metric in metrics:
            for model in models:
                temp_values = [loss[metric][model][str(key)] for key in n_states_list]
                plot_box(data=temp_values,
                         labels=n_states_list,
                         mark_best=False,
                         x_label=r'$N_{states}$',
                         y_label=metric,
                         title='Training loss',
                         filename=os.path.join(self.analysis_path, f'{model}_{metric}.svg')
                         )
                plot_box(data=temp_values,
                         labels=n_states_list,
                         mark_best=False,
                         x_label=r'$N_{states}$',
                         y_label=metric,
                         title='Training loss',
                         filename=os.path.join(self.analysis_path, f'{model}_{metric}.pdf')
                         )

    def plot_naive_cv(self):
        models = self.config_root['model'].keys()
        n_states_list = self.config_root['n_states'].copy()
        # Remove the case where n_states = 1 because no dFC model
        if 1 in n_states_list:
            n_states_list.remove(1)

        free_energy = {model: {str(int(num)): [] for num in n_states_list} for model in models}
        for i in range(len(self.config_list)):
            config = self.indexparser.parse(i)
            model = next(iter(config['model']))
            if model == 'swc':
                file_name = 'metrics.json'
            else:
                file_name = 'ncv_free_energy.json'
            n_states = config.get('n_states', config.get('n_modes'))
            save_dir = config['save_dir']
            mode = config['mode']
            if 'ncv' in mode and int(n_states) > 1:
                try:
                    with open(f'{save_dir}/{file_name}', 'r') as file:
                        if model == 'swc':
                            free_energy[model][str(int(n_states))].append(float(json.load(file)['log_likelihood']))
                        else:
                            free_energy[model][str(int(n_states))].append(float(json.load(file)[0]))
                except Exception:
                    print(f'save_dir {save_dir} fails!')

        for model in models:
            temp_keys = list(free_energy[model].keys())
            temp_values = [free_energy[model][key] for key in temp_keys]
            x_label = {'hmm': 'N_states', 'swc': 'N_states', 'dynemo': 'N_modes'}
            plot_box(data=temp_values,
                     labels=temp_keys,
                     mark_best=False,
                     demean=False,
                     x_label=x_label[model],
                     y_label='Free energy',
                     title='Naive cross validation Analysis',
                     filename=os.path.join(self.analysis_path, f'{model}_naive_free_energy.svg')
                     )

    def plot_split_half_reproducibility(self):
        models = self.config_root['model'].keys()
        n_states_list = self.config_root['n_states'].copy()
        rep = {model: {str(int(num)): [] for num in n_states_list} for model in models}
        rep_path = os.path.join(self.analysis_path, 'rep')
        if not os.path.exists(rep_path):
            os.makedirs(rep_path)

        for i in range(len(self.config_list)):
            config = self.indexparser.parse(i)
            model = next(iter(config['model']))
            if model != 'dynemo':
                n_states = config['n_states']
            else:
                n_states = config['n_modes']
            save_dir = config['save_dir']
            mode = config['mode']

            if 'split' in mode:
                try:
                    cov_1 = np.load(f'{save_dir}/partition_1/inf_params/covs.npy')
                    cov_2 = np.load(f'{save_dir}/partition_2/inf_params/covs.npy')
                    rep[model][str(int(n_states))].append(self._reproducibility_analysis(cov_1, cov_2,
                                                                                         filename=os.path.join(rep_path,
                                                                                                               f'{model}_state_{n_states}_{mode}.svg')))
                except Exception:
                    print(f'save_dir {save_dir} fails!')
                    rep[model][str(int(n_states))].append(np.nan)

        for model in models:
            temp_values = [rep[model][str(key)] for key in n_states_list]
            plot_box(data=temp_values,
                     labels=n_states_list,
                     mark_best=False,
                     demean=False,
                     x_label='N_states',
                     y_label='Average Riemannian distance',
                     title='Reproducibility Analysis',
                     filename=os.path.join(self.analysis_path, f'{model}_reproducibility.svg')
                     )
            plot_box(data=temp_values,
                     labels=n_states_list,
                     x_label='N_states',
                     y_label='Average Riemannian distance',
                     title='Reproducibility Analysis',
                     filename=os.path.join(self.analysis_path, f'{model}_reproducibility.pdf')
                     )

    def plot_fo(self, plot_mode='repeat_1'):
        from osl_dynamics.inference.modes import argmax_time_courses, fractional_occupancies
        from osl_dynamics.utils.plotting import plot_violin
        models = self.config_root['batch_variable']['model']
        n_states = self.config_root['batch_variable']['n_states']
        fo_path = os.path.join(self.analysis_path, 'fo')
        if not os.path.exists(fo_path):
            os.makedirs(fo_path)
        for i in range(len(self.config_list)):
            config = self.indexparser.parse(i)
            model = config['model']
            n_states = config['n_states']
            save_dir = config['save_dir']
            mode = config['mode']
            if plot_mode == mode:
                try:
                    with open(f'{save_dir}/inf_params/alp.pkl', "rb") as file:
                        alpha = pickle.load(file)
                    stc = argmax_time_courses(alpha)
                    fo = fractional_occupancies(stc)
                    plot_violin(fo.T, x_label="State", y_label="FO", title=f'Fractional Occupancy, {n_states} states',
                                filename=os.path.join(fo_path, f'state_{n_states}.jpg'))


                except Exception:
                    print(f'save_dir {save_dir} fails!')

    def _reproducibility_analysis(self, cov_1, cov_2, filename=None):
        if not os.path.exists(os.path.join(self.analysis_path, 'rep')):
            os.makedirs(os.path.join(self.analysis_path, 'rep'))
        from osl_dynamics.inference.metrics import twopair_riemannian_distance
        from osl_dynamics.inference.modes import hungarian_pair
        from osl_dynamics.utils.plotting import plot_mode_pairing
        riem = twopair_riemannian_distance(cov_1, cov_2)
        indice, riem_reorder = hungarian_pair(riem, distance=True)
        plot_mode_pairing(riem_reorder, indice, x_label='2nd half states', y_label='1st half states',
                          filename=filename)
        return np.mean(np.diagonal(riem_reorder))

    def _spatial_reproducibility(self, cov_1, cov_2, normalisation=False, filename=None):

        if isinstance(cov_1, str):
            cov_1 = np.load(cov_1)
        if isinstance(cov_2, str):
            cov_2 = np.load(cov_2)
        riem = twopair_riemannian_distance(cov_1, cov_2)
        indice, riem_reorder = hungarian_pair(riem, distance=True)
        plot_mode_pairing(riem_reorder, indice, x_label='2nd half', y_label='1st half',
                          filename=filename)
        mean_diagonal = np.mean(np.diagonal(riem_reorder))
        if normalisation:
            off_diagonal_indices = np.where(~np.eye(riem_reorder.shape[0], dtype=bool))
            mean_off_diagonal = np.mean(riem_reorder[off_diagonal_indices])
            var_off_diagonal = np.var(riem_reorder[off_diagonal_indices])

            if len(riem_reorder) == 2:
                var_off_diagonal = riem_reorder[0, 1] ** 2

            # Return the mean diagonal value divided by the mean off-diagonal value
            return (mean_diagonal - mean_off_diagonal) / np.sqrt(var_off_diagonal)
        else:
            return mean_diagonal

    def _temporal_reproducibility(self, alpha_1, alpha_2, n_states, normalisation=False, filename=None):
        from osl_dynamics.inference.metrics import alpha_correlation
        from osl_dynamics.inference.modes import hungarian_pair, argmax_time_courses
        from osl_dynamics.utils.plotting import plot_mode_pairing
        from osl_dynamics.array_ops import get_one_hot

        if isinstance(alpha_1, str):
            with open(alpha_1, 'rb') as file:
                alpha_1 = pickle.load(file)

        if isinstance(alpha_2, str):
            with open(alpha_2, 'rb') as file:
                alpha_2 = pickle.load(file)

        # Get one-hot coding if the original one is not
        if alpha_1[0].ndim == 1:
            alpha_1 = get_one_hot(alpha_1, n_states)
        if alpha_2[0].ndim == 1:
            alpha_2 = get_one_hot(alpha_2, n_states)
        # Argmax time courses.
        alpha_1 = argmax_time_courses(alpha_1)
        alpha_2 = argmax_time_courses(alpha_2)

        corr = alpha_correlation(alpha_1, alpha_2, return_diagonal=False)
        indice, corr_reorder = hungarian_pair(corr, distance=False)
        plot_mode_pairing(corr_reorder, indice, x_label='2nd half', y_label='1st half',
                          filename=filename)
        mean_diagonal = np.mean(np.diagonal(corr_reorder))
        if normalisation:
            off_diagonal_indices = np.where(~np.eye(corr_reorder.shape[0], dtype=bool))
            mean_off_diagonal = np.mean(corr_reorder[off_diagonal_indices])
            var_off_diagonal = np.var(corr_reorder[off_diagonal_indices])

            if n_states == 2:
                var_off_diagonal = corr_reorder[0, 1] ** 2

            # Return the mean diagonal value divided by the mean off-diagonal value
            return (mean_diagonal - mean_off_diagonal) / np.sqrt(var_off_diagonal)
        else:
            return mean_diagonal

    def post_hoc_analysis(self, model='hmm', n_state=6, sampling_frequency=1.389):
        save_dir = (f'{self.config_path}/{model}_state_{n_state}/repeat_1/')

        plot_dir = f'{self.analysis_path}/{model}_state_{n_state}/'

        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
        os.makedirs(plot_dir)

        # Copy the loss function to confirm that training is successful.
        loss_function_file = f'{save_dir}/loss_function.pdf'

        # Check if the file exists before copying
        if os.path.exists(loss_function_file):
            shutil.copy(loss_function_file, plot_dir)
            print(f"Copied {loss_function_file} to {plot_dir}")
        else:
            print(f"File {loss_function_file} does not exist. Skipping copy.")

        # Deal with time courses.
        alpha_path = f'{save_dir}/inf_params/alp.pkl'
        cov_path = f'{save_dir}/inf_params/covs.npy'

        # Check if the pickle file exists before reading
        if os.path.exists(alpha_path):
            with open(alpha_path, 'rb') as file:
                alpha = pickle.load(file)  # Load the pickle file into the variable `alpha`
                print("Pickle file loaded successfully. Variable 'alpha' is now available.")
        else:
            alpha = None
            print(f"Pickle file {alpha_path} does not exist. Variable 'alpha' is set to None.")

        # Check if the pickle file exists before reading
        if os.path.exists(cov_path):
            covs = np.load(cov_path)
            print("Covariance loaded successfully. Variable 'covs' is now available.")
        else:
            covs = None
            print(f"Covariance file does not exist. Variable 'covs' is set to None.")

        plot_alpha(alpha[0], n_samples=1200, filename=f'{plot_dir}/alpha.pdf')

        if model == 'hmm':
            stc = argmax_time_courses(alpha)

            # Fractional occupancy
            fo = fractional_occupancies(stc)
            print(f'Fractional occupancy shape: {fo.shape}')
            plot_box(fo.T.tolist(),
                     labels=list(range(1, len(fo.T) + 1)),
                     plot_samples=False,
                     mark_best=False,
                     plot_kwargs={'showfliers': True},
                     x_label="State",
                     y_label="Fractional Occupancy",
                     filename=f'{plot_dir}/fo.svg'
                     )

            # Mean lifetime
            lt = mean_lifetimes(stc, sampling_frequency)
            plot_box(lt.T.tolist(),
                     labels=list(range(1, len(lt.T) + 1)),
                     plot_samples=False,
                     mark_best=False,
                     plot_kwargs={'showfliers': True},
                     x_label="State",
                     y_label="Mean Lifetime (s)",
                     filename=f'{plot_dir}/lt.pdf'
                     )

            # Mean intervals
            intv = mean_intervals(stc, sampling_frequency)
            plot_box(intv.T.tolist(),
                     labels=list(range(1, len(intv.T) + 1)),
                     plot_samples=False,
                     mark_best=False,
                     plot_kwargs={'showfliers': True},
                     x_label="State",
                     y_label="Mean Interval (s)",
                     filename=f'{plot_dir}/intv.pdf')

        elif model == 'dynemo':
            norm_alpha = reweight_alphas(alpha, covs)
            plot_alpha(norm_alpha[0], n_samples=1200, filename=f'{plot_dir}norm_alpha.pdf')

            mean_norm_alpha = np.array([np.mean(a, axis=0) for a in norm_alpha])
            plot_box(mean_norm_alpha.T.tolist(),
                     labels=list(range(1, len(mean_norm_alpha.T) + 1)),
                     plot_samples=False,
                     mark_best=False,
                     plot_kwargs={'showfliers': True},
                     x_label="Mode",
                     y_label="Mean alpha",
                     filename=f'{plot_dir}/mean_norm_alpha.pdf'
                     )

            max_norm_alpha = np.array([np.max(a, axis=0) for a in norm_alpha])
            plot_box(max_norm_alpha.T.tolist(),
                     labels=list(range(1, len(max_norm_alpha.T) + 1)),
                     plot_samples=False,
                     mark_best=False,
                     plot_kwargs={'showfliers': True},
                     x_label="Mode",
                     y_label="Max alpha",
                     filename=f'{plot_dir}/max_norm_alpha.pdf'
                     )

            std_norm_alpha = np.array([np.std(a, axis=0) for a in norm_alpha])
            plot_box(std_norm_alpha.T.tolist(),
                     labels=list(range(1, len(std_norm_alpha.T) + 1)),
                     plot_samples=False,
                     mark_best=False,
                     plot_kwargs={'showfliers': True},
                     x_label="Mode",
                     y_label="Std alpha",
                     filename=f'{plot_dir}/std_norm_alpha.svg'
                     )

        if covs is not None:
            # Plot the covariance matrix first.
            plot_matrices(covs,
                          group_color_scale=False,
                          # titles=[f'Matrix {i+1}' for i in range(len(covs))],
                          filename=f'{plot_dir}/covs.pdf')

            # Convert covs to corrs and then plot
            corrs = cov2corr(covs)

            corrs_zero_diag = np.copy(corrs)

            # Set the diagonal of each (n_channels, n_channels) matrix to zero
            for i in range(corrs_zero_diag.shape[0]):
                np.fill_diagonal(corrs_zero_diag[i], 0)
            plot_matrices(corrs_zero_diag,
                          group_color_scale=True,
                          cmap='coolwarm',
                          v_min=-1.0,
                          v_max=1.0,
                          # titles=[f'Matrix {i+1}' for i in range(len(covs))],
                          filename=f'{plot_dir}/corrs.pdf')
            # Rank-one approximation
            r1_approxs = []
            sum_of_degrees = []
            for i in range(len(corrs)):
                correlation = corrs[i, :, :]
                r1_approxs.append(first_eigenvector(correlation))
                np.fill_diagonal(correlation, 0)
                sum_of_degrees.append(np.sum(correlation, axis=1))
            r1_approxs = np.array(r1_approxs)
            sum_of_degrees = np.array(sum_of_degrees)
            np.save(f'{save_dir}r1_approx_FC.npy', r1_approxs)
            np.save(f'{plot_dir}/corr_sum_of_degree.npy', sum_of_degrees)
            '''
            ic2surface(ica_spatial_maps = f'{self.config_root["spatial_map"]}/melodic_IC.dscalar.nii',
                       ic_values=sum_of_degrees,
                       output_file=f'{plot_dir}/corr_sum_of_degree_surface_map.dscalar.nii')
            ic2surface(ica_spatial_maps=f'{self.config_root["spatial_map"]}/melodic_IC.dscalar.nii',
                       ic_values=r1_approxs,
                       output_file=f'{plot_dir}/corr_r1_approx_surface_map.dscalar.nii')

            render(img=f'{plot_dir}/corr_sum_of_degree_surface_map.dscalar.nii',
                   save_dir=f'{plot_dir}/brain_map/sum_of_degree',
                   gui=False,
                   image_name=f'{plot_dir}/brain_map/fc_sum_of_degree',
                   input_is_cifti=True)
            render(img=f'{plot_dir}/corr_r1_approx_surface_map.dscalar.nii',
                   save_dir=f'{plot_dir}/brain_map/r1_approx',
                   gui=False,
                   image_name=f'{plot_dir}/brain_map/fc_r1_approx',
                   input_is_cifti=True)
            '''

            # Calculate the Riemannian distance of covariance matrices
            riem = twopair_riemannian_distance(covs, covs)
            if model == 'dynemo':
                index = np.argsort(np.median(std_norm_alpha, axis=0))[::-1]
            else:
                index = np.argsort(np.median(fo, axis=0))[::-1]
            riem_reordered = riem[index, :]
            riem_reordered = riem_reordered[:, index]
            indices = {'row': (index + 1).tolist(), 'col': (index + 1).tolist()}
            plot_mode_pairing(riem, fig_kwargs={'figsize': (12, 9)}, filename=f'{plot_dir}/riem.pdf')
            plot_mode_pairing(riem_reordered, indices, fig_kwargs={'figsize': (12, 9)},
                              filename=f'{plot_dir}/riem_reordered.svg')

    def mode_contribution_analysis(self, model='dynemo', n_state=16, mode='bcv_1'):
        save_dir = (f'{self.config_path}/{model}_state_{n_state}/{mode}/')
        plot_dir = f'{self.analysis_path}/{model}_state_{n_state}_mode_contribution/'
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
        os.makedirs(plot_dir)
        # Analyse how much log likelihood each "mode" contributes to the overall log likelihood
        if model != 'dynemo':
            raise ValueError('This function is only for DyNeMo at present!')

        with open(f'{save_dir}/Y_test/prepared_config.yaml', "r") as file:
            config = yaml.safe_load(file)

        # Load data
        from osl_dynamics.config_api.wrappers import load_data
        load_data_kwargs = config['load_data']
        data = load_data(**load_data_kwargs)
        keep_list = config['keep_list']
        ts = [data[i] for i in keep_list]

        # Load covariances
        covs = np.load(f'{save_dir}/Y_train/inf_params/covs.npy')

        # Load posterior time courses
        with open(f'{save_dir}/Y_test/inf_params/alp.pkl', 'rb') as file:
            alp = pickle.load(file)

        # Concatenate everything to numpy
        ts_np = np.concatenate(ts)
        alp_np = np.concatenate(alp)

        print('Check the size of each numpy array')
        print(f'Shape of data:{ts_np.shape}')
        print(f'shape of alpha: {alp_np.shape}')

        # Compute the moment-to-moment covariance matrices
        C_t = np.einsum('ns,sij->nij', alp_np, covs)

        # Compute inverse and log determinant of each covariance matrix
        inv_C = np.linalg.inv(C_t)
        logdet_C = np.log(np.linalg.det(C_t))

        # Compute quadratic term using einsum
        quad_term = np.einsum('ni,nij,nj->n', ts_np, inv_C, ts_np)

        # Calculate log likelihood
        d = ts_np.shape[1]
        log_likelihood = -0.5 * (quad_term + logdet_C + d * np.log(2 * np.pi))
        print(f'Average log likelihood is: {log_likelihood}')

        contributions = np.zeros(16)

        for s in range(16):
            # Precompute inverse and logdet for mode s
            C_s = covs[s]
            invC_s = np.linalg.inv(C_s)
            logdet_s = np.log(np.linalg.det(C_s))

            # Compute quadratic term for all data points under mode s
            quad_term_s = np.einsum('ni,ij,nj->n', ts_np, invC_s, ts_np)

            # Log likelihood for mode s across all data points
            logpdf_s = -0.5 * (quad_term_s + logdet_s + d * np.log(2 * np.pi))

            # Weight by alpha and sum
            contributions[s] = np.sum(alp_np[:, s] * logpdf_s)

        print(f'Contribution of each mode:{contributions}')
        np.save(f'{plot_dir}/likelihood_contribute.npy', contributions)

    def silencing_analysis(self, model='dynemo', mode='bcv_1', start_index=12, threshold=0.002):
        import matplotlib.pyplot as plt
        from osl_dynamics.config_api.wrappers import load_data
        from osl_dynamics.inference.modes import reweight_alphas

        def calculate_log_likelihood(covs, alpha, ts, remove_modes=None):
            if isinstance(alpha, list):
                alpha = np.concatenate(alpha)
            if isinstance(ts, list):
                ts = np.concatenate(ts)

            if remove_modes is not None:
                alpha[:, np.array(remove_modes) - 1] = 0
            # Compute the moment-to-moment covariance matrices
            C_t = np.einsum('ns,sij->nij', alpha, covs)

            # Compute inverse and log determinant of each covariance matrix
            inv_C = np.linalg.inv(C_t)
            logdet_C = np.log(np.linalg.det(C_t))

            # Compute quadratic term using einsum
            quad_term = np.einsum('ni,nij,nj->n', ts, inv_C, ts)

            # Calculate log likelihood
            d = ts_np.shape[1]
            log_likelihood = -0.5 * (quad_term + logdet_C + d * np.log(2 * np.pi))
            return np.mean(log_likelihood)

        # Plot directory
        plot_dir = f'{self.analysis_path}/silencing/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_dir = f'{plot_dir}/{mode}/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        n_states = self.config_root['n_states']

        ll_before_silencing = []
        ll_after_silencing = []
        remove_modes = {}

        with open(f'{self.config_path}/{model}_state_1/{mode}/Y_test/prepared_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        load_data_kwargs = config['load_data']
        data = load_data(**load_data_kwargs)
        keep_list = config['keep_list']

        ts = [data[i] for i in keep_list]
        ts_np = np.concatenate(ts)

        for n_state in n_states:
            with open(f'{self.config_path}/{model}_state_{n_state}/{mode}/Y_test/metrics.json', 'r') as file:
                metric = json.load(file)['log_likelihood']
            ll_before_silencing.append(metric)

            # Check whether states need silencing
            if n_state <= start_index:
                ll_after_silencing.append(metric)
            else:
                # Load covariances
                covs = np.load(f'{self.config_path}/{model}_state_{n_state}/{mode}/Y_train/inf_params/covs.npy')
                # Load posterior time courses
                with open(f'{self.config_path}/{model}_state_{n_state}/{mode}//Y_test/inf_params/alp.pkl',
                          'rb') as file:
                    alp = pickle.load(file)
                norm_alp = reweight_alphas(alp, covs)
                std_norm_alpha = np.array([np.std(a, axis=0) for a in norm_alp])

                plot_box(std_norm_alpha.T.tolist(),
                         labels=list(range(1, len(std_norm_alpha.T) + 1)),
                         plot_samples=False,
                         mark_best=False,
                         plot_kwargs={'showfliers': True},
                         x_label="Mode",
                         y_label="Std Norm Alpha",
                         filename=f'{plot_dir}/std_norm_alpha_state_{n_state}.pdf'
                         )
                remove_modes[n_state] = (np.where(np.median(std_norm_alpha, axis=0) < threshold)[0] + 1).tolist()

                ll_after_silencing.append(
                    float(calculate_log_likelihood(covs, alp, ts, remove_modes=remove_modes[n_state])))

                print(f'Remove modes:{remove_modes}')

        with open(f'{plot_dir}/ll_before_silencing.json', "w") as f:
            json.dump(ll_before_silencing, f)
        with open(f'{plot_dir}/ll_after_silencing.json', "w") as f:
            json.dump(ll_after_silencing, f)
        with open(f'{plot_dir}/remove_modes.json', "w") as f:
            json.dump(remove_modes, f)

        # Create plot
        plt.figure(figsize=(6, 4))  # Set figure size
        plt.plot(n_states, ll_before_silencing, label="Before Silencing", marker="o", linestyle="-")
        plt.plot(n_states, ll_after_silencing, label="After Silencing", marker="s", linestyle="--")

        # Labels and title
        plt.xlabel("Number of Modes")
        plt.ylabel("Log-Likelihood")
        plt.title("Comparison of Log-Likelihood Before and After Silencing")
        plt.legend()  # Add legend
        plt.grid(True, linestyle="--", alpha=0.6)  # Add grid for better readability

        # Save the plot
        plt.savefig(f'{plot_dir}/ll_comparison.pdf', bbox_inches="tight")
        plt.close()