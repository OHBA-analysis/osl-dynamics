import time
import sys
import yaml

#These are for the class IndexParser
import os
import numpy as np
import random
import time
import pandas as pd

def model_train_wrapper(config:dict):
    """
    This is the new main function for model training
    Parameters
    ----------
    config: dict
        the configuration file for model training

    Returns
    -------
    """
    from osl_dynamics.models.hmm import Config, Model

    # Currently, only HMM model works
    if config["model"] != 'HMM':
        raise ValueError('At present, only HMM model works!')
    # make the directory and save the configuration files.
    save_dir = f'{config["save_dir"]}{config["model"]}_ICA_{config["channel"]}' \
               f'_state_{config["state"]}/{config["mode"]}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f'{save_dir}config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    def model_train(config):
        import time
        import sys
        import yaml

        # These are for the class IndexParser
        import os
        import random
        import time
        import pandas as pd

        def model_train_wrapper(config: dict):
            """
            This is the new main function for model training
            Parameters
            ----------
            config: dict
                the configuration file for model training

            Returns
            -------
            """
            from osl_dynamics.models.hmm import Config, Model

            # Currently, only HMM model works
            if config["model"] != 'HMM':
                raise ValueError('At present, only HMM model works!')
            # make the directory and save the configuration files.
            save_dir = f'{config["save_dir"]}{config["model"]}_ICA_{config["channel"]}' \
                       f'_state_{config["state"]}/{config["mode"]}/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(f'{save_dir}config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

            def model_train(config):
                # Record the start time before training
                start_time = time.time()

                # Initiate a Model class and print a summary
                model = Model(config)
                model.summary()

                # Initialization
                init_history = model.random_state_time_course_initialization(dataset, n_epochs=2, n_init=10)

                # Full training
                history = model.fit(dataset)
                model.save(save_dir)

                # loss_history = history["loss"]
                np.save(f'{save_dir}/loss_history.npy', np.array(init_history['loss'] + history['loss']))

                end_time = time.time()

                # Calculate the training duration
                training_duration = end_time - start_time

                # Print or save the training duration
                print(f"Training time: {training_duration} seconds")
                # Save the training duration to a JSON file
                data = {"duration": training_duration}

                with open(f'{save_dir}/time.json', "w") as json_file:
                    json.dump(data, json_file)


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

        self.header = config['header']
        self.batch_variable = config['batch_variable']
        self.non_batch_variable = config['non_batch_variable']

        # Check whether the save_dir exists, make directory if not
        if not os.path.exists(self.header['save_dir']):
            os.makedirs(self.header['save_dir'])
        # Check whether the root configuration file exists, save if not
        if not os.path.exists(f'{self.header["save_dir"]}config_root.yaml'):
            with open(f'{self.header["save_dir"]}config_root.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

        # Check if the config list file exists, create if not
        if not os.path.exists(f'{self.header["save_dir"]}config_list.csv'):
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
        config_list = pd.read_csv(f'{self.header["save_dir"]}config_list.csv', index_col=0)

        # sv represents batch_variable given specific row
        bv = config_list.iloc[index].to_dict()

        # concatenate three parts of the dictionary
        new_config = {}
        new_config.update(self.header)
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
        df.to_csv(f'{self.header["save_dir"]}config_list.csv', index=True)


if __name__ == '__main__':
    index = int(sys.argv[1]) - 1
    config_path = sys.argv[2]
    with open(config_path,'r') as file:
        config = yaml.safe_load(file)
    index_parser = IndexParser(config)
    config = index_parser.parse(index)
    model_train_wrapper(config)
