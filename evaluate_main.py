import time
import sys
import yaml
import json

from osl_dynamics.config_api.batch import IndexParser, BatchTrain


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





if __name__ == '__main__':
    index = int(sys.argv[1]) - 1
    config_path = sys.argv[2]
    with open(config_path,'r') as file:
        config_batch = yaml.safe_load(file)
    index_parser = IndexParser(config_batch)
    config = index_parser.parse(index)
    batch_train = BatchTrain()
    batch_train.model_train(config)
