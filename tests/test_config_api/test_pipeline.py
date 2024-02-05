import numpy as np
import numpy.testing as npt
from osl_dynamics.config_api.pipeline import run_pipeline_from_file

def test_run_pipeline_from_file():
    import os
    import json
    import yaml
    from shutil import rmtree
    save_dir = './test_pipeline_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def generate_obs(cov,n_timepoints=300):
        return np.random.multivariate_normal(np.zeros(len(cov)),cov,n_timepoints)
    # Define the covariance matrices of state 1,2 in both splits
    cor = [0.25,0.8]
    split_1_cov_1 = np.array([[1.0,cor[0]],[cor[0],1.0]]) * 2
    split_1_cov_2 = np.array([[1.0,-cor[0]],[-cor[0],1.0]]) * 2
    split_2_cov_1 = np.array([[1.0,cor[1]],[cor[1],1.0]]) * 0.5
    split_2_cov_2 = np.array([[1.0,-cor[1]],[-cor[1],1.0]]) * 0.5
    split_0_cov = np.array([[100.,-25.0],[-25.0,100.0]])
    split_0_state = generate_obs(split_0_cov, n_timepoints=3600)

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for i in range(0, 500, 2):
        state_1 = generate_obs(split_1_cov_1)
        state_2 = generate_obs(split_1_cov_2)
        obs = np.tile(np.concatenate([state_1,state_2], axis=0),(2,1))

        filename = f"{data_dir}{10001 + i}.txt"
        np.savetxt(filename, np.concatenate([obs,split_0_state],axis=0))

    for i in range(0, 500, 2):
        state_1 = generate_obs(split_2_cov_1)
        state_2 = generate_obs(split_2_cov_2)
        obs = np.tile(np.concatenate([state_1, state_2], axis=0), (2, 1))
        filename = f"{data_dir}{10002 + i}.txt"
        np.savetxt(filename, np.concatenate([obs,split_0_state],axis=0))

    with open(f'{save_dir}list_1.json','w') as file:
        json.dump(list(range(0, 1000, 2)),file)

    with open(f'{save_dir}list_2.json','w') as file:
        json.dump(list(range(1, 1001, 2)),file)

    for i in range(1,3):
        config = f"""
                load_data:
                    inputs: {data_dir}
                    prepare:
                        select:
                          timepoints:
                            - 0
                            - 1200
                        standardize: {{}}
                keep_list: {save_dir}list_{i}.json 
                train_hmm:
                    config_kwargs:
                        n_states: 2
                        learn_means: False
                        learn_covariances: True
                        learn_trans_prob: True
                        learning_rate: 0.005
                        n_epochs: 100
                        sequence_length: 100
                    init_kwargs:
                        n_init: 10
                        n_epochs: 2

            """
        output_dir = f'{save_dir}split_{i}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f'{output_dir}train_config.yaml', "w") as file:
            yaml.safe_dump(yaml.safe_load(config), file,default_flow_style=False)

        run_pipeline_from_file(f'{output_dir}train_config.yaml',output_dir)

        covs = np.load(f'{output_dir}/inf_params/covs.npy')
        npt.assert_almost_equal(np.sort(covs[:, 0, 1]), np.array([-cor[i-1], cor[i-1]]),decimal=2)

    # Remove the directory after testing
    rmtree(save_dir)








