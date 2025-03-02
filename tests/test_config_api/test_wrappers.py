import numpy as np
import numpy.testing as npt
from osl_dynamics.config_api.wrappers import load_data


def test_load_data():
    import os
    import json
    save_dir = './test_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vector = np.array([-1.5 ** 0.5, 0, 1.5 ** 0.5])
    input_1 = np.array([vector, vector + 10.0]).T
    input_2 = np.array([vector * 0.5 + 1., vector * 100]).T
    input_3 = np.zeros((3,2))
    np.savetxt(f'{save_dir}10001.txt', input_1)
    np.savetxt(f'{save_dir}10002.txt', input_2)
    np.savetxt(f'{save_dir}10003.txt',input_3)
    prepare = {'select':{'sessions':[0,1]},'standardize': {}}

    data = load_data(inputs=save_dir, prepare=prepare)
    npt.assert_equal(data.n_sessions,2)
    npt.assert_almost_equal(data.arrays[0], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.arrays[1], np.array([vector, vector]).T)
    data.delete_dir()
    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)

def test_train_model_hmm():
    import os
    import shutil
    from osl_dynamics.config_api.pipeline import run_pipeline_from_file

    save_dir = './test_train_model_hmm/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    select_sessions = [1, 2]
    select_channels = [0, 2]

    # Construct the data
    def generate_obs(cov, mean=None, n_timepoints=100):
        if mean is None:
            mean = np.zeros(len(cov))
        return np.random.multivariate_normal(mean, cov, n_timepoints)

    # Define the covariance matrices of state 1,2 in both splits
    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]

    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in range(0, 2):
        obs = []
        for j in range(1500):
            observations_Y = [generate_obs(covs_Y[i]), generate_obs(covs_Y[i + 1])]
            observations_X = [generate_obs([[vars_X[i]]], [means_X[i]]),
                              generate_obs([[vars_X[i + 1]]], [means_X[i + 1]])]
            observations = np.concatenate(
                [np.hstack((Y[:, :1], X, Y[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)

    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        sessions: {select_sessions}
                        channels: {select_channels}
                        timepoints:
                            - 0
                            - 300000
            train_model:
                model_type: hmm
                config_kwargs: 
                    n_states: {n_states}
                    learn_means: False
                    learn_covariances: True
                    learning_rate: 0.01
                    n_epochs: 3
                    sequence_length: 600
                init_kwargs:
                    n_init: 1
                    n_epochs: 1

            """

    with open(f'{save_dir}/config.yaml', "w") as file:
        file.write(config)
    run_pipeline_from_file(f'{save_dir}config.yaml',save_dir)


    result_means = np.load(f'{save_dir}/inf_params/means.npy')
    result_covs = np.load(f'{save_dir}/inf_params/covs.npy')
    npt.assert_array_equal(result_means, np.zeros((n_states, 2)))

    # Assert diagonal elements are all one
    npt.assert_allclose(np.diagonal(result_covs, axis1=-2, axis2=-1), 1.0, rtol=0.05, atol=0.05)

    # Assert off-diagonal elements are equal to cors
    off_diagonal = np.array([float(result_covs[i, 0, 1]) for i in range(n_states)])
    npt.assert_allclose(np.sort(off_diagonal), cors_Y, atol=0.05, rtol=0.05)