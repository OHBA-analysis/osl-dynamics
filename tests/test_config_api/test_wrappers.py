import numpy as np
import numpy.testing as npt
from osl_dynamics.config_api.wrappers import load_data
from osl_dynamics.config_api.pipeline import run_pipeline_from_file


def generate_obs(cov, mean=None, n_timepoints=100):
    if mean is None:
        mean = np.zeros(len(cov))
    return np.random.multivariate_normal(mean, cov, n_timepoints)
def test_load_data():
    import os
    import json
    save_dir = './test_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vector = np.array([-1.5 ** 0.5, 0, 1.5 ** 0.5])
    input_1 = np.array([vector, vector + 10.0]).T
    input_2 = np.array([vector * 0.5 + 1., vector * 100]).T
    input_3 = np.zeros((3, 2))
    np.savetxt(f'{save_dir}10001.txt', input_1)
    np.savetxt(f'{save_dir}10002.txt', input_2)
    np.savetxt(f'{save_dir}10003.txt', input_3)
    prepare = {'select': {'sessions': [0, 1]}, 'standardize': {}}

    data = load_data(inputs=save_dir, prepare=prepare)
    npt.assert_equal(data.n_sessions, 2)
    npt.assert_almost_equal(data.arrays[0], np.array([vector, vector]).T)
    npt.assert_almost_equal(data.arrays[1], np.array([vector, vector]).T)
    data.delete_dir()
    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)

def test_train_model_hmm():
    import os
    import shutil

    save_dir = './test_train_model_hmm/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 2
    n_states = 3
    select_sessions = [1, 2]
    select_channels = [0, 2]

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
                    n_epochs: 30
                    sequence_length: 600
                init_kwargs:
                    n_init: 3
                    n_epochs: 1

            """

    with open(f'{save_dir}/config.yaml', "w") as file:
        file.write(config)
    run_pipeline_from_file(f'{save_dir}config.yaml', save_dir)

    result_means = np.load(f'{save_dir}/inf_params/means.npy')
    result_covs = np.load(f'{save_dir}/inf_params/covs.npy')
    npt.assert_array_equal(result_means, np.zeros((n_states,n_channels)))

    # Assert diagonal elements are all one
    npt.assert_allclose(np.diagonal(result_covs, axis1=-2, axis2=-1), 1.0, rtol=0.05, atol=0.05)

    # Assert off-diagonal elements are equal to cors
    off_diagonal = np.array([float(result_covs[i, 0, 1]) for i in range(n_states)])
    npt.assert_allclose(np.sort(off_diagonal), cors_Y, atol=0.05, rtol=0.05)


def test_train_model_dynemo():
    import os
    import pickle
    import shutil
    import yaml

    save_dir = './test_train_model_dynemo/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    select_sessions = [1, 2]
    select_channels = [0, 2]

    # Define the covariance matrices of state 1,2 in both splits
    cors_Y = [-0.5, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]

    means_Y = [[10.0, 0.0], [0.0, 10.0]]

    means_X = [1.0, 2.0]
    vars_X = [0.5, 2.0]

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    timepoints = 100  # Number of timepoints per segment
    alpha_truth = []

    for i in range(0, 2):
        obs = []
        alphas = []
        for j in range(3000):
            t = np.linspace(0, timepoints - 1, timepoints) / timepoints

            # Alpha coefficients as a staircase function
            alpha_t1 = np.ones_like(t)
            alpha_t1[len(t) // 2:] = 0  # Set second half to 0
            alpha_t2 = 1 - alpha_t1

            alphas.append(np.stack((alpha_t1, alpha_t2), axis=1))

            Y_mean_t = np.outer(alpha_t1, means_Y[0]) + np.outer(alpha_t2, means_Y[1])
            Y_cov_t = np.einsum('t,ij->tij', alpha_t1, covs_Y[0]) + np.einsum('t,ij->tij', alpha_t2, covs_Y[1])

            Y_obs = np.array(
                [np.random.multivariate_normal(Y_mean_t[t_idx], Y_cov_t[t_idx]) for t_idx in range(timepoints)])

            # Generate X observations
            X_mean_t = alpha_t1 * means_X[0] + alpha_t2 * means_X[1]
            X_var_t = alpha_t1 * vars_X[0] + alpha_t2 * vars_X[1]

            X_obs = np.reshape(np.random.normal(X_mean_t, np.sqrt(X_var_t)), (-1, 1))

            # Combine X and Y observations
            observations = np.hstack((Y_obs[:, :1], X_obs, Y_obs[:, 1:]))
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)
        alpha_truth.append(np.concatenate(alphas, axis=0))

    # Generate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))
    with open(f"{save_dir}alpha_truth.pkl", "wb") as f:
        pickle.dump(alpha_truth, f)
    np.save(f'{save_dir}/means_truth.npy', np.array(means_Y))
    np.save(f'{save_dir}/covs_truth.npy', np.stack(covs_Y))

    means_truth_dir = f'{save_dir}/means_truth.npy'
    covs_truth_dir = f'{save_dir}/covs_truth.npy'

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
                model_type: dynemo
                config_kwargs:
                    batch_size: 64
                    do_kl_annealing: true
                    inference_n_units: 64
                    inference_normalization: layer
                    initial_alpha_temperature: 1.0
                    kl_annealing_curve: tanh
                    kl_annealing_sharpness: 5
                    learn_alpha_temperature: true
                    learn_covariances: True
                    learn_means: False
                    initial_means: {means_truth_dir}
                    initial_covariances: {covs_truth_dir}
                    learning_rate: 0.001
                    model_n_units: 64
                    model_normalization: layer
                    n_channels: 2
                    n_epochs: 30
                    n_kl_annealing_epochs: 10
                    n_modes: 2
                    sequence_length: 100
                init_kwargs:
                    n_init: 1
                    n_epochs: 1
            """

    with open(f'{save_dir}/config.yaml', "w") as file:
        file.write(config)
    run_pipeline_from_file(f'{save_dir}config.yaml', save_dir)

    result_means = np.load(f'{save_dir}/inf_params/means.npy')
    result_covs = np.load(f'{save_dir}/inf_params/covs.npy')
    with open(f'{save_dir}/inf_params/alp.pkl', 'rb') as file:
        alpha = pickle.load(file)

    # We check the reconstruction of means and covariances.
    def reconstruct(alpha, mode_stat, stat_type):
        if stat_type == 'mean':
            return np.dot(alpha, mode_stat)
        elif stat_type == 'cov':
            return np.einsum('tm,mij->tij', alpha, mode_stat)
        else:
            raise ValueError('Incorrect statistics!')

    alpha_np = np.concatenate(alpha)
    alpha_truth_np = np.concatenate(alpha_truth)

    npt.assert_allclose(result_means, np.array(means_Y), rtol=1e-6, atol=1e-6)
    npt.assert_allclose(result_covs, np.stack(covs_Y), rtol=1e-2, atol=1e-2)

    # Test whether the inferred alphas are close to the ground truth
    mean_difference = np.mean(np.abs(alpha_np - alpha_truth_np))
    npt.assert_array_less(mean_difference, 1e-3, err_msg=f"Mean difference exceeds 1e-3")

def test_infer_spatial_hmm():
    import os
    import pickle
    import shutil
    import yaml

    save_dir = './test_infer_spatial_hmm/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    select_sessions = [1, 2]
    select_channels = [1]

    # Define the covariance matrices of state 1,2 in both splits
    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]

    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]

    means_X_estimate = [1.4, 2, 2.6]
    vars_X_estimate = [0.94, 1.5, 1.84]

    n_timepoints = 100

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    hidden_states = []

    for i in range(0, 2):
        # Build up the hidden variable
        hv_temp = np.zeros((n_timepoints * 2, n_states))
        hv_temp[:, i] = np.array([0.6] * n_timepoints + [0.4] * n_timepoints)
        hv_temp[:, i + 1] = np.array([0.4] * n_timepoints + [0.6] * n_timepoints)
        hidden_states.append(np.tile(hv_temp, (1500, 1)))

        obs = []
        for j in range(1500):
            observations_Y = [generate_obs(covs_Y[i], n_timepoints=n_timepoints),
                              generate_obs(covs_Y[i + 1], n_timepoints=n_timepoints)]
            observations_X = [generate_obs([[vars_X[i]]], [means_X[i]], n_timepoints=n_timepoints),
                              generate_obs([[vars_X[i + 1]]], [means_X[i + 1]], n_timepoints=n_timepoints)]
            observations = np.concatenate(
                [np.hstack((Y[:, :1], X, Y[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)

    alpha_file_path = f'{save_dir}/alp.pkl'
    with open(alpha_file_path, "wb") as file:
        pickle.dump(hidden_states, file)
    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        channels: {select_channels}
                        sessions: {select_sessions}
                        timepoints:
                            - 0
                            - 300000
            infer_spatial:
                model_type: hmm
                config_kwargs:
                    n_states: {n_states}
                    learn_means: True
                    learn_covariances: True
                    learning_rate: 0.01
                    n_epochs: 3
                    sequence_length: 600
                temporal_params:
                    alpha: {alpha_file_path}
            """

    with open(f'{save_dir}/config.yaml', "w") as file:
        file.write(config)
    run_pipeline_from_file(f'{save_dir}config.yaml', save_dir)

    result_means = np.load(f'{save_dir}/inf_params/means.npy')
    result_covs = np.load(f'{save_dir}/inf_params/covs.npy')
    npt.assert_allclose(means_X_estimate, result_means, rtol=1e-2, atol=1e-2)
    npt.assert_allclose(vars_X_estimate, result_covs, rtol=1e-2, atol=1e-2)

def test_infer_spatial_dynemo():
    import os
    import pickle
    import shutil
    import yaml

    save_dir = './test_infer_spatial_dynemo/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 2
    n_channels = 2
    n_modes = 2
    select_sessions = [1, 2]
    select_channels = [0, 2]


    # Define the covariance matrices of state 1,2 in both splits
    cors_X = [-0.5, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]
    means_X = [[1.0, -1.0], [-1.0, 1.0]]

    means_Y = [1.0, 2.0]
    vars_Y = [0.5, 2.0]

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    timepoints = 100  # Number of timepoints per segment
    alpha_truth = []

    for i in range(0, 2):
        obs = []
        alphas = []
        for j in range(3000):
            t = np.linspace(0, timepoints - 1, timepoints) / timepoints

            # Alpha coefficients
            alpha_t1 = np.sin(2 * np.pi * t) ** 2
            alpha_t2 = np.cos(2 * np.pi * t) ** 2

            alphas.append(np.stack((alpha_t1, alpha_t2), axis=1))

            X_mean_t = np.outer(alpha_t1, means_X[0]) + np.outer(alpha_t2, means_X[1])
            X_cov_t = np.einsum('t,ij->tij', alpha_t1, covs_X[0]) + np.einsum('t,ij->tij', alpha_t2, covs_X[1])

            X_obs = np.array(
                [np.random.multivariate_normal(X_mean_t[t_idx], X_cov_t[t_idx]) for t_idx in range(timepoints)])

            # Generate X observations
            Y_mean_t = alpha_t1 * means_Y[0] + alpha_t2 * means_Y[1]
            Y_var_t = alpha_t1 * vars_Y[0] + alpha_t2 * vars_Y[1]

            Y_obs = np.reshape(np.random.normal(Y_mean_t, np.sqrt(Y_var_t)), (-1, 1))

            # Combine X and Y observations
            observations = np.hstack((X_obs[:, :1], Y_obs, X_obs[:, 1:]))
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)
        alpha_truth.append(np.concatenate(alphas, axis=0))

    # Generate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    np.save(f'{save_dir}/fixed_means.npy', np.array(means_X))
    np.save(f'{save_dir}/fixed_covs.npy', np.stack(covs_X))

    alpha_file_path = f"{save_dir}/alp.pkl"
    with open(alpha_file_path, "wb") as f:
        pickle.dump(alpha_truth, f)

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        channels: {select_channels}
                        sessions: {select_sessions}
                        timepoints:
                            - 0
                            - 300000

            infer_spatial:
                model_type: dynemo
                config_kwargs:
                    batch_size: 64
                    do_kl_annealing: true
                    inference_n_units: 64
                    inference_normalization: layer
                    initial_alpha_temperature: 1.0
                    kl_annealing_curve: tanh
                    kl_annealing_sharpness: 5
                    learn_alpha_temperature: true
                    learn_covariances: true
                    learn_means: true
                    learning_rate: 0.01
                    model_n_units: 64
                    model_normalization: layer
                    n_channels: {n_channels}
                    n_epochs: 30
                    n_kl_annealing_epochs: 10
                    n_modes: {n_modes}
                    sequence_length: 150
                init_kwargs:
                    n_init: 10
                    n_epochs: 2
                temporal_params:
                    alpha: {alpha_file_path}
            """
    with open(f'{save_dir}/config.yaml', "w") as file:
        file.write(config)
    run_pipeline_from_file(f'{save_dir}config.yaml', save_dir)

    result_means = np.load(f'{save_dir}/inf_params/means.npy')
    result_covs = np.load(f'{save_dir}/inf_params/covs.npy')

    npt.assert_allclose(result_means, np.array(means_X), rtol=1e-2, atol=1e-2)
    npt.assert_allclose(result_covs, np.stack(covs_X), rtol=3e-2, atol=3e-2)
def test_infer_temporal_hmm():
    import os
    import shutil
    import yaml
    import pickle

    save_dir = './test_infer_temporal_hmm/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 2
    n_channels = 2
    n_states = 3
    select_sessions = [1, 2]
    select_channels = [0, 2]

    # Define the covariance matrices of state 1,2 in both splits
    means_X = [np.array([-10.0, -10.0]), np.array([0.0, 0.0]), np.array([10.0, 10.0])]
    cors_X = [-0.5, 0.0, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]

    means_Y = [1.0, 2.0, 3.0]
    vars_Y = [0.5, 1.0, 2.0]

    np.save(f'{save_dir}/fixed_means.npy', np.array(means_X))
    np.save(f'{save_dir}/fixed_covs.npy', np.stack(covs_X))
    spatial_means = f'{save_dir}/fixed_means.npy'
    spatial_covariances = f'{save_dir}/fixed_covs.npy'

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    hidden_states = []
    n_timepoints = 100

    for i in range(0, 2):

        # Build up the hidden variable
        hv_temp = np.zeros((n_timepoints * 2, n_states))
        hv_temp[:, i] = np.array([1.0] * n_timepoints + [0.0] * n_timepoints)
        hv_temp[:, i + 1] = np.array([0.0] * n_timepoints + [1.0] * n_timepoints)
        hidden_states.append(np.tile(hv_temp, (1500, 1)))

        obs = []
        for j in range(1500):
            observations_X = [generate_obs(covs_X[i], means_X[i], n_timepoints),
                              generate_obs(covs_X[i + 1], means_X[i + 1], n_timepoints)]
            observations_Y = [generate_obs([[vars_Y[i]]], [means_Y[i]], n_timepoints),
                              generate_obs([[vars_Y[i + 1]]], [means_Y[i + 1]]), n_timepoints]
            observations = np.concatenate(
                [np.hstack((X[:, :1], Y, X[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
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
                infer_temporal:
                    model_type: hmm
                    config_kwargs:
                        n_states: {n_states}
                        learn_means: True
                        learn_covariances: True
                        learning_rate: 0.01
                        n_epochs: 30
                        sequence_length: 600
                    init_kwargs:
                        n_init: 1
                        n_epochs: 1
                    spatial_params:
                        means: {spatial_means}
                        covariances: {spatial_covariances}
                """
    with open(f'{save_dir}/config.yaml', "w") as file:
        file.write(config)

    run_pipeline_from_file(f'{save_dir}/config.yaml', save_dir)
    # Read the alpha
    with open(f'{save_dir}/inf_params/alp.pkl', 'rb') as file:
        alpha = pickle.load(file)

    for i in range(2):
        npt.assert_allclose(alpha[i], hidden_states[i], atol=1e-6)

def test_infer_temporal_dynemo():
    import os
    import pickle
    import shutil
    import yaml

    save_dir = './test_infer_temporal_dynemo/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 2
    select_sessions = [1, 2]
    select_channels = [0, 2]

    # Define the covariance matrices of state 1,2 in both splits
    cors_X = [-0.5, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]
    means_X = [[1.0, -1.0], [-1.0, 1.0]]

    means_Y = [1.0, 2.0]
    vars_Y = [0.5, 2.0]

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    timepoints = 100  # Number of timepoints per segment
    alpha_truth = []

    for i in range(0, 2):
        obs = []
        alphas = []
        for j in range(3000):
            t = np.linspace(0, timepoints - 1, timepoints) / timepoints

            # Alpha coefficients
            alpha_t1 = np.sin(2 * np.pi * t) ** 2
            alpha_t2 = np.cos(2 * np.pi * t) ** 2

            alphas.append(np.stack((alpha_t1, alpha_t2), axis=1))

            X_mean_t = np.outer(alpha_t1, means_X[0]) + np.outer(alpha_t2, means_X[1])
            X_cov_t = np.einsum('t,ij->tij', alpha_t1, covs_X[0]) + np.einsum('t,ij->tij', alpha_t2, covs_X[1])

            X_obs = np.array(
                [np.random.multivariate_normal(X_mean_t[t_idx], X_cov_t[t_idx]) for t_idx in range(timepoints)])

            # Generate X observations
            Y_mean_t = alpha_t1 * means_Y[0] + alpha_t2 * means_Y[1]
            Y_var_t = alpha_t1 * vars_Y[0] + alpha_t2 * vars_Y[1]

            Y_obs = np.reshape(np.random.normal(Y_mean_t, np.sqrt(Y_var_t)), (-1, 1))

            # Combine X and Y observations
            observations = np.hstack((X_obs[:, :1], Y_obs, X_obs[:, 1:]))
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)
        alpha_truth.append(np.concatenate(alphas, axis=0))

    # Generate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    np.save(f'{save_dir}/fixed_means.npy', np.array(means_X))
    np.save(f'{save_dir}/fixed_covs.npy', np.stack(covs_X))
    with open(f"{save_dir}alpha_truth.pkl", "wb") as f:
        pickle.dump(alpha_truth, f)

    spatial_means = f'{save_dir}/fixed_means.npy'
    spatial_covariances = f'{save_dir}/fixed_covs.npy'

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
            infer_temporal:
                model_type: dynemo
                config_kwargs:
                    batch_size: 64
                    do_kl_annealing: true
                    inference_n_units: 64
                    inference_normalization: layer
                    initial_alpha_temperature: 1.0
                    kl_annealing_curve: tanh
                    kl_annealing_sharpness: 5
                    learn_alpha_temperature: true
                    learn_covariances: true
                    learn_means: true
                    learning_rate: 0.01
                    model_n_units: 64
                    model_normalization: layer
                    n_channels: 2
                    n_epochs: 30
                    n_kl_annealing_epochs: 10
                    n_modes: 2
                    sequence_length: 100
                init_kwargs:
                    n_init: 10
                    n_epochs: 2
                spatial_params:
                    means: {spatial_means}
                    covariances: {spatial_covariances}
            """
    with open(f'{save_dir}/config.yaml', "w") as file:
        file.write(config)

    run_pipeline_from_file(f'{save_dir}/config.yaml', save_dir)

    # Read the alpha
    with open(f'{save_dir}/inf_params/alp.pkl', 'rb') as file:
        alpha = pickle.load(file)

    # Test whether the inferred alphas are close to the ground truth
    for truth, inferred in zip(alpha_truth, alpha):
        mean_difference = np.mean(np.abs(truth - inferred))
        npt.assert_array_less(mean_difference, 5e-2, err_msg=f"Mean difference {mean_difference} exceeds 5e-2")

def test_calculate_log_likelihood_hmm():
    import os
    import json
    import shutil
    import yaml
    import pickle

    save_dir = './test_calculate_log_likelihood_hmm/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 2
    n_channels = 2
    n_states = 3
    select_sessions = [1, 2]
    select_channels = [0, 2]

    # Generate the data
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Build up subject data
    data_1 = np.zeros((2, 3))
    np.save(f'{data_dir}10001.npy', data_1)
    data_2 = np.array([[1., 0., 1., ], [-1., 0., 0.]])
    np.save(f'{data_dir}10002.npy', data_2)
    data_3 = np.array([[-1., 0., -1.], [1., 0., 0.]])
    np.save(f'{data_dir}10003.npy', data_3)

    def multivariate_gaussian_log_likelihood(x, mu, cov):
        """
        Calculate the log-likelihood for a multivariate Gaussian distribution.

        Parameters:
            x (ndarray): Observations (N, d), where N is the number of samples and d is the dimensionality.
            mu (ndarray): Mean vector of the distribution (d,).
            cov (ndarray): Covariance matrix of the distribution (d, d).

        Returns:
            float: Log-likelihood value.
        """
        # Dimensionality of the data
        d = len(mu)

        # Calculate the log determinant of the covariance matrix
        log_det_cov = np.log(np.linalg.det(cov))

        # Calculate the quadratic term in the exponent
        quad_term = np.sum((x - mu) @ np.linalg.inv(cov) * (x - mu), axis=1)

        # Calculate the log-likelihood
        log_likelihood = -0.5 * (d * np.log(2 * np.pi) + log_det_cov + quad_term)

        return log_likelihood

    means = np.zeros((3, 2))
    covs = np.array([[[1.0, 0.0], [0.0, 1.0]],
                     [[1.5, 0.8], [0.8, 1.5]],
                     [[0.5, -0.25], [-0.25, 0.5]]])
    spatial_means = f'{save_dir}/means.npy'
    spatial_covariances = f'{save_dir}/covs.npy'
    np.save(spatial_means, means)
    np.save(spatial_covariances, covs)

    # Set up the alpha.pkl
    alpha = [np.array([[1., 0., 0.], [0.0, 0.5, 0.5]]),
             np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])]
    alpha_file_path = f'{save_dir}/alp.pkl'
    with open(alpha_file_path, "wb") as file:
        pickle.dump(alpha, file)

    config = f"""
                load_data:
                    inputs: {data_dir}
                    prepare:
                        select:
                            sessions: {select_sessions}
                            channels: {select_channels}
                            timepoints:
                                - 0
                                - 2
                calculate_log_likelihood:
                    model_type: hmm
                    config_kwargs:
                        n_states: {n_states}
                        learn_means: False
                        learn_covariances: True
                        learning_rate: 0.01
                        n_epochs: 10
                        sequence_length: 600
                    init_kwargs:
                        n_init: 1
                        n_epochs: 1
                    spatial_params:
                        means: {spatial_means}
                        covariances: {spatial_covariances}
                    temporal_params:
                        alpha: {alpha_file_path}
                """
    with open(f'{save_dir}/config.yaml', "w") as file:
        file.write(config)

    run_pipeline_from_file(f'{save_dir}/config.yaml', save_dir)



    ll_1 = multivariate_gaussian_log_likelihood(data_2[:1, [0, 2]], np.array([0, 0]), covs[0])
    ll_2 = 0.5 * multivariate_gaussian_log_likelihood(data_2[1:2, [0, 2]], np.array([0, 0]), covs[1]) + \
           0.5 * multivariate_gaussian_log_likelihood(data_2[1:2, [0, 2]], np.array([0, 0]), covs[2])
    ll_3 = 0.5 * multivariate_gaussian_log_likelihood(data_3[:1, [0, 2]], np.array([0, 0]), covs[0]) + \
           0.5 * multivariate_gaussian_log_likelihood(data_3[:1, [0, 2]], np.array([0, 0]), covs[1])
    ll_4 = multivariate_gaussian_log_likelihood(data_3[1:2, [0, 2]], np.array([0, 0]), covs[2])

    ll = (ll_1 + ll_2 + ll_3 + ll_4) / 4 # average of each time point.
    with open(f'{save_dir}/metrics.json', 'r') as file:
        # Load the JSON data
        metrics = json.load(file)

    npt.assert_almost_equal(ll, metrics['log_likelihood'], decimal=3)