import os
import json
import shutil
import yaml
import pickle

import pytest
import numpy as np
import numpy.testing as npt

from osl_dynamics.evaluate.cross_validation import CrossValidationSplit, BiCrossValidation


def generate_obs(cov, mean=None, n_timepoints=100):
    if mean is None:
        mean = np.zeros(len(cov))
    return np.random.multivariate_normal(mean, cov, n_timepoints)
def test_CrossValidationSplit():

    save_dir = './tests/tmp_cross_validation_split/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    ### We test four cases for bi-cross-validation split
    ### Case 1: row ShuffleSplit, column KFold, combination
    config_1 = {
        "split_row": {
            "n_samples": 10,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 2,
                "train_size": 0.8,
                "random_state": 42,
            }
        },
        "split_column": {
            "n_samples": 4,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 2,
                "random_state": 42,
                "shuffle": True
            }
        },
        "strategy": "combination",
    }
    splitter_1 = CrossValidationSplit(**config_1)
    # Get the number of splits
    npt.assert_equal(splitter_1.get_n_splits(), 4, err_msg="Total splits do not match expected value!")

    answer_1 = [{'row_train': [5, 0, 7, 2, 9, 4, 3, 6], 'row_test': [8, 1],
                 'col_train': [0, 2], 'col_test': [1, 3]},
                {'row_train': [5, 0, 7, 2, 9, 4, 3, 6], 'row_test': [8, 1],
                 'col_train': [1, 3], 'col_test': [0, 2]},
                {'row_train': [8, 5, 3, 4, 7, 9, 6, 2], 'row_test': [0, 1],
                 'col_train': [0, 2], 'col_test': [1, 3]},
                {'row_train': [8, 5, 3, 4, 7, 9, 6, 2], 'row_test': [0, 1],
                 'col_train': [1, 3], 'col_test': [0, 2]}]
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_1.split()):
        npt.assert_equal(row_train, answer_1[i]['row_train'])
        npt.assert_equal(row_test, answer_1[i]['row_test'])
        npt.assert_equal(col_train, answer_1[i]['col_train'])
        npt.assert_equal(col_test, answer_1[i]['col_test'])

    ### Case 2: row ShuffleSplit, column KFold, pairing
    config_2 = {
        "split_row": {
            "n_samples": 5,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 5,
                "train_size": 0.8,
                "random_state": 42,
            }
        },
        "split_column": {
            "n_samples": 4,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 2,
                "random_state": 42,
                'shuffle': True
            }
        },
        "strategy": "pairing"
    }
    splitter_2 = CrossValidationSplit(**config_2)
    # Get the number of splits
    npt.assert_equal(splitter_2.get_n_splits(), 2, err_msg="Total splits do not match expected value!")

    answer_2 = [{'row_train': [4, 2, 0, 3], 'row_test': [1],
                 'col_train': [0, 2], 'col_test': [1, 3]},
                {'row_train': [1, 2, 0, 4], 'row_test': [3],
                 'col_train': [1, 3], 'col_test': [0, 2]}, ]
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_2.split()):
        npt.assert_equal(row_train, answer_2[i]['row_train'])
        npt.assert_equal(row_test, answer_2[i]['row_test'])
        npt.assert_equal(col_train, answer_2[i]['col_train'])
        npt.assert_equal(col_test, answer_2[i]['col_test'])

    ### Case 3: row KFold, column ShuffleSplit, combination
    config_3 = {
        "split_row": {
            "n_samples": 3,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 3,
                'random_state': 42,
                'shuffle': True
            }
        },
        "split_column": {
            "n_samples": 4,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 2,
                "train_size": 0.5,
                'random_state': 0,
            }
        },
        "strategy": "combination"
    }
    splitter_3 = CrossValidationSplit(**config_3)
    # Get the number of splits
    npt.assert_equal(splitter_3.get_n_splits(), 6, err_msg="Total splits do not match expected value!")

    answer_3 = [{'row_train': [1, 2], 'row_test': [0],
                 'col_train': [1, 0], 'col_test': [2, 3]},
                {'row_train': [1, 2], 'row_test': [0],
                 'col_train': [1, 3], 'col_test': [0, 2]},
                {'row_train': [0, 2], 'row_test': [1],
                 'col_train': [1, 0], 'col_test': [2, 3]},
                {'row_train': [0, 2], 'row_test': [1],
                 'col_train': [1, 3], 'col_test': [0, 2]},
                {'row_train': [0, 1], 'row_test': [2],
                 'col_train': [1, 0], 'col_test': [2, 3]},
                {'row_train': [0, 1], 'row_test': [2],
                 'col_train': [1, 3], 'col_test': [0, 2]}
                ]

    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_3.split()):
        npt.assert_equal(row_train, answer_3[i]['row_train'])
        npt.assert_equal(row_test, answer_3[i]['row_test'])
        npt.assert_equal(col_train, answer_3[i]['col_train'])
        npt.assert_equal(col_test, answer_3[i]['col_test'])

    ### Case 4: row KFold, column ShuffleSplit, pairing
    config_4 = {
        "split_row": {
            "n_samples": 5,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 5,
                'random_state': 42,
                'shuffle': True
            }
        },
        "split_column": {
            "n_samples": 4,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 2,
                "train_size": 0.5,
                'random_state': 0
            }
        },
        "strategy": "pairing"
    }
    splitter_4 = CrossValidationSplit(**config_4)
    # Get the number of splits
    npt.assert_equal(splitter_4.get_n_splits(), 2, err_msg="Total splits do not match expected value!")

    answer_4 = [{'row_train': [0, 2, 3, 4], 'row_test': [1],
                 'col_train': [1, 0], 'col_test': [2, 3]},
                {'row_train': [0, 1, 2, 3], 'row_test': [4],
                 'col_train': [1, 3], 'col_test': [0, 2]}, ]
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_4.split()):
        npt.assert_equal(row_train, answer_4[i]['row_train'])
        npt.assert_equal(row_test, answer_4[i]['row_test'])
        npt.assert_equal(col_train, answer_4[i]['col_train'])
        npt.assert_equal(col_test, answer_4[i]['col_test'])

    ### We then test two cases for naive cross-validation split
    ### Case 5: row ShuffleSplit, column no-splitting
    config_5 = {
        "split_row": {
            "n_samples": 5,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 3,
                "train_size": 0.8,
                'random_state': 42
            }
        },
    }
    splitter_5 = CrossValidationSplit(**config_5)
    # Get the number of splits
    npt.assert_equal(splitter_5.get_n_splits(), 3, err_msg="Total splits do not match expected value!")

    answer_5 = [{'row_train': [4, 2, 0, 3], 'row_test': [1],
                 'col_train': [], 'col_test': []},
                {'row_train': [1, 2, 0, 4], 'row_test': [3],
                 'col_train': [], 'col_test': []},
                {'row_train': [0, 3, 4, 2], 'row_test': [1],
                 'col_train': [], 'col_test': []}]

    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_5.split()):
        npt.assert_equal(row_train, answer_5[i]['row_train'])
        npt.assert_equal(row_test, answer_5[i]['row_test'])
        npt.assert_equal(col_train, answer_5[i]['col_train'])
        npt.assert_equal(col_test, answer_5[i]['col_test'])

    ### Case 6: row KFold, column no-splitting
    config_6 = {
        "split_row": {
            "n_samples": 5,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 5,
                'random_state': 42,
                'shuffle': True
            }
        }
    }
    splitter_6 = CrossValidationSplit(**config_6)
    # Get the number of splits
    npt.assert_equal(splitter_6.get_n_splits(), 5, err_msg="Total splits do not match expected value!")

    answer_6 = [{'row_train': [0, 2, 3, 4], 'row_test': [1],
                 'col_train': [], 'col_test': []},
                {'row_train': [0, 1, 2, 3], 'row_test': [4],
                 'col_train': [], 'col_test': []},
                {'row_train': [0, 1, 3, 4], 'row_test': [2],
                 'col_train': [], 'col_test': []},
                {'row_train': [1, 2, 3, 4], 'row_test': [0],
                 'col_train': [], 'col_test': []},
                {'row_train': [0, 1, 2, 4], 'row_test': [3],
                 'col_train': [], 'col_test': []}
                ]
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_6.split()):
        npt.assert_equal(row_train, answer_6[i]['row_train'])
        npt.assert_equal(row_test, answer_6[i]['row_test'])
        npt.assert_equal(col_train, answer_6[i]['col_train'])
        npt.assert_equal(col_test, answer_6[i]['col_test'])

    # Save splits to directory
    save_dir_2 = f'{save_dir}/case_2/'
    if not os.path.exists(save_dir_2):
        os.makedirs(save_dir_2)
    save_dir_5 = f'{save_dir}/case_5/'
    if not os.path.exists(save_dir_5):
        os.makedirs(save_dir_5)

    # Save these files
    splitter_2.save(save_dir_2)
    splitter_5.save(save_dir_5)

    answer_2_file = [{'row_train': [0, 2, 3, 4], 'row_test': [1],
                      'column_train': [0, 2], 'column_test': [1, 3]},
                     {'row_train': [0, 1, 2, 4], 'row_test': [3],
                      'column_train': [1, 3], 'column_test': [0, 2]}]

    answer_5_file = [{'row_train': [0, 2, 3, 4], 'row_test': [1],
                      'column_train': [], 'column_test': []},
                     {'row_train': [0, 1, 2, 4], 'row_test': [3],
                      'column_train': [], 'column_test': []},
                     {'row_train': [0, 2, 3, 4], 'row_test': [1],
                      'column_train': [], 'column_test': []}]

    for i in range(len(answer_2_file)):
        with open(f'{save_dir_2}fold_indices_{i+1}.json', "r") as f:
            fold = json.load(f)
        npt.assert_equal(fold['row_train'],answer_2_file[i]['row_train'])
        npt.assert_equal(fold['row_test'], answer_2_file[i]['row_test'])
        npt.assert_equal(fold['column_train'], answer_2_file[i]['column_train'])
        npt.assert_equal(fold['column_test'], answer_2_file[i]['column_test'])

    for i in range(len(answer_5_file)):
        with open(f'{save_dir_5}fold_indices_{i+1}.json', "r") as f:
            fold = json.load(f)
        npt.assert_equal(fold['row_train'],answer_5_file[i]['row_train'])
        npt.assert_equal(fold['row_test'], answer_5_file[i]['row_test'])
        npt.assert_equal(fold['column_train'], answer_5_file[i]['column_train'])
        npt.assert_equal(fold['column_test'], answer_5_file[i]['column_test'])

    ### Two edge cases where _validation_strategy throws an error
    config_7 = {
        "split_row": {
            "n_samples": 10,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 2,
                "train_size": 0.8,
                "random_state": 42,
            }
        },
        "split_column": {
            "n_samples": 4,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 2,
                "random_state": 42,
                "shuffle": True
            }
        },
        "strategy": "nonsense",
    }
    with pytest.raises(ValueError, match="Invalid strategy. Choose 'combination' or 'pairing'."):
        CrossValidationSplit(**config_7)

    config_8 = {
        "strategy": "pairing",
    }
    with pytest.raises(ValueError, match="At least one of split_row or split_column must be provided."):
        CrossValidationSplit(**config_8)

def test_BiCrossValidation_full_train_hmm():

    save_dir = './test_BiCrossValidation_full_train_hmm/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_states = 3
    row_train = [1, 2]
    row_test = [0]
    column_X = [1]
    column_Y = [0, 2]

    # Save the indices
    indices = {
        "row_train": row_train,
        "row_test": row_test,
        "column_X": column_X,
        "column_Y": column_Y
    }

    # Save to a JSON file
    indices_dir = f"{save_dir}/indices.json"
    with open(indices_dir, "w") as file:
        json.dump(indices, file, indent=4)

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
               model:
                   hmm:
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
               save_dir: {save_dir}
               indices: {indices_dir}
               mode: bcv_1
               """
    config = yaml.safe_load(config)

    bcv = BiCrossValidation(config)
    spatial_Y_train, temporal_Y_train = bcv.full_train(row_train, column_Y,save_dir)

    result_means = np.load(spatial_Y_train['means'])
    result_covs = np.load(spatial_Y_train['covariances'])
    npt.assert_array_equal(result_means, np.zeros((n_states, len(column_Y))))

    # Assert diagonal elements are all one
    npt.assert_allclose(np.diagonal(result_covs, axis1=-2, axis2=-1), 1.0, rtol=0.05, atol=0.05)

    # Assert off-diagonal elements are equal to cors
    off_diagonal = np.array([float(result_covs[i, 0, 1]) for i in range(n_states)])
    npt.assert_allclose(np.sort(off_diagonal), cors_Y, atol=0.05, rtol=0.05)


def test_BiCrossValidation_full_train_dynemo():
    import os
    import pickle
    import shutil
    import yaml
    from osl_dynamics.evaluate.cross_validation import BiCrossValidation

    save_dir = './test_BiCrossValidation_full_train_dynemo/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_modes = 2
    row_train = [1, 2]
    row_test = [0]
    column_X = [1]
    column_Y = [0, 2]

    # Save the indices
    indices = {
        "row_train": row_train,
        "row_test": row_test,
        "column_X": column_X,
        "column_Y": column_Y
    }

    # Save to a JSON file
    indices_dir = f"{save_dir}/indices.json"
    with open(indices_dir, "w") as file:
        json.dump(indices, file, indent=4)

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
            model:
                dynemo:
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
                        learn_means: False
                        initial_means: {means_truth_dir}
                        initial_covariances: {covs_truth_dir}
                        learning_rate: 0.001
                        model_n_units: 64
                        model_normalization: layer
                        n_epochs: 30
                        n_kl_annealing_epochs: 10
                        n_modes: {n_modes}
                        sequence_length: 100
                    init_kwargs:
                        n_init: 1
                        n_epochs: 1
            save_dir: {save_dir}
            indices: {indices_dir}
            mode: bcv_1
            """
    config = yaml.safe_load(config)

    bcv = BiCrossValidation(config)
    spatial_Y_train, temporal_Y_train = bcv.full_train(row_train, column_Y,save_dir)

    result_means = np.load(spatial_Y_train['means'])
    result_covs = np.load(spatial_Y_train['covariances'])
    with open(temporal_Y_train['alpha'], 'rb') as file:
        alpha = pickle.load(file)

    npt.assert_allclose(result_means, np.array(means_Y), rtol=1e-6, atol=1e-6)
    npt.assert_allclose(result_covs, np.stack(covs_Y), rtol=1e-2, atol=1e-2)

    # Test whether the inferred alphas are close to the ground truth
    for truth, inferred in zip(alpha_truth, alpha):
        mean_difference = np.mean(np.abs(truth - inferred))
        npt.assert_array_less(mean_difference, 5e-2, err_msg=f"Mean difference {mean_difference} exceeds 5e-2")

def test_BiCrossValidation_infer_temporal_hmm():

    save_dir = './test_BiCrossValidation_infer_temporal_hmm/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_states = 3
    row_train = [0]
    row_test = [1, 2]
    column_X = [0, 2]
    column_Y = [1]

    # Save the indices
    indices = {
        "row_train": row_train,
        "row_test": row_test,
        "column_X": column_X,
        "column_Y": column_Y
    }

    # Save to a JSON file
    indices_dir = f"{save_dir}/indices.json"
    with open(indices_dir, "w") as file:
        json.dump(indices, file, indent=4)

    # Define the covariance matrices of state 1,2 in both splits
    means_X = [np.array([-10.0, -10.0]), np.array([0.0, 0.0]), np.array([10.0, 10.0])]
    cors_X = [-0.5, 0.0, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]

    means_Y = [1.0, 2.0, 3.0]
    vars_Y = [0.5, 1.0, 2.0]

    np.save(f'{save_dir}/fixed_means.npy', np.array(means_X))
    np.save(f'{save_dir}/fixed_covs.npy', np.stack(covs_X))
    spatial_X_train = {'means': f'{save_dir}/fixed_means.npy', 'covariances': f'{save_dir}/fixed_covs.npy'}

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
            model:
                hmm:
                    config_kwargs:
                        n_states: {n_states}
                        learn_means: True
                        learn_covariances: True
                        learning_rate: 0.01
                        n_epochs: 10
                        sequence_length: 600
                    init_kwargs:
                        n_init: 1
                        n_epochs: 1
            save_dir: {save_dir}
            indices: {indices_dir}
            mode: bcv_1
            """
    config = yaml.safe_load(config)

    bcv = BiCrossValidation(config)
    temporal_X_test = bcv.infer_temporal(row_test, column_X, spatial_X_train)

    # Read the alpha
    with open(temporal_X_test['alpha'], 'rb') as file:
        alpha = pickle.load(file)

    for i in range(2):
        npt.assert_allclose(alpha[0], hidden_states[0], atol=1e-6)