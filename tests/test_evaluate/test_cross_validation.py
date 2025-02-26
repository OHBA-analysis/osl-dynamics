import pytest
import numpy as np
import numpy.testing as npt


def test_CrossValidationSplit():
    from osl_dynamics.evaluate.cross_validation import CrossValidationSplit

    import os
    import json
    import shutil

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