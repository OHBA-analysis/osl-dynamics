import numpy as np
import numpy.testing as npt

def test_CrossValidationSplit():
    from osl_dynamics.evaluate.cross_validation import CrossValidationSplit

    import os
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
                "n_splits": 5,
                "train_size": 0.8
            }
        },
        "split_column": {
            "n_samples": 4,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 2
            }
        },
        "strategy": "combination"
    }
    splitter_1 = CrossValidationSplit(**config_1)
    # Get the number of splits
    print("Total Splits:", splitter_1.get_n_splits())
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_1.split()):
        print(f"Split {i+1}:")
        print(f"  Row Train: {row_train}, Row Test: {row_test}")
        print(f"  Col Train: {col_train}, Col Test: {col_test}")
    print('###################################################')

    ### Case 2: row ShuffleSplit, column KFold, pairing
    config_2 = {
        "split_row": {
            "n_samples": 5,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 3,
                "train_size": 0.8
            }
        },
        "split_column": {
            "n_samples": 4,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 2
            }
        },
        "strategy": "pairing"
    }
    splitter_2 = CrossValidationSplit(**config_2)
    # Get the number of splits
    print("Total Splits:", splitter_2.get_n_splits())
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_2.split()):
        print(f"Split {i + 1}:")
        print(f"  Row Train: {row_train}, Row Test: {row_test}")
        print(f"  Col Train: {col_train}, Col Test: {col_test}")
    print('###################################################')

    ### Case 3: row KFold, column ShuffleSplit, combination
    config_3 = {
        "split_row": {
            "n_samples": 5,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 5,
            }
        },
        "split_column": {
            "n_samples": 4,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 2,
                "train_size": 0.5,
            }
        },
        "strategy": "combination"
    }
    splitter_3 = CrossValidationSplit(**config_3)
    # Get the number of splits
    print("Total Splits:", splitter_3.get_n_splits())
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_3.split()):
        print(f"Split {i + 1}:")
        print(f"  Row Train: {row_train}, Row Test: {row_test}")
        print(f"  Col Train: {col_train}, Col Test: {col_test}")
    print('###################################################')

    ### Case 4: row KFold, column ShuffleSplit, pairing
    config_4 = {
        "split_row": {
            "n_samples": 5,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 5,
            }
        },
        "split_column": {
            "n_samples": 4,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 2,
                "train_size": 0.5,
            }
        },
        "strategy": "pairing"
    }
    splitter_4 = CrossValidationSplit(**config_4)
    # Get the number of splits
    print("Total Splits:", splitter_4.get_n_splits())
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_4.split()):
        print(f"Split {i + 1}:")
        print(f"  Row Train: {row_train}, Row Test: {row_test}")
        print(f"  Col Train: {col_train}, Col Test: {col_test}")
    print('###################################################')

    ### We then test two cases for naive cross-validation split
    ### Case 5: row ShuffleSplit, column no-splitting
    config_5 = {
        "split_row": {
            "n_samples": 5,
            "method": "ShuffleSplit",
            "method_kwargs": {
                "n_splits": 3,
                "train_size": 0.8
            }
        },
    }
    splitter_5 = CrossValidationSplit(**config_5)
    # Get the number of splits
    print("Total Splits:", splitter_5.get_n_splits())
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_5.split()):
        print(f"Split {i + 1}:")
        print(f"  Row Train: {row_train}, Row Test: {row_test}")
        print(f"  Col Train: {col_train}, Col Test: {col_test}")
    print('###################################################')

    ### Case 6: row KFold, column no-splitting
    config_6 = {
        "split_row": {
            "n_samples": 5,
            "method": "KFold",
            "method_kwargs": {
                "n_splits": 5,
            }
        }
    }
    splitter_6 = CrossValidationSplit(**config_6)
    # Get the number of splits
    print("Total Splits:", splitter_6.get_n_splits())
    # Generate and display splits
    for i, (row_train, row_test, col_train, col_test) in enumerate(splitter_6.split()):
        print(f"Split {i + 1}:")
        print(f"  Row Train: {row_train}, Row Test: {row_test}")
        print(f"  Col Train: {col_train}, Col Test: {col_test}")
    print('###################################################')

    # Save splits to directory
    save_dir_1 = f'{save_dir}/case_1/'
    if not os.path.exists(save_dir_1):
        os.makedirs(save_dir_1)
    save_dir_5 = f'{save_dir}/case_5/'
    if not os.path.exists(save_dir_5):
        os.makedirs(save_dir_5)
    splitter_1.save(save_dir_1)
    splitter_5.save(save_dir_5)