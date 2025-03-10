import os
import json
import yaml
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
from typing import Dict, Tuple, Generator, Optional, Any
from osl_dynamics.config_api.pipeline import run_pipeline_from_file


class CrossValidationSplit:
    """
    A class to handle row and column splitting for (bi-)cross-validation.

    Parameters
    ----------
    split_row : dict, optional
        Configuration for row splitting.
    split_column : dict, optional
        Configuration for column splitting.
    strategy : str, optional
        Strategy for combining row and column splits:
        - `"combination"`: Generates all row-column split combinations.
        - `"pairing"`: Matches row splits to column splits in sequential order.
    """

    def __init__(self, split_row: Optional[Dict[str, Any]] = None,
                 split_column: Optional[Dict[str, Any]] = None,
                 strategy: str = "combination"):
        self.split_row = split_row
        self.split_column = split_column
        self.strategy = strategy

        self._validate_strategy()
        self.row_splitter = self._init_splitter(split_row) if split_row else None
        self.column_splitter = self._init_splitter(split_column) if split_column else None

    def _validate_strategy(self):
        """Validates the chosen strategy."""
        if self.strategy not in {"combination", "pairing"}:
            raise ValueError("Invalid strategy. Choose 'combination' or 'pairing'.")

        if not (self.split_row or self.split_column):
            raise ValueError("At least one of split_row or split_column must be provided.")

    def _init_splitter(self, split_config: Dict[str, Any]) -> Tuple[Any, int]:
        """
        Initialize a cross-validation splitter (ShuffleSplit or KFold).

        Parameters
        ----------
        split_config : dict
            Configuration for the splitter.

        Returns
        -------
        tuple
            Initialized splitter and the number of samples.
        """
        if "n_samples" not in split_config:
            raise ValueError("split_config must include 'n_samples'.")

        n_samples = split_config["n_samples"]
        method = split_config.get("method", "ShuffleSplit")
        method_kwargs = split_config.get("method_kwargs", {})

        # Include **self.kwargs to handle extra parameters
        if method == "ShuffleSplit":
            splitter = ShuffleSplit(**method_kwargs)
        elif method == "KFold":
            splitter = KFold(**method_kwargs)
        else:
            raise ValueError("Unsupported cross-validation method: use 'ShuffleSplit' or 'KFold'.")

        return splitter, n_samples

    def split(self) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate train-test splits for rows and columns.

        Yields
        ------
        tuple of np.ndarray
            Train and test indices for rows and columns.
        """
        row_splits = list(self._generate_splits(self.row_splitter)) if self.row_splitter else [([], [])]
        column_splits = list(self._generate_splits(self.column_splitter)) if self.column_splitter else [([], [])]

        if self.strategy == "combination":
            for row_train, row_test in row_splits:
                for col_train, col_test in column_splits:
                    yield row_train, row_test, col_train, col_test
        elif self.strategy == "pairing":
            n_splits = min(len(row_splits), len(column_splits))
            for i in range(n_splits):
                yield (*row_splits[i], *column_splits[i])

    def _generate_splits(self, splitter: Tuple[Any, int]) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Helper method to generate splits from a splitter object.

        Parameters
        ----------
        splitter : tuple
            The splitter instance and the number of samples.

        Returns
        -------
        generator of tuple of np.ndarray
            Train-test split indices.
        """
        return splitter[0].split(range(splitter[1]))

    def get_n_splits(self) -> int:
        """
        Get the total number of splits based on the strategy.

        Returns
        -------
        int
            The total number of cross-validation realizations.
        """
        row_splits = len(list(self._generate_splits(self.row_splitter))) if self.row_splitter else 1
        column_splits = len(list(self._generate_splits(self.column_splitter))) if self.column_splitter else 1

        return row_splits * column_splits if self.strategy == "combination" else min(row_splits, column_splits)

    def save(self, save_dir: str):
        """
        Save the splits as JSON files.

        Parameters
        ----------
        save_dir : str
            Directory to save the splits.

        Notes
        -----
        The splits are saved in JSON format with train-test indices.
        """
        os.makedirs(save_dir, exist_ok=True)

        def to_sorted_list(array):
            """Convert array to sorted list, handling empty arrays."""
            return sorted(array.tolist()) if len(array) > 0 else []

        for i, (row_train, row_test, col_train, col_test) in enumerate(self.split()):
            split_dict = {
                "row_train": to_sorted_list(row_train),
                "row_test": to_sorted_list(row_test),
                "column_train": to_sorted_list(col_train),
                "column_test": to_sorted_list(col_test),
            }
            with open(os.path.join(save_dir, f"fold_indices_{i + 1}.json"), "w") as f:
                json.dump(split_dict, f, indent=4)

class BiCrossValidation:
    """
    Implement bi-cross-validation for evaluating dynamic functional connectivity models.
    """

    def __init__(self, config, n_temp_save=3):
        """
        Initializes the BiCrossValidation class with configuration settings.

        Parameters
        ----------
        config : dict
            Dictionary containing model training details, file paths, and configuration settings.
        n_temp_save : int, optional
            Number of temporary realisations to preserve in bi-cross-validation.
        """

        # Create save directory if it doesn't exist
        self.save_dir = config['save_dir']
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.load_data = config['load_data']
        self.model, self.model_kwargs = next(iter(config['model'].items()))

        # Load indices
        with open(config['indices'], 'r') as file:
            indices = json.load(file)

        self.row_train = indices['row_train']
        self.row_test = indices['row_test']
        self.column_X = indices.get('column_X', None)
        self.column_Y = indices.get('column_Y', None)

        self.bcv_variant = str(config.get('cv_variant', 'fu_perry'))
        if self.bcv_variant not in ['fu_perry','owen_perry','smith','woolrich']:
            raise ValueError(f'Bi Cross Validation variant {self.bcv_variant} not unavailable!')

        # Determine if temporary results should be saved
        _, bcv_index = config['mode'].rsplit("_", 1)
        self.save_temp = int(bcv_index) <= n_temp_save

    def _prepare_load_data_config(self,row,column,save_dir):
        load_data_config = self.load_data
        load_data_config.setdefault('kwargs', {})['store_dir'] = f'{save_dir}/tmp/'
        load_data_config.setdefault('prepare', {}).setdefault('select', {})['sessions'] = row
        load_data_config.setdefault('prepare', {}).setdefault('select', {})['channels'] = column
        return load_data_config

    def full_train(self, row, column, save_dir=None):
        """Full training of the model"""
        from osl_dynamics.config_api.wrappers import train_model

         # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, 'full_train/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Prepare the config
        config = {}
        config['model_type'] = self.model
        config['data'] = self._prepare_load_data_config(row,column,save_dir)
        config['output_dir'] = save_dir

        # Add keys only if they exist in self.model_kwargs
        for key in ["config_kwargs", "init_kwargs", "fit_kwargs"]:
            if key in self.model_kwargs:
                config[key] = self.model_kwargs[key]

        with open(f'{save_dir}/config.yaml', 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)
        return train_model(**config)

    def infer_temporal(self,row,column,spatial_params,save_dir=None):
        from osl_dynamics.config_api.wrappers import infer_temporal
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, 'infer_temporal/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Prepare the config
        config = {}
        config['model_type'] = self.model
        config['data'] = self._prepare_load_data_config(row, column, save_dir)
        config['output_dir'] = save_dir

        # Add the spatial parameters
        config['spatial_params'] = spatial_params

        # Add keys only if they exist in self.model_kwargs
        for key in ["config_kwargs", "init_kwargs", "fit_kwargs"]:
            if key in self.model_kwargs:
                config[key] = self.model_kwargs[key]

        with open(f'{save_dir}/config.yaml', 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)
        return infer_temporal(**config)

    def validate(self):
        if self.bcv_variant == 'fu_perry':
            spatial_Y_train, temporal_Y_train = self.full_train(self.row_train, self.column_Y,
                                                                save_dir=os.path.join(self.save_dir, 'Y_train/'))
            spatial_X_train = self.infer_spatial(self.row_train, self.column_X, temporal_Y_train,
                                                 save_dir=os.path.join(self.save_dir, 'X_train/'))
            temporal_X_test = self.infer_temporal(self.row_test, self.column_X, spatial_X_train,
                                                  save_dir=os.path.join(self.save_dir, 'X_test/'))
            metric = self.calculate_error(self.row_test, self.column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(self.save_dir, 'Y_test/'))
        elif self.bcv_variant == 'owen_perry':
            spatial_X_train, temporal_X_train = self.full_train(self.row_train, self.column_X,
                                                                save_dir=os.path.join(self.save_dir, 'X_train/'))
            spatial_Y_train = self.infer_spatial(self.row_train, self.column_Y, temporal_X_train,
                                                 save_dir=os.path.join(self.save_dir, 'Y_train/'))
            temporal_X_test = self.infer_temporal(self.row_test, self.column_X, spatial_X_train,
                                                  save_dir=os.path.join(self.save_dir, 'X_test/'))
            metric = self.calculate_error(self.row_test, self.column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(self.save_dir, 'Y_test/'))
        elif self.bcv_variant == 'smith':
            spatial_XY_train, _ = self.full_train(self.row_train, sorted(self.column_X + self.column_Y),
                                                  save_dir=os.path.join(self.save_dir, 'XY_train/'))
            spatial_X_train, spatial_Y_train = self.split_column(self.column_X, self.column_Y, spatial_XY_train,
                                                                 save_dir=[os.path.join(self.save_dir, 'X_train/'),
                                                                           os.path.join(self.save_dir, 'Y_train/')]
                                                                 )
            temporal_X_test = self.infer_temporal(self.row_test, self.column_X, spatial_X_train,
                                                  save_dir=os.path.join(self.save_dir, 'X_test/'))
            metric = self.calculate_error(self.row_test, self.column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(self.save_dir, 'Y_test/'))
        elif self.bcv_variant == 'woolrich':
            _, temporal_X_traintest = self.full_train(sorted(self.row_train + self.row_test), self.column_X,
                                                      save_dir=os.path.join(self.save_dir, 'X_traintest/'))
            temporal_X_train, temporal_X_test = self.split_row(self.row_train, self.row_test, temporal_X_traintest,
                                                               save_dir=[os.path.join(self.save_dir, 'X_train/'),
                                                                         os.path.join(self.save_dir, 'X_test/')])
            spatial_Y_train = self.infer_spatial(self.row_train, self.column_Y, temporal_X_train,
                                                 save_dir=os.path.join(self.save_dir, 'Y_train/'))
            metric = self.calculate_error(self.row_test, self.column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(self.save_dir, 'Y_test/'))
        return metric


class NaiveCrossValidation:
    def __init__(self):
        pass