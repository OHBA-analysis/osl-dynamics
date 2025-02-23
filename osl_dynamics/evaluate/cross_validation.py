import os
import json
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
from typing import Dict, Tuple, Generator, Optional, Any


class CrossValidationSplit:
    """
    A class to handle row and column splitting for (bi-)cross-validation.

    Parameters
    ----------
    split_row : dict, optional
        Configuration for row splitting. Should contain:
        - `n_samples` (int): Number of samples.
        - `method` (str): Cross-validation method (`"ShuffleSplit"` or `"KFold"`).
        - `method_kwargs` (dict): Additional arguments for the chosen method.
    split_column : dict, optional
        Configuration for column splitting. Same structure as `split_row`.
    strategy : str, optional
        Strategy for combining row and column splits:
        - `"combination"`: Generates all row-column split combinations.
        - `"pairing"`: Matches row splits to column splits in sequential order.
    kwargs : dict, optional
        Additional configurations for cross-validation (e.g., `random_state`).

    Raises
    ------
    ValueError
        If an invalid strategy is provided.
        If neither `split_row` nor `split_column` is provided.

    Notes
    -----
    This class supports two cross-validation strategies:
    1. **Combination**: Produces all possible (row, column) splits.
    2. **Pairing**: Matches row splits to column splits in sequential order.
    """

    def __init__(self, split_row: Optional[Dict[str, Any]] = None,
                 split_column: Optional[Dict[str, Any]] = None,
                 strategy: str = "combination", **kwargs):
        self.split_row = split_row
        self.split_column = split_column
        self.strategy = strategy
        self.kwargs = kwargs

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

        Raises
        ------
        ValueError
            If `split_config` does not contain `n_samples`.
            If the provided method is not `"ShuffleSplit"` or `"KFold"`.
        """
        if "n_samples" not in split_config:
            raise ValueError("split_config must include 'n_samples'.")

        n_samples = split_config["n_samples"]
        method = split_config.get("method", "ShuffleSplit")
        method_kwargs = split_config.get("method_kwargs", {})

        if method == "ShuffleSplit":
            splitter = ShuffleSplit(n_splits=method_kwargs.get("n_splits", 5),
                                    train_size=method_kwargs.get("train_size", 0.8),
                                    **self.kwargs)
        elif method == "KFold":
            splitter = KFold(n_splits=method_kwargs.get("n_splits", 5), **self.kwargs)
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

        for i, (row_train, row_test, col_train, col_test) in enumerate(self.split()):
            split_dict = {
                "row_train": sorted(row_train.tolist()) if row_train else [],
                "row_test": sorted(row_test.tolist()) if row_test else [],
                "column_train": sorted(col_train.tolist()) if col_train else [],
                "column_test": sorted(col_test.tolist()) if col_test else [],
            }
            with open(os.path.join(save_dir, f"fold_indices_{i + 1}.json"), "w") as f:
                json.dump(split_dict, f, indent=4)
