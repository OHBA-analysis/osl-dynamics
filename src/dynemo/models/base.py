"""Base class for models.

"""

import pickle
import re
from abc import abstractmethod
from io import StringIO
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import yaml
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.data import Dataset
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tqdm.auto import tqdm as tqdm_auto
from tqdm.keras import TqdmCallback
from dynemo.data import Data
from dynemo.inference import callbacks
from dynemo.inference.tf_ops import tensorboard_run_logdir
from dynemo.utils.misc import check_iterable_type, class_from_yaml
from dynemo.utils.model import HTMLTable, LatexTable


class Base:
    """Base class for all models.

    Acts as a wrapper for a standard Keras model.

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):
        self._identifier = np.random.randint(100000)
        self.config = config

        # Build and compile the model
        self.model = None
        with self.config.strategy.scope():
            self.build_model()
        self.compile()

    # Allow access to the keras model attributes
    def __getattr__(self, attr):
        return getattr(self.model, attr)

    @abstractmethod
    def build_model(self):
        """Build a keras model."""
        pass

    @abstractmethod
    def compile(self):
        """Wrapper for the standard keras compile method."""
        pass

    def _make_dataset(self, inputs: Data):
        """Make a dataset.

        Parameters
        ----------
        inputs : dynemo.data.Data
            Data object.

        Returns
        -------
        tensorflow.data.Dataset
            Tensorflow dataset that can be used for training.
        """
        if isinstance(inputs, Data):
            return inputs.dataset(self.config.sequence_length, shuffle=False)
        if isinstance(inputs, Dataset):
            return [inputs]
        if isinstance(inputs, str):
            return [Data(inputs).dataset(self.config.sequence_length, shuffle=False)]
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 2:
                return [
                    Data(inputs).dataset(self.config.sequence_length, shuffle=False)
                ]
            if inputs.ndim == 3:
                return [
                    Data(subject).dataset(self.config.sequence_length, shuffle=False)
                    for subject in inputs
                ]
        if check_iterable_type(inputs, Dataset):
            return inputs
        if check_iterable_type(inputs, str):
            datasets = [
                Data(subject).dataset(self.config.sequence_length, shuffle=False)
                for subject in inputs
            ]
            return datasets

    def create_callbacks(
        self,
        use_tqdm: bool,
        tqdm_class,
        use_tensorboard: bool,
        tensorboard_dir: str,
        save_best_after: int,
        save_filepath: str,
        additional_callbacks: list,
    ):
        """Create callbacks for training.

        Parameters
        ----------
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : tqdm
            Class for the tqdm progress bar.
        use_tensorboard : bool
            Should we use TensorBoard?
        tensorboard_dir : str
            Path to the location to save the TensorBoard log files.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.
        additional_callbacks : list
            Callbacks to include during training.

        Returns
        -------
        list
            A list of callbacks to use during training.
        """
        # Callback to display a progress bar with tqdm
        if use_tqdm:
            if tqdm_class is not None:
                tqdm_callback = TqdmCallback(verbose=0, tqdm_class=tqdm_class)

            else:
                # Create a tqdm class with a progress bar width of 98 characters
                class tqdm_class(tqdm_auto):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs, ncols=98)

                tqdm_callback = TqdmCallback(verbose=0, tqdm_class=tqdm_class)
            additional_callbacks.append(tqdm_callback)

        # Callback for Tensorboard visulisation
        if use_tensorboard:
            if tensorboard_dir is not None:
                tensorboard_cb = TensorBoard(
                    tensorboard_dir, histogram_freq=1, profile_batch="2,10"
                )
            else:
                tensorboard_cb = TensorBoard(
                    tensorboard_run_logdir(), histogram_freq=1, profile_batch="2,10"
                )
            additional_callbacks.append(tensorboard_cb)

        # Callback to save the best model after a certain number of epochs
        if save_best_after is not None:
            if save_filepath is None:
                save_filepath = f"/tmp/model_weights/best_{self._identifier}"
            save_best_callback = callbacks.SaveBestCallback(
                save_best_after=save_best_after,
                filepath=save_filepath,
            )
            additional_callbacks.append(save_best_callback)

        return additional_callbacks

    def load_weights(self, filepath: str):
        """Load weights of the model from a file.

        Parameters
        ----------
        str
            Path to file containing model weights.
        """
        with self.config.strategy.scope():
            self.model.load_weights(filepath)

    def summary_string(self):
        stringio = StringIO()
        self.model.summary(
            print_fn=lambda s: stringio.write(s + "\n"), line_length=1000
        )
        return stringio.getvalue()

    def summary_table(self, renderer):
        summary = self.summary_string()

        renderers = {"html": HTMLTable, "latex": LatexTable}

        # Extract information
        headers = [h for h in re.split(r"\s{2,}", summary.splitlines()[2]) if h != ""]
        columns = [summary.splitlines()[2].find(title) for title in headers] + [-1]

        # Create HTML table.
        table = renderers.get(renderer, HTMLTable)(headers)
        for line in summary.splitlines()[4:]:
            if (
                line.startswith("_")
                or line.startswith("=")
                or line.startswith('Model: "')
            ):
                continue
            elements = [
                line[start:stop].strip() for start, stop in zip(columns, columns[1:])
            ]
            if "params:" in elements[0]:
                parts = re.search(r"(.*? params): (.*?)$", elements[0]).groups()
                parts = [*parts, "", ""]
                table += parts
            elif elements[:3] == ["", "", ""]:
                table.append_last(elements[3])
            else:
                table += elements

        return table.output()

    def html_summary(self):
        return self.summary_table(renderer="html")

    def latex_summary(self):
        return self.summary_table(renderer="latex")

    def _repr_html_(self):
        return self.html_summary()

    def get_all_model_info(self, prediction_dataset, file=None):
        # Inferred mode mixing factors and mode time courses
        alpha = self.get_alpha(prediction_dataset)
        history = self.history.history

        info = dict(
            free_energy=self.free_energy(prediction_dataset),
            alpha=alpha,
            covs=self.get_covariances(),
            history=history,
        )

        if file:
            path = Path(file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(info, f)

        return info

    @classmethod
    def from_yaml(cls, file, **kwargs):
        return class_from_yaml(cls, file, kwargs)

    def fit_yaml(self, training_dataset, file, prediction_dataset=None):
        with open(file) as f:
            settings = yaml.load(f, Loader=yaml.Loader)

        results_file = settings.pop("results_file", None)

        history = self.fit(
            training_dataset,
            **settings,
        )

        save_filepath = settings.get("save_filepath", None)
        if save_filepath:
            self.load_weights(save_filepath)

        if results_file and prediction_dataset:
            try:
                self.get_all_model_info(prediction_dataset, results_file)
            except AttributeError:
                pass

        return history


@dataclass
class BaseConfig:
    """Base class for setting for any DyNeMo model."""

    # Training parameters
    batch_size: int = None
    learning_rate: float = None
    gradient_clip: float = None
    n_epochs: int = None
    optimizer: tensorflow.keras.optimizers.Optimizer = "adam"
    multi_gpu: bool = False
    strategy: str = None

    # Dimension parameters
    n_modes: int = None
    n_channels: int = None
    sequence_length: int = None

    def validate_training_parameters(self):
        if self.batch_size is None:
            raise ValueError("batch_size must be passed.")

        if self.batch_size < 1:
            raise ValueError("batch_size must be one or greater.")

        if self.n_epochs is None:
            raise ValueError("n_epochs must be passed.")

        if self.n_epochs < 1:
            raise ValueError("n_epochs must be one or greater.")

        if self.learning_rate is None:
            raise ValueError("learning_rate must be passed.")

        if self.learning_rate < 0:
            raise ValueError("learning_rate must be greater than zero.")

        # Strategy for distributed learning
        if self.multi_gpu:
            self.strategy = MirroredStrategy()
        elif self.strategy is None:
            self.strategy = get_strategy()

    def validate_dimension_parameters(self):
        if self.sequence_length is None:
            raise ValueError("sequence_length must be passed.")

        if self.n_modes is not None:
            if self.n_modes < 1:
                raise ValueError("n_modes must be one or greater.")

        if self.n_channels is not None:
            if self.n_channels < 1:
                raise ValueError("n_channels must be one or greater.")

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be one or greater.")
