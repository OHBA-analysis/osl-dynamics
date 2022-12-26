"""Base class for models.

"""

import os
import re
import pprint as pp
from abc import abstractmethod
from io import StringIO
from dataclasses import dataclass

import numpy as np
import yaml
import tensorflow
from tensorflow.data import Dataset
from tensorflow.keras import optimizers
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tqdm.auto import tqdm as tqdm_auto
from tqdm.keras import TqdmCallback

from osl_dynamics import data
from osl_dynamics.inference import callbacks, initializers
from osl_dynamics.utils.misc import get_argument, replace_argument, NumpyLoader
from osl_dynamics.utils.model import HTMLTable, LatexTable


@dataclass
class BaseModelConfig:
    """Base class for settings for all models."""

    # Model choices
    model_name: str = None
    multiple_dynamics: bool = False

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
    n_states: int = None
    n_channels: int = None
    sequence_length: int = None

    def validate_training_parameters(self):
        if self.batch_size is None:
            raise ValueError("batch_size must be passed.")
        elif self.batch_size < 1:
            raise ValueError("batch_size must be one or greater.")

        if self.n_epochs is None:
            raise ValueError("n_epochs must be passed.")
        elif self.n_epochs < 1:
            raise ValueError("n_epochs must be one or greater.")

        if self.learning_rate is None:
            raise ValueError("learning_rate must be passed.")
        elif self.learning_rate < 0:
            raise ValueError("learning_rate must be greater than zero.")

        # Strategy for distributed learning
        if self.multi_gpu:
            self.strategy = MirroredStrategy()
        elif self.strategy is None:
            self.strategy = get_strategy()

    def validate_dimension_parameters(self):
        if self.n_modes is None and self.n_states is None:
            raise ValueError("Either n_modes or n_states must be passed.")

        if self.n_modes is not None:
            if self.n_modes < 1:
                raise ValueError("n_modes must be one or greater.")

        if self.n_states is not None:
            if self.n_states < 1:
                raise ValueError("n_states must be one or greater.")

        if self.n_channels is None:
            raise ValueError("n_channels must be passed.")
        elif self.n_channels < 1:
            raise ValueError("n_channels must be one or greater.")

        if self.sequence_length is None:
            raise ValueError("sequence_length must be passed.")
        elif self.sequence_length < 1:
            raise ValueError("sequence_length must be one or greater.")


class ModelBase:
    """Base class for all models.

    Acts as a wrapper for a standard Keras model.
    """

    config_type = None

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

    def compile(self, optimizer=None):
        """Compile the model.

        Parameters
        ----------
        optimizer : str or tensorflow.keras.optimizers.Optimizer
            Optimizer to use when compiling.
        """
        if optimizer is None:
            optimizer = optimizers.get(
                {
                    "class_name": self.config.optimizer.lower(),
                    "config": {
                        "learning_rate": self.config.learning_rate,
                        "clipnorm": self.config.gradient_clip,
                    },
                }
            )
        self.model.compile(optimizer)

    def fit(
        self,
        *args,
        use_tqdm=False,
        tqdm_class=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        args : arguments
            Arguments for keras.Model.fit().
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : TqdmCallback
            Class for the tqdm progress bar.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.
        additional_callbacks : list
            List of keras callback objects.
        kwargs : keyword arguments
            Keyword arguments for keras.Model.fit()

        Returns
        -------
        history : history
            The training history.
        """
        # If a osl_dynamics.data.Data object has been passed for the x arguments,
        # replace it with a tensorflow dataset
        x = get_argument(self.model.fit, "x", args, kwargs)
        x = self.make_dataset(x, shuffle=True, concatenate=True)
        args, kwargs = replace_argument(self.model.fit, "x", x, args, kwargs)

        # Use the number of epochs in the config if it has not been passed
        if get_argument(self.model.fit, "epochs", args, kwargs) is None:
            args, kwargs = replace_argument(
                self.model.fit, "epochs", self.config.n_epochs, args, kwargs
            )

        # If we're saving the model after a particular epoch make sure it's
        # less than the total number of epochs
        if save_best_after is not None:
            epochs = get_argument(self.model.fit, "epochs", args, kwargs)
            if epochs < save_best_after:
                raise ValueError("save_best_after must be less than epochs.")

        # Callbacks to add to the ones the user passed
        additional_callbacks = []

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

        # Callback to save the best model after a certain number of epochs
        if save_best_after is not None:
            if save_filepath is None:
                save_filepath = f"/tmp/model_weights/best_{self._identifier}"
            save_best_callback = callbacks.SaveBestCallback(
                save_best_after=save_best_after,
                filepath=save_filepath,
            )
            additional_callbacks.append(save_best_callback)

        # Update arguments/keyword arguments to pass to the fit method
        args, kwargs = replace_argument(
            self.model.fit,
            "callbacks",
            additional_callbacks,
            args,
            kwargs,
            append=True,
        )
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        return self.model.fit(*args, **kwargs)

    def load_weights(self, filepath):
        """Load weights of the model from a file.

        Parameters
        ----------
        filepath : str
            Path to file containing model weights.
        """
        with self.config.strategy.scope():
            return self.model.load_weights(filepath)

    def reset_weights(self):
        """Resets trainable variables in the model to their initial value."""
        initializers.reinitialize_model_weights(self.model)

    def reset(self):
        """Reset the model as if you've built a new model."""
        self.reset_weights()
        self.compile()

    def make_dataset(self, inputs, shuffle=False, concatenate=False, subj_id=False):
        """Converts a Data object into a TensorFlow Dataset.

        Parameters
        ----------
        inputs : osl_dynamics.data.Data
            Data object. If a str or numpy array is passed this function will
            convert it into a Data object.
        shuffle : bool
            Should we shuffle the data?
        concatenate : bool
            Should we return a single TensorFlow Dataset or a list of Datasets.
        subj_id : bool
            Should we include the subject id in the dataset?

        Returns
        -------
        dataset : tensorflow.data.Dataset or list
            TensorFlow Dataset (or list of Datasets) that can be used for
            training/evaluating.
        """
        if isinstance(inputs, str) or isinstance(inputs, np.ndarray):
            # str or numpy array -> Data object
            inputs = data.Data(inputs)

        if isinstance(inputs, data.Data):
            # Data object -> list of Dataset if concatenate=False or
            # Data object -> Dataset if concatenate=True
            outputs = inputs.dataset(
                self.config.sequence_length,
                self.config.batch_size,
                shuffle=shuffle,
                concatenate=concatenate,
                subj_id=subj_id,
            )
        elif isinstance(inputs, Dataset) and not concatenate:
            # Dataset -> list of Dataset if concatenate=False
            outputs = [inputs]
        else:
            outputs = inputs

        return outputs

    def get_training_time_series(self, training_data, prepared=True, concatenate=False):
        """Get the time series used for training from a Data object.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data
            Data object.
        prepared : bool
            Should we return the prepared data? If not, we return the raw data.
        concatenate : bool
            Should we concatenate the data for each subject?

        Returns
        -------
        training_data : np.ndarray or list
            Training data time series.
        """
        return training_data.trim_time_series(
            self.config.sequence_length, prepared=prepared, concatenate=concatenate
        )

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

    def save_config(self, dirname):
        """Saves config object as a .yml file.

        Parameters
        ----------
        dirname : str
            Directory to save config.yml.
        """
        os.makedirs(dirname, exist_ok=True)

        config_dict = self.config.__dict__.copy()
        # for serialisability of the dict
        non_serializable_keys = [
            key for key in list(config_dict.keys()) if "regularizer" in key
        ]
        non_serializable_keys.append("strategy")
        for key in non_serializable_keys:
            config_dict[key] = None

        with open(f"{dirname}/config.yml", "w") as file:
            yaml.dump(config_dict, file)

    def save(self, dirname):
        """Saves config object and weights of the model.

        This is a wrapper for self.save_config and self.save_weights.

        Parameters
        ----------
        dirname : str
            Directory to save the config object and weights of the model.
        """
        self.save_config(dirname)
        self.save_weights(
            f"{dirname}/weights"
        )  # will use the keras method: self.model.save_weights()

    @classmethod
    def load(cls, dirname):
        """
        Load model from dirname.

        Parameters
        ----------
        dirname : str
            Directory where config.yml and weights are stored.

        Returns
        -------
        model
            Model object.
        """
        print("Loading model:", dirname)

        # Get the config
        with open(f"{dirname}/config.yml", "r") as f:
            config_dict = yaml.load(f, NumpyLoader)

        config = cls.config_type(**config_dict)
        model = cls(config)

        # Restore weights
        model.load_weights(f"{dirname}/weights").expect_partial()

        return model
