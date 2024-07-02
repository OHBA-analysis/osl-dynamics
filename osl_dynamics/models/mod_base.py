"""Base class for models.

"""

import logging
import os
import re
from abc import abstractmethod
from dataclasses import dataclass
from io import StringIO
from contextlib import contextmanager

import yaml
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import optimizers
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tqdm.auto import tqdm as tqdm_auto
from tqdm.keras import TqdmCallback

import osl_dynamics
from osl_dynamics import data
import osl_dynamics.data.tf as dtf
from osl_dynamics.inference import callbacks, initializers
from osl_dynamics.utils.misc import NumpyLoader, get_argument, replace_argument
from osl_dynamics.utils.model import HTMLTable, LatexTable

_logger = logging.getLogger("osl-dynamics")


@dataclass
class BaseModelConfig:
    """Base class for settings for all models."""

    # Model choices
    model_name: str = None
    multiple_dynamics: bool = False

    # Training parameters
    batch_size: int = None
    learning_rate: float = None
    lr_decay: float = 0.1
    gradient_clip: float = None
    n_epochs: int = None
    optimizer: tf.keras.optimizers.Optimizer = "adam"
    loss_calc: str = "mean"
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

        if self.lr_decay < 0:
            raise ValueError("lr_decay must be non-negative.")

        if self.loss_calc not in ["mean", "sum"]:
            raise ValueError("loss_calc must be 'mean' or 'sum'.")

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

    osld_version = None
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
        optimizer : str or tf.keras.optimizers.Optimizer
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
            Arguments for :code:`keras.Model.fit()`.
        use_tqdm : bool, optional
            Should we use a :code:`tqdm` progress bar instead of the usual
            output from tensorflow.
        tqdm_class : TqdmCallback, optional
            Class for the :code:`tqdm` progress bar.
        save_best_after : int, optional
            Epoch number after which we should save the best model. The best
            model is that which achieves the lowest loss.
        save_filepath : str, optional
            Path to save the best model to.
        additional_callbacks : list, optional
            List of keras callback objects.
        kwargs : keyword arguments, optional
            Keyword arguments for :code:`keras.Model.fit()`.

        Returns
        -------
        history : history
            The training history.
        """
        # If a osl_dynamics.data.Data object has been passed for the x
        # arguments, replace it with a tensorflow dataset
        x = get_argument(self.model.fit, "x", args, kwargs)
        x = self.make_dataset(x, shuffle=True, concatenate=True, drop_last_batch=True)
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
            tqdm_class = tqdm_class or tqdm_auto
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
            args, kwargs = replace_argument(
                self.model.fit,
                "verbose",
                0,
                args,
                kwargs,
            )

        # Set the scaling factor for losses that are associated with static
        # quantities
        self.set_static_loss_scaling_factor(x)

        history = self.model.fit(*args, **kwargs)
        return history.history

    def load_weights(self, filepath):
        """Load weights of the model from a file.

        Parameters
        ----------
        filepath : str
            Path to file containing model weights.
        """
        with self.config.strategy.scope():
            return self.model.load_weights(filepath)

    def reset_weights(self, keep=None):
        """Resets trainable variables in the model to their initial value."""
        initializers.reinitialize_model_weights(self.model, keep=keep)

    def reset(self):
        """Reset the model as if you've built a new model."""
        self.reset_weights()
        self.compile()

    def make_dataset(
        self,
        inputs,
        shuffle=False,
        concatenate=False,
        step_size=None,
        drop_last_batch=False,
    ):
        """Converts a Data object into a TensorFlow Dataset.

        Parameters
        ----------
        inputs : osl_dynamics.data.Data or str or np.ndarray
            Data object. If a :code:`str` or :np.ndarray: is passed this
            function will first convert it into a Data object.
        shuffle : bool, optional
            Should we shuffle the data?
        concatenate : bool, optional
            Should we return a single TensorFlow Dataset or a list of Datasets.
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?

        Returns
        -------
        dataset : tf.data.Dataset or list
            TensorFlow Dataset (or list of Datasets) that can be used for
            training/evaluating.
        """
        if isinstance(inputs, str) or isinstance(inputs, np.ndarray):
            # str or numpy array -> Data object
            inputs = data.Data(inputs)

        if isinstance(inputs, data.Data):
            # Validation
            if (
                isinstance(self.config.strategy, MirroredStrategy)
                and not inputs.use_tfrecord
            ):
                _logger.warning(
                    "Using a multiple GPUs with a non-TFRecord dataset. "
                    + "This will result in poor performance. "
                    + "Consider using a TFRecord dataset with Data(..., use_tfrecord=True)."
                )

            # Data object -> list of Dataset if concatenate=False
            # or Data object -> Dataset if concatenate=True
            if inputs.use_tfrecord:
                outputs = inputs.tfrecord_dataset(
                    self.config.sequence_length,
                    self.config.batch_size,
                    shuffle=shuffle,
                    concatenate=concatenate,
                    step_size=step_size,
                    drop_last_batch=drop_last_batch,
                )
            else:
                outputs = inputs.dataset(
                    self.config.sequence_length,
                    self.config.batch_size,
                    shuffle=shuffle,
                    concatenate=concatenate,
                    step_size=step_size,
                    drop_last_batch=drop_last_batch,
                )

        elif isinstance(inputs, Dataset) and not concatenate:
            # Dataset -> list of Dataset if concatenate=False
            outputs = [inputs]

        else:
            outputs = inputs

        return outputs

    def get_training_time_series(
        self,
        training_data,
        prepared=True,
        concatenate=False,
    ):
        """Get the time series used for training from a Data object.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data
            Data object.
        prepared : bool, optional
            Should we return the prepared data? If not, we return the raw data.
        concatenate : bool, optional
            Should we concatenate the data for each session?

        Returns
        -------
        training_data : np.ndarray or list
            Training data time series.
        """
        return training_data.trim_time_series(
            self.config.sequence_length,
            prepared=prepared,
            concatenate=concatenate,
        )

    def summary_string(self):
        """Return a summary of the model as a string.

        This is a modified version of the :code:`keras.Model.summary()` method
        which makes the output easier to parse.
        """
        stringio = StringIO()
        self.model.summary(
            print_fn=lambda s: stringio.write(s + "\n"), line_length=1000
        )
        return stringio.getvalue()

    def summary_table(self, renderer):
        """Return a summary of the model as a table (HTML or LaTeX).

        Parameters
        ----------
        renderer : str
            Renderer to use. Either :code:`"html"` or :code:`"latex"`.

        Returns
        -------
        table : str
            Summary of the model as a table.
        """

        summary = self.summary_string()

        renderers = {"html": HTMLTable, "latex": LatexTable}

        # Extract information
        headers = [h for h in re.split(r"\s{2,}", summary.splitlines()[2]) if h != ""]
        columns = [summary.splitlines()[2].find(title) for title in headers] + [-1]

        # Create HTML table
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
        """Return a summary of the model as an HTML table."""
        return self.summary_table(renderer="html")

    def latex_summary(self):
        """Return a summary of the model as a LaTeX table."""
        return self.summary_table(renderer="latex")

    def _repr_html_(self):
        """Display the model as an HTML table in Jupyter notebooks.

        This is called when you type the variable name of the model in a
        Jupyter notebook. It is unlikely that you will need to call this.
        """
        return self.html_summary()

    def save_config(self, dirname):
        """Saves config object as a .yml file.

        Parameters
        ----------
        dirname : str
            Directory to save :code:`config.yml`.
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
            file.write(f"# osl-dynamics version: {osl_dynamics.__version__}\n")
            yaml.dump(config_dict, file)

    def save(self, dirname):
        """Saves config object and weights of the model.

        This is a wrapper for :code:`self.save_config` and
        :code:`self.save_weights`.

        Parameters
        ----------
        dirname : str
            Directory to save the :code:`config` object and weights of
            the model.
        """
        self.save_config(dirname)
        self.save_weights(
            f"{dirname}/weights"
        )  # will use the keras method: self.model.save_weights()

    def set_static_loss_scaling_factor(self, dataset):
        """Set the :code:`n_batches` attribute of the
        :code:`"static_loss_scaling_factor"` layer.

        Parameters
        ----------
        dataset : tf.data.Dataset
            TensorFlow dataset.

        Note
        ----
        This assumes every model has a layer called
        :code:`"static_loss_scaling_factor"`, with an attribure called
        :code:`"n_batches"`.
        """
        if "static_loss_scaling_factor" in [layer.name for layer in self.model.layers]:
            n_batches = dtf.get_n_batches(dataset)
            self.model.get_layer("static_loss_scaling_factor").n_batches = n_batches

    @contextmanager
    def set_trainable(self, layers, values):
        """Context manager to temporarily set the :code:`trainable`
        attribute of layers.

        Parameters
        ----------
        layers : str or list of str
            List of layers to set the :code:`trainable` attribute of.
        values : bool or list of bool
            Value to set the :code:`trainable` attribute of the layers to.
        """
        # Validation
        if isinstance(layers, str):
            layers = [layers]
        if isinstance(values, bool):
            values = [values] * len(layers)
        if len(layers) != len(values):
            raise ValueError(
                "layers and trainable must be the same length, "
                + f"but got {len(layers)} and {len(values)}."
            )

        available_layers = [layer.name for layer in self.layers]
        for layer in layers:
            if layer not in available_layers:
                raise ValueError(
                    f"No such layer: {layer}. "
                    + f"Available layers are: {available_layers}."
                )

        original_values = [self.get_layer(layer).trainable for layer in layers]

        try:
            for layer, trainable in zip(layers, values):
                self.get_layer(layer).trainable = trainable
            self.compile()
            yield
        finally:
            for layer, trainable in zip(layers, original_values):
                self.get_layer(layer).trainable = trainable
            self.compile()

    @staticmethod
    def load_config(dirname):
        """Load a :code:`config` object from a :code:`.yml` file.

        Parameters
        ----------
        dirname : str
            Directory to load :code:`config.yml` from.

        Returns
        -------
        config : dict
            Dictionary containing values used to create the :code:`config`
            object.
        version : str
            Version used to train the model.
        """

        # Load config dict
        with open(f"{dirname}/config.yml", "r") as file:
            config_dict = yaml.load(file, NumpyLoader)

        # Check what version the model was trained with
        version_comments = []
        with open(f"{dirname}/config.yml", "r") as file:
            for line in file.readlines():
                if "osl-dynamics version" in line:
                    version_comments.append(line)
        if len(version_comments) == 0:
            version = "<1.1.6"
        elif len(version_comments) == 1:
            version = version_comments[0].split(":")[-1].strip()
        else:
            raise ValueError(
                "version could not be read from config.yml. Make sure there "
                + "is only one comment containing the version in config.yml"
            )

        return config_dict, version

    @classmethod
    def load(cls, dirname, single_gpu=True):
        """Load model from :code:`dirname`.

        Parameters
        ----------
        dirname : str
            Directory where :code:`config.yml` and weights are stored.
        single_gpu : bool, optional
            Should we compile the model on a single GPU?

        Returns
        -------
        model : Model
            Model object.
        """
        _logger.info(f"Loading model: {dirname}")

        # Load config dict and version from yml file
        config_dict, version = cls.load_config(dirname)

        if single_gpu:
            config_dict["multi_gpu"] = False

        if version in ["<1.1.6", "1.1.6", "1.1.7"]:
            raise ValueError(
                f"model was trained using osl-dynamics version {version}. "
                + "This is incompatible with the current version. "
                + "Please revert to v1.1.7 to load this model."
            )

        # Create config object
        config = cls.config_type(**config_dict)

        # Create model
        model = cls(config)
        model.osld_version = version

        # Restore weights
        model.load_weights(f"{dirname}/weights").expect_partial()

        return model

    @property
    def is_multi_gpu(self):
        """Returns True if the model's strategy is MirroredStrategy."""
        return isinstance(self.config.strategy, MirroredStrategy)
