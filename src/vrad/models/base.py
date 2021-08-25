"""Base class for models.

"""

import pickle
import re
from abc import abstractmethod
from io import StringIO
from pathlib import Path

import numpy as np
import yaml
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.data import Dataset
from tqdm.auto import tqdm as tqdm_auto
from tqdm.keras import TqdmCallback
from vrad.data import Data
from vrad.inference import callbacks
from vrad.inference.tf_ops import tensorboard_run_logdir
from vrad.utils.misc import check_iterable_type, class_from_yaml
from vrad.utils.model import HTMLTable, LatexTable


class Base:
    """Base class for all models.

    Acts as a wrapper for a standard Keras model.

    Parameters
    ----------
    config : vrad.models.Config
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
        inputs : vrad.data.Data
            Data object.

        Returns
        -------
        tensorflow.data.Dataset
            Tensorflow dataset that can be used for training.
        """
        if isinstance(inputs, Data):
            return inputs.prediction_dataset(self.config.sequence_length)
        if isinstance(inputs, Dataset):
            return [inputs]
        if isinstance(inputs, str):
            return [Data(inputs).prediction_dataset(self.config.sequence_length)]
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 2:
                return [Data(inputs).prediction_dataset(self.config.sequence_length)]
            if inputs.ndim == 3:
                return [
                    Data(subject).prediction_dataset(self.config.sequence_length)
                    for subject in inputs
                ]
        if check_iterable_type(inputs, Dataset):
            return inputs
        if check_iterable_type(inputs, str):
            datasets = [
                Data(subject).prediction_dataset(self.config.sequence_length)
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

                # Create tqdm callback
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
        # Inferred state mixing factors and state time courses
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
