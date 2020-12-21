"""Base class for models.

"""
import re
from abc import abstractmethod
from io import StringIO

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.data import Dataset
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

from tqdm import trange
from tqdm.auto import tqdm as tqdm_auto
from tqdm.keras import TqdmCallback

from vrad.data import Data
from vrad.inference import initializers
from vrad.inference.callbacks import AnnealingCallback, SaveBestCallback
from vrad.utils.model import HTMLTable, LatexTable
from vrad.utils.misc import check_iterable_type


class Base:
    """Base class for models.

    Acts as a wrapper for a standard Keras model.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    learning_rate : float
        Learning rate for updating model parameters/weights.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        learning_rate: float,
        multi_gpu: bool,
        strategy: str,
    ):
        # Validation
        if sequence_length < 1:
            raise ValueError("sequence_length must be greater than zero.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        self._identifier = np.random.randint(100000)
        self.n_states = n_states
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        # Strategy for distributed learning
        self.strategy = strategy
        if multi_gpu:
            self.strategy = MirroredStrategy()
        elif strategy is None:
            self.strategy = get_strategy()

        # Build and compile the model
        self.model = None
        with self.strategy.scope():
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
            return inputs.prediction_dataset(self.sequence_length)
        if isinstance(inputs, Dataset):
            return [inputs]
        if isinstance(inputs, str):
            return [Data(inputs).prediction_dataset(self.sequence_length)]
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 2:
                return [Data(inputs).prediction_dataset(self.sequence_length)]
            if inputs.ndim == 3:
                return [
                    Data(subject).prediction_dataset(self.sequence_length)
                    for subject in inputs
                ]
        if check_iterable_type(inputs, Dataset):
            return inputs
        if check_iterable_type(inputs, str):
            datasets = [
                Data(subject).prediction_dataset(self.sequence_length)
                for subject in inputs
            ]
            return datasets

    def create_callbacks(
        self,
        no_annealing_callback: bool,
        use_tqdm: bool,
        tqdm_class,
        use_tensorboard: bool,
        tensorboard_dir: str,
        save_best_after: int,
        save_filepath: str,
    ):
        """Create callbacks for training.

        Parameters
        ----------
        no_annealing_callback : bool
            Should we NOT update the annealing factor during training?
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

        Returns
        -------
        list
            A list of callbacks to use during training.
        """
        additional_callbacks = []

        # Callback for KL annealing
        if not no_annealing_callback:
            annealing_callback = AnnealingCallback(
                annealing_factor=self.annealing_factor,
                annealing_sharpness=self.annealing_sharpness,
                n_epochs_annealing=self.n_epochs_annealing,
            )
            additional_callbacks.append(annealing_callback)

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
            save_best_callback = SaveBestCallback(
                save_best_after=save_best_after, filepath=save_filepath,
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
        with self.strategy.scope():
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
            if line.startswith("_") or line.startswith("=") or (":" in line):
                continue
            elements = [
                line[start:stop].strip() for start, stop in zip(columns, columns[1:])
            ]
            if elements[:3] == ["", "", ""]:
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
