"""Base class for models.

"""
import re
from abc import abstractmethod
from io import StringIO

import numpy as np
from tensorflow import Variable
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.data import Dataset
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tqdm.auto import tqdm as tqdm_auto
from tqdm.keras import TqdmCallback
from vrad.data import Data
from vrad.inference import initializers
from vrad.inference.callbacks import AnnealingCallback, SaveBestCallback
from vrad.inference.losses import KullbackLeiblerLoss, LogLikelihoodLoss
from vrad.inference.tf_ops import tensorboard_run_logdir
from vrad.utils.misc import check_iterable_type, replace_argument
from vrad.utils.model import HTMLTable, LatexTable


class BaseModel:
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
    rnn_type : str
        RNN to use, either 'lstm' or 'gru'.
    rnn_normalization : str
        Type of normalization to use in the inference network and generative model.
        Either 'layer', 'batch' or None.
    n_layers_inference : int
        Number of layers in the inference network.
    n_layers_model : int
        Number of layers in the generative model neural network.
    n_units_inference : int
        Number of units/neurons in the inference network.
    n_units_model : int
        Number of units/neurons in the generative model neural network.
    dropout_rate_inference : float
        Dropout rate in the inference network.
    dropout_rate_model : float
        Dropout rate in the generative model neural network.
    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta_t.
        Either 'layer', 'batch' or None.
    do_annealing : bool
        Should we use KL annealing during training?
    annealing_sharpness : float
        Parameter to control the annealing curve.
    n_epochs_annealing : int
        Number of epochs to perform annealing.
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
        rnn_type: str,
        rnn_normalization: str,
        n_layers_inference: int,
        n_layers_model: int,
        n_units_inference: int,
        n_units_model: int,
        dropout_rate_inference: float,
        dropout_rate_model: float,
        theta_normalization: str,
        do_annealing: bool,
        annealing_sharpness: float,
        n_epochs_annealing: int,
        learning_rate: float,
        multi_gpu: bool,
        strategy: str,
    ):
        # Validation
        if sequence_length < 1:
            raise ValueError("sequence_length must be greater than zero.")

        if rnn_type not in ["lstm", "gru"]:
            raise ValueError("rnn_type must be 'lstm' or 'gru'.")

        if n_layers_inference < 1 or n_layers_model < 1:
            raise ValueError("n_layers must be greater than zero.")

        if n_units_inference < 1 or n_units_model < 1:
            raise ValueError("n_units must be greater than zero.")

        if dropout_rate_inference < 0 or dropout_rate_model < 0:
            raise ValueError("dropout_rate must be greater than or equal to zero.")

        if rnn_normalization not in [
            "layer",
            "batch",
            None,
        ] or theta_normalization not in ["layer", "batch", None]:
            raise ValueError("normalization type must be 'layer', 'batch' or None.")

        if annealing_sharpness <= 0:
            raise ValueError("annealing_sharpness must be greater than zero.")

        if n_epochs_annealing < 0:
            raise ValueError(
                "n_epochs_annealing must be equal to or greater than zero."
            )

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        # Identifier for the model
        self._identifier = np.random.randint(100000)

        # Number of latent states and dimensionality of the data
        self.n_states = n_states
        self.n_channels = n_channels

        # Model hyperparameters
        self.sequence_length = sequence_length
        self.rnn_type = rnn_type
        self.rnn_normalization = rnn_normalization
        self.n_layers_inference = n_layers_inference
        self.n_layers_model = n_layers_model
        self.n_units_inference = n_units_inference
        self.n_units_model = n_units_model
        self.dropout_rate_inference = dropout_rate_inference
        self.dropout_rate_model = dropout_rate_model
        self.theta_normalization = theta_normalization

        # KL annealing
        self.do_annealing = do_annealing
        self.annealing_factor = Variable(0.0) if do_annealing else Variable(1.0)
        self.annealing_sharpness = annealing_sharpness
        self.n_epochs_annealing = n_epochs_annealing

        # Training Parameters
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

    def compile(self, optimizer=None):
        """Wrapper for the standard keras compile method.

        Sets up the optimiser and loss functions.

        Parameters
        ----------
        tensorflow.keras.optimizers
            Optimizer to use for training. Default is Adam.
        """
        # Setup optimizer
        if optimizer is None:
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # Loss functions
        ll_loss = LogLikelihoodLoss()
        kl_loss = KullbackLeiblerLoss(self.annealing_factor)
        loss = [ll_loss, kl_loss]

        # Compile
        self.model.compile(optimizer=optimizer, loss=loss)

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
        """ "Create callbacks for training.

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

    def fit(
        self,
        *args,
        no_annealing_callback=False,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

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
        history
            The training history.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                no_annealing_callback,
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

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

    def reset_model(self):
        """Reset the model as if you've built a new model.

        Resets the model weights, optimizer and annealing factor.
        """
        self.compile()
        initializers.reinitialize_model_weights(self.model)
        if self.do_annealing:
            self.annealing_factor.assign(0.0)

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
