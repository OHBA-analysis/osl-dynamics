"""Classes and methods for abstracting the training loop.

"""
import logging
from abc import ABC, abstractmethod
from typing import List, Union

import matplotlib.pyplot as plt
import tensorflow as tf
from taser.inference.callbacks import Callback
from taser.utils.decorators import timing
from taser.utils.misc import listify
from tensorflow.keras.optimizers import Adam
from tqdm import trange


class Trainer(ABC):
    """Trainer base class which all Trainers must subclass.

    Given a model and an optimizer, this class provides a series of methods to
    make training easier. The training loop becomes a method which can be called on the
    object from the main script. It's similar to the `.fit` and `.predict` methods
    which are part of `tf.keras.Model`, but they are more flexible.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be optimized.
    optimizer : tf.keras.optimizer.Optimizer
        Optimizer instance.

    Notes
    -----
    All classes which inherit from `Trainer` must implement the function `loss`.
    For anything other than the most basic training loops, the `train` function should
    also be overridden.


    """

    def __init__(self, model, optimizer=None, lr=0.02):
        self.model = model
        self.optimizer = Adam(lr=lr, clipnorm=0.1) if optimizer is None else optimizer
        self.epoch = None
        self.n_epochs = None

        self.loss_value = tf.zeros(1)
        self.loss_history = [0.0]
        self.batch_mean = tf.keras.metrics.Mean()

        self.prediction_dataset = None
        self.epoch_iterator = None
        self.dice = None
        self.callbacks = None
        self.post_fix = {}

    @timing
    def train(
        self, dataset: tf.data.Dataset, n_epochs: int, callbacks: List[Callback] = None
    ):
        """Vanilla custom training loop. No bells or whistles.

        A method to train a model. It contains the training loop for the optimization
        of the model. Override this method to add features to training.

        Parameters
        ----------
        dataset : tf.data.Dataset
            A batched dataset for the model to be trained on
        n_epochs : int
            The number of epochs to train for.
        """
        self.callbacks = listify(callbacks)

        self.n_epochs = n_epochs

        self.epoch_iterator = trange(n_epochs)

        for self.epoch in self.epoch_iterator:
            self.train_epoch(dataset=dataset)
            for callback in self.callbacks:
                callback.epoch_end()

        del self.loss_history[0]

    def train_epoch(self, dataset: tf.data.Dataset):
        """Train a single epoch using input data.

        Parameters
        ----------
        dataset : tf.Dataset
        """

        for i, y in enumerate(dataset):
            loss_value, grads = self.grad(y)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.batch_mean(loss_value)
            self.post_fix = {"it": i, "loss": self.loss_history[-1]}
            for callback in self.callbacks:
                callback.tqdm_update()
            self.epoch_iterator.set_postfix(self.post_fix)
        self.loss_history.append(self.batch_mean.result().numpy())
        self.batch_mean.reset_states()

    def grad(self, data: List[tf.Tensor]):
        """Calculate the gradient of operations.

        Parameters
        ----------
        data : tf.Tensor
            A batch of data to calculate the gradients of the model for.

        Returns
        -------
        loss_value : tf.Tensor
            Scalar Tensor containing the total loss of the model evaluated on `data`.
        gradient : List[tf.Tensor]
            The gradients returned by tf.GradientTape().gradient.

        """
        with tf.GradientTape() as tape:
            loss_value = self.loss(data)

        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    @abstractmethod
    def loss(self, *args, **kwargs):
        """Loss method for `Trainer` class. Must be overridden.

        Parameters
        ----------
        args
        kwargs
        """
        pass

    def predict(self, dataset: tf.data.Dataset) -> List:
        results = []
        for y in dataset:
            results.append(self.model(y, training=False))

        if callable(getattr(self.model, "result_combination", None)):
            results = self.model.result_combination(results)

        return results

    def predict_latent_variable(self, dataset: tf.data.Dataset, **kwargs):
        self.prediction_dataset = dataset
        if callable(getattr(self.model, "latent_variable", None)):
            results = self.predict(dataset=dataset)
            latent_variable = self.model.latent_variable(results_list=results, **kwargs)
            return latent_variable
        else:
            logging.warning(
                f"This instance of {self.__class__.__name__} does not have a "
                f"latent_variable method defined. "
            )

    def plot_loss(self):
        fig, axis = plt.subplots(1)
        axis.plot(self.loss_history[1:], color="k", lw=1)

        axis.set_title("Trainer loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")

        plt.setp([axis.spines["right"], axis.spines["top"]], visible=False)

        axis.ticklabel_format(
            axis="y", style="scientific", scilimits=(0, 0), useMathText=True
        )

        plt.show()


class AnnealingTrainer(Trainer):
    """A class for performing training on a model.

    Parameters
    ----------
    model : tf.keras.Model
    annealing_sharpness : float
        A parameter which controls the rate at which the weight of the KL loss is
        introduced to the total loss. Larger values are steeper tanh functions.
    optimizer : tf.keras.optimizers.Optimizer
    """

    def __init__(
        self,
        model: tf.keras.Model,
        annealing_sharpness: float = 20,
        update_frequency: int = 10,
        optimizer: tf.keras.optimizers.Optimizer = None,
        lr: float = None,
        annealing_scale: float = 1,
        burn_in: int = 0,
    ):
        super().__init__(model, optimizer, lr=lr)
        self.annealing_sharpness = annealing_sharpness
        self.annealing_scale = annealing_scale
        self.update_frequency = update_frequency

        self.annealing_factor = None
        self.burn_in = burn_in

    def check_burn_in(self):
        return self.epoch < self.burn_in

    def calculate_annealing_factor(self):
        """Calculate the weighting of the KL loss

        Calculate the weight of the KL loss in the total loss of the model by
        evaluating a tanh function at different epochs.
        """
        epoch = self.epoch
        n_epochs = self.n_epochs
        sharpness = self.annealing_sharpness

        self.annealing_factor = (
            0.5 * tf.math.tanh(sharpness * (epoch - n_epochs / 2.0) / n_epochs) + 0.5
        )

        self.annealing_factor *= self.annealing_scale

    @timing
    def train(
        self,
        dataset: tf.data.Dataset,
        n_epochs: int,
        callbacks: Union[List[Callback], Callback] = None,
    ):
        """Train the model.

        Override of `Trainer` method. Includes KL annealing step.

        Parameters
        ----------
        dataset : tf.data.Dataset
            A batched dataset for the model to be trained on
        n_epochs : int
            The number of epochs to train for.
        """
        self.callbacks = listify(callbacks)
        if self.burn_in > 0:
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.n_epochs = n_epochs
        self.epoch_iterator = trange(n_epochs)
        for self.epoch in self.epoch_iterator:
            if self.epoch % self.update_frequency == 0:
                self.calculate_annealing_factor()
            self.train_epoch(dataset=dataset)
            for callback in self.callbacks:
                callback.epoch_end()

    def loss(self, inputs: List[tf.Tensor], training: bool = True):
        """Calculate the loss of the model from the log likelihood loss and KL loss.

        Given the log_likelihood and KL losses, calculate LL + weight * KL

        Parameters
        ----------
        inputs : List[tf.Tensor]
            Batch of input data
        training : bool
            Training mode affects things like Dropout which is only active during
            training
        Returns
        -------
        loss_value : tf.Tensor
            A scalar Tensor containing the weighted sum of the log likelihood and KL
            divergence losses.

        """
        log_likelihood_loss, kl_loss = self.model(
            inputs, training=training, burn_in=self.check_burn_in()
        )[:2]

        loss_value = log_likelihood_loss + self.annealing_factor * kl_loss

        return loss_value


class RepeatedAnnealer(AnnealingTrainer):
    def __init__(
        self,
        model: tf.keras.Model,
        annealing_sharpness: float,
        reset: int = None,
        update_frequency: int = None,
        lr: float = None,
    ):
        super().__init__(model, annealing_sharpness, update_frequency, lr=lr)
        self.reset = reset

    def calculate_annealing_factor(self):
        """Calculate the weighting of the KL loss

        Calculate the weight of the KL loss in the total loss of the model by
        evaluating a tanh function at different epochs.
        """
        epoch = self.epoch
        n_epochs = self.n_epochs
        sharpness = self.annealing_sharpness
        reset = self.reset

        self.annealing_factor = (
            0.5
            * tf.math.tanh(
                sharpness
                * (epoch % reset - min(n_epochs, reset) / 2.0)
                / min(n_epochs, reset)
            )
            + 0.5
        )
