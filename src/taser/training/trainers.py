"""Classes and methods for abstracting the training loop.

"""
import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from taser.helpers.misc import listify
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from taser.helpers.decorators import timing


class Callback(ABC):
    def __init__(self):
        pass

    def epoch_end(self, *args, **kwargs):
        pass


class SaveCallback(Callback):
    def __init__(self, trainer, basename, predict_dataset, frequency: int = 1):
        super().__init__()
        self.trainer = trainer
        self.basename = basename
        self.dataset = predict_dataset
        self.frequency = frequency

        Path(basename).mkdir(parents=True, exist_ok=True)

    def epoch_end(self):
        if self.trainer.epoch % self.frequency == 0:
            fn_length = len(str(self.trainer.n_epochs))
            np.save(
                f"{self.basename}/{str(self.trainer.epoch).zfill(fn_length)}",
                self.trainer.predict_latent_variable(self.dataset),
            )


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

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = Adam(lr=0.02, clipnorm=0.1) if optimizer is None else optimizer
        self.epoch = None
        self.n_epochs = None

        self.loss_value = tf.zeros(1)
        self.loss_history = [0.0]
        self.batch_mean = tf.keras.metrics.Mean()

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
        callbacks = listify(callbacks)

        self.n_epochs = n_epochs
        for self.epoch in range(n_epochs):
            self.train_epoch(dataset=dataset)
            for callback in callbacks:
                callback.epoch_end()

        del self.loss_history[0]

    def train_epoch(self, dataset: tf.data.Dataset):
        """Train a single epoch using input data.

        Parameters
        ----------
        dataset : tf.Dataset
        """
        epoch_iterator = tqdm(
            dataset, leave=False, postfix={"loss": self.loss_history[-1]},
        )

        for y in epoch_iterator:
            epoch_iterator.set_description_str(f"epoch={self.epoch}")
            loss_value, grads = self.grad(y)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.batch_mean(loss_value)
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

    # @timing
    def predict(self, dataset: tf.data.Dataset) -> List:
        results = []
        for y in dataset:
            results.append(self.model(y, training=False))

        if callable(getattr(self.model, "result_combination", None)):
            results = self.model.result_combination(results)

        return results

    def predict_latent_variable(self, dataset: tf.data.Dataset, **kwargs):
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
        plt.plot(self.loss_history[1:])
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
        annealing_sharpness: float,
        update_frequency: int = 10,
        optimizer: tf.keras.optimizers.Optimizer = None,
    ):
        super().__init__(model, optimizer)
        self.annealing_sharpness = annealing_sharpness
        self.update_frequency = update_frequency

        self.annealing_factor = None

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

    @timing
    def train(
        self, dataset: tf.data.Dataset, n_epochs: int, callbacks: List[Callback] = None
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
        callbacks = listify(callbacks)

        self.n_epochs = n_epochs
        for self.epoch in range(n_epochs):
            if self.epoch % self.update_frequency == 0:
                self.calculate_annealing_factor()
            self.train_epoch(dataset=dataset)
            for callback in callbacks:
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
        log_likelihood_loss, kl_loss = self.model(inputs, training=training)[:2]

        loss_value = log_likelihood_loss + self.annealing_factor * kl_loss

        return loss_value


class RepeatedAnnealer(AnnealingTrainer):
    def __init__(
        self,
        model: tf.keras.Model,
        annealing_sharpness: float,
        reset: int = None,
        update_frequency: int = None,
    ):
        super().__init__(model, annealing_sharpness, update_frequency)
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
