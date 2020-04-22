"""Classes and methods for abstracting the training loop.

"""
from typing import List

import tensorflow as tf
from tqdm import tqdm
from tqdm.notebook import tnrange, tqdm_notebook

from abc import ABC, abstractmethod


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

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.epoch = None
        self.n_epochs = None

        self.tqdm = tqdm_notebook
        self.trange = tnrange

        self.loss_value = tf.zeros(1)
        self.loss_history = [0.0]
        self.batch_mean = tf.keras.metrics.Mean()

        self.check_tqdm()

    def train(self, dataset: tf.data.Dataset, n_epochs: int):
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
        self.n_epochs = n_epochs
        for self.epoch in self.trange(n_epochs):
            self.train_epoch(dataset=dataset)
        del self.loss_history[0]

    def train_epoch(self, dataset: tf.data.Dataset):
        """Train a single epoch using input data.

        Parameters
        ----------
        dataset : tf.Dataset
        """
        for y in self.tqdm(
            dataset,
            leave=False,
            postfix={"epoch": self.epoch, "loss": self.loss_history[-1]},
        ):
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

    def check_tqdm(self):
        """Check if tqdm_notebook throws an error and use CLI version if it does.

        """
        try:
            for i in self.trange(1, leave=False):
                pass
            print("tqdm notebook seems to be working.")
        except ImportError:
            print("Fallback to commandline version of tqdm.\nWarning can be ignored.")
            self.tqdm = tqdm
            self.trange = range


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
        optimizer: tf.keras.optimizers.Optimizer,
        annealing_sharpness: float,
    ):
        super().__init__(model, optimizer)
        self.annealing_sharpness = annealing_sharpness
        self.annealing_factor = None

    def calculate_annealing_factor(self):
        """Calculate the weighting of the KL loss

        Calculate the weight of the KL loss in the total loss of the model by
        evaluating a tanh function at different epochs.
        """
        self.annealing_factor = (
            0.5
            * tf.math.tanh(
                self.annealing_sharpness
                * (self.epoch - self.n_epochs / 2.0)
                / self.n_epochs
            )
            + 0.5
        )

    def train(self, dataset: tf.data.Dataset, n_epochs: int):
        """Train the model.

        Override of `Trainer` method. Includes KL annealing step.

        Parameters
        ----------
        dataset : tf.data.Dataset
            A batched dataset for the model to be trained on
        n_epochs : int
            The number of epochs to train for.
        """
        self.n_epochs = n_epochs
        for self.epoch in self.trange(n_epochs):
            if self.epoch % 10 == 0:
                self.calculate_annealing_factor()
            self.train_epoch(dataset=dataset)

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
