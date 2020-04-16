"""Classes and methods for abstracting the training loop.

"""
from typing import List

import tensorflow as tf
from tqdm import tqdm, trange
from tqdm.notebook import tnrange, tqdm_notebook


class Trainer:
    """A class for performing training on a model.

    Given a model and an optimizer, this class provides a series of methods to
    make training easier. The training loop becomes a method which can be called on the
    object from the main script. It's similar to the `.fit` and `.predict` methods
    which are part of `tf.keras.Model`, but they are more flexible.

    I intend to create an abstract base class to make altering training details easier.

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
        optimizer: tf.keras.optimizers.Optimizer,
    ):
        self.model = model
        self.annealing_sharpness = annealing_sharpness
        self.optimizer = optimizer
        self.epoch = None
        self.annealing_factor = None
        self.n_epochs = None

        self.tqdm = tqdm_notebook
        self.trange = tnrange

        self.check_tqdm()

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

    def train(self, dataset: tf.data.Dataset, n_batches: int, n_epochs: int = 1):
        """Train the model.

        A method to train a model. It contains the training loop for the optimization
        of the model.

        Parameters
        ----------
        dataset : tf.data.Dataset
            A batched dataset for the model to be trained on
        n_batches : int
            The number of batches. Only used for tqdm to estimate progress.
        n_epochs : int
            The number of epochs to train for.
        """
        loss_value = tf.zeros(1)
        self.n_epochs = n_epochs
        for self.epoch in self.trange(n_epochs):
            if self.epoch % 10 == 0:
                self.calculate_annealing_factor()
            for y in self.tqdm(
                dataset,
                total=n_batches,
                leave=False,
                postfix={"epoch": self.epoch, "loss": loss_value.numpy()},
            ):
                loss_value, grads = self.grad(y)
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

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

    def loss(self, inputs: List[tf.Tensor], training: bool = True):
        """Calculate the loss of the model from the log likelihood loss and KL loss.

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

    def check_tqdm(self):
        """Check if tqdm_notebook throws an error and use CLI version if it does.

        """
        try:
            for i in self.trange(1, leave=False):
                pass
            self.tqdm = tqdm_notebook
            self.trange = trange
        except ImportError:
            self.tqdm = tqdm
            self.trange = range
