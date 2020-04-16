import tensorflow as tf
from tqdm.notebook import tqdm_notebook, tnrange
from tqdm import tqdm, trange


class Trainer:
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
        """

        Parameters
        ----------
        dataset
        n_batches
        n_epochs
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

    def grad(self, data):
        with tf.GradientTape() as tape:
            loss_value = self.loss(data)

        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def loss(self, inputs, training=True):
        log_likelihood_loss, kl_loss = self.model(inputs, training=training)[:2]

        loss_value = log_likelihood_loss + self.annealing_factor * kl_loss

        return loss_value

    def check_tqdm(self):
        try:
            for i in self.trange(1, leave=False):
                pass
        except ImportError:
            self.tqdm = tqdm
            self.trange = range
