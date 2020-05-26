from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout

from taser.array_ops import get_one_hot
from taser.inference.layers import (
    MVNLayer,
    ReparameterizationLayer,
    TrainableVariablesLayer,
)
from taser.losses.layers import KLDivergenceLayer, LogLikelihoodLayer


class InferenceRNN(Model):
    """Model to infer a state state time course from multi-channel data.

    Given a multichannel input, this model is capable of learning a set of covariances
    and means corresponding to a series of states. From its outputs, a state time
    course can be determined.

    Parameters
    ----------
    dropout_rate : float
        Probability that a layer will have dropout applied. Must be between 0 and 1.
    n_units_inference : int
        Dimentionality of the output space for the inference `Dense` layers.
    n_units_model : int
        Dimentionality of the output space for the inference `Dense` layers.
    n_states : int
        Number of states to infer.
    n_channels : int
        Number of channels in the input data.
    mus_initial : tf.Tensor
        Initialisation values for the state means.
    cholesky_djs_initial : tf.Tensor
        Initialisation values for the Cholesky decomposition of the full covariances.
    learn_means : bool
        If True, learn the means of the states. If False, assume they are zero.
    learn_covariances : bool
        If True, learn the covariances of the states. If False,

    """

    def __init__(
        self,
        dropout_rate: float,
        n_units_inference: int,
        n_units_model: int,
        n_states: int,
        n_channels: int,
        mus_initial: tf.Tensor = None,
        cholesky_djs_initial: tf.Tensor = None,
        learn_means: bool = False,
        learn_covariances: bool = True,
        alpha_xform: str = "softmax",
        log_likelihood_diagonal_constant: float = 1e-8,
    ):
        super().__init__()

        if n_channels is None:
            raise ValueError("n_channels must be specified.")

        self.dropout_layer_0 = Dropout(dropout_rate)

        self.inference_rnn = Bidirectional(
            GRU(
                n_units_inference,
                return_sequences=True,
                stateful=False,
                name="inference_rnn",
            )
        )
        self.dropout_layer_1 = Dropout(dropout_rate)

        self.inference_dense_layer_mu = Dense(n_states, name="inference_mu")
        self.inference_dense_layer_sigma = Dense(n_states, name="inference_sigma")

        self.theta_ast = ReparameterizationLayer(name="theta_ast")

        self.dropout_layer_2 = Dropout(dropout_rate)
        self.model_rnn = GRU(n_units_model, return_sequences=True, name="model_rnn")
        self.dropout_layer_3 = Dropout(dropout_rate)

        self.model_dense_layer_mu = Dense(n_states, name="model_mu")
        self.log_sigma_theta_j = TrainableVariablesLayer(
            [n_states],
            initial_values=0.1 * tf.math.log(tf.ones(n_states)),
            name="log_sigma_theta_j",
        )

        self.mvn_layer = MVNLayer(
            n_states,
            n_channels,
            learn_means,
            learn_covariances,
            mus_initial,
            cholesky_djs_initial,
        )

        self.log_likelihood_loss = LogLikelihoodLayer(
            n_states,
            n_channels,
            alpha_xform=alpha_xform,
            diagonal_constant=log_likelihood_diagonal_constant,
        )
        self.kl_loss = KLDivergenceLayer(n_states, n_channels)

    def call(
        self, inputs: List[tf.Tensor], training: bool = True, **kwargs
    ) -> List[tf.Tensor]:
        """Pass inputs to the model to get outputs.

        Parameters
        ----------
        **kwargs
        inputs : tf.Tensor
        training : bool

        Returns
        -------
        log_likelihood_loss : tf.Tensor
        kl_divergence_loss : tf.Tensor
        theta_ast : tf.Tensor
            Re-sampled distribution derived from the inference RNN.
        inference_mu : tf.Tensor
        model_mu : tf.Tensor
        mus : tf.Tensor
            Means from the multivariate normal layer (`MVNLayer`)
        djs : tf.Tensor
            Covariances from the multivariate normal layer (`MVNLayer`)


        """
        dropout_0 = self.dropout_layer_0(inputs, training=training)
        inference_rnn = self.inference_rnn(dropout_0)
        dropout_1 = self.dropout_layer_1(inference_rnn, training=training)

        inference_mu = self.inference_dense_layer_mu(dropout_1)
        inference_sigma = self.inference_dense_layer_sigma(inference_rnn)

        theta_ast = self.theta_ast([inference_mu, inference_sigma])

        dropout_2 = self.dropout_layer_2(theta_ast, training=training)
        model_rnn = self.model_rnn(dropout_2)
        dropout_3 = self.dropout_layer_3(model_rnn, training=training)

        model_mu = self.model_dense_layer_mu(dropout_3)
        log_sigma_theta_j = self.log_sigma_theta_j(inputs)

        mus, djs = self.mvn_layer(inputs)

        log_likelihood_loss = self.log_likelihood_loss([inputs, theta_ast, mus, djs])
        kl_loss = self.kl_loss(
            [inference_mu, inference_sigma, model_mu, log_sigma_theta_j]
        )

        return [
            log_likelihood_loss,
            kl_loss,
            theta_ast,
            inference_mu,
            inference_sigma,
            model_mu,
            mus,
            djs,
        ]

    @staticmethod
    def result_combination(results: List) -> dict:

        results = np.array(results)

        names = [
            "log_likelihood_loss",
            "kl_loss",
            "theta_ast",
            "inference_mu",
            "inference_sigma",
            "model_mu",
            "mus",
            "djs",
        ]

        operations = {
            "log_likelihood_loss": lambda x: np.array(x),
            "kl_loss": lambda x: np.array(x),
            "theta_ast": lambda x: np.array(
                tf.math.softplus(
                    tf.reshape(
                        tf.concat(x.squeeze().tolist(), axis=0), [-1, x[0, 0].shape[-1]]
                    )
                )
            ),
            "inference_mu": lambda x: np.array(
                tf.reshape(
                    tf.math.softplus(tf.concat(x.squeeze().tolist(), axis=0)),
                    (-1, x[0, 0].shape[-1]),
                )
            ),
            "inference_sigma": lambda x: np.array(
                tf.reshape(
                    tf.concat(x.squeeze().tolist(), axis=0), [-1, x[0, 0].shape[-1]]
                )
            ),
            "model_mu": lambda x: tf.concat(x.squeeze().tolist(), axis=0),
            "mus": lambda x: x[0, 0],
            "djs": lambda x: x[0, 0],
        }

        combined_results = [
            operations["log_likelihood_loss"](results[:, 0::8]),
            operations["kl_loss"](results[:, 1::8]),
            operations["theta_ast"](results[:, 2::8]),
            operations["inference_mu"](results[:, 3::8]),
            operations["inference_sigma"](results[:, 4::8]),
            operations["model_mu"](results[:, 5::8]),
            operations["mus"](results[:, 6::8]),
            operations["djs"](results[:, 7::8]),
        ]

        return dict(zip(names, combined_results))

    def latent_variable(self, results_list: dict, one_hot: bool = True):
        inference_mu = results_list["inference_mu"]
        if one_hot:
            return get_one_hot(inference_mu.argmax(axis=1))
        return inference_mu


def get_default_config(n_states, n_channels):
    return {
        "dropout_rate": 0.3,
        "n_units_inference": 5 * n_states,
        "n_units_model": 5 * n_states,
        "n_states": n_states,
        "n_channels": n_channels,
        "learn_means": True,
    }
