from typing import List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout

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

    """

    def __init__(
        self,
        dropout_rate: float,
        n_units_inference: int,
        n_units_model: int,
        n_states: int,
        n_channels: int,
        mus_initial: tf.Tensor,
        cholesky_djs_initial: tf.Tensor,
        learn_means: bool = False,
        learn_covariances: bool = True,
    ):
        super().__init__()

        self.dropout_layer = Dropout(dropout_rate)

        self.inference_rnn = Bidirectional(
            GRU(
                n_units_inference,
                return_sequences=True,
                stateful=False,
                name="inference_rnn",
            )
        )

        self.inference_dense_layer_mu = Dense(n_states, name="inference_mu")
        self.inference_dense_layer_sigma = Dense(n_states, name="inference_sigma")

        self.theta_ast = ReparameterizationLayer(name="theta_ast")

        self.model_rnn = GRU(n_units_model, return_sequences=True, name="model_rnn")

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
            n_states, n_channels, alpha_xform="softplus"
        )
        self.kl_loss = KLDivergenceLayer(n_states, n_channels)

    def call(
        self, inputs: List[tf.Tensor], training: bool = None, **kwargs
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
        dropout = self.dropout_layer(inputs)
        inference_rnn = self.inference_rnn(dropout)
        inference_mu = self.inference_dense_layer_mu(inference_rnn)
        inference_sigma = self.inference_dense_layer_sigma(inference_rnn)

        theta_ast = self.theta_ast([inference_mu, inference_sigma])

        model_rnn = self.model_rnn(theta_ast)
        model_mu = self.model_dense_layer_mu(model_rnn)

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
            model_mu,
            mus,
            djs,
        ]
