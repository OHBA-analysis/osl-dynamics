from typing import List

import numpy as np
import tensorflow as tf
from taser.array_ops import get_one_hot
from taser.inference.layers import (
    MVNLayer,
    ReparameterizationLayer,
    TrainableVariablesLayer,
)
from taser.inference.loss import KLDivergenceLayer, LogLikelihoodLayer
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout


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
        mus_initial: np.ndarray = None,
        cholesky_djs_initial: np.ndarray = None,
        covariance_initial: np.ndarray = None,
        learn_means: bool = False,
        learn_covariances: bool = True,
        alpha_xform: str = "softmax",
        log_likelihood_diagonal_constant: float = 1e-8,
    ):
        super().__init__()

        if n_channels is None:
            raise ValueError("n_channels must be specified.")

        self.n_states = n_states

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
            num_gaussians=n_states,
            dim=n_channels,
            learn_means=learn_means,
            learn_covariances=learn_covariances,
            initial_means=mus_initial,
            initial_pseudo_sigmas=cholesky_djs_initial,
            initial_sigmas=covariance_initial,
        )

        self.log_likelihood_loss = LogLikelihoodLayer(
            n_states,
            n_channels,
            alpha_xform=alpha_xform,
            diagonal_constant=log_likelihood_diagonal_constant,
        )
        self.kl_loss = KLDivergenceLayer(n_states, n_channels)

    def call(
        self,
        inputs: List[tf.Tensor],
        training: bool = True,
        burn_in: bool = False,
        **kwargs
    ) -> List[tf.Tensor]:
        """Pass inputs to the model to get outputs.

        Parameters
        ----------
        **kwargs
        inputs : tf.Tensor
            The data to which to apply the model.
        training : bool
            If True, the model runs in training mode. Important for Dropout layers.
        burn_in : bool
            If True, the gradient will not be calculated for covariances in the MVN
            layer.

        Returns
        -------
        log_likelihood_loss : tf.Tensor
        kl_divergence_loss : tf.Tensor
        theta_ast : tf.Tensor
            Re-sampled distribution derived from the inference RNN.
        inference_mu : tf.Tensor
            Means from the inference RNN.
        model_mu : tf.Tensor
            Means from the model RNN.
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

        mus, djs = self.mvn_layer(inputs, burn_in=burn_in)

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
        """Apply transforms to results.

        When predicting, apply transforms to results such that they are easier to work
        with without needing to remember the necessary modifications. Unmodified
        results can still be accessed by calling the function over a dataset.

        Parameters
        ----------
        results : list
            The outputs from the model when run over a dataset.

        Returns
        -------
        formatted_results : dict
            A dictionary of results with transforms applied.

        """
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
        """Get the variable of most interest.

        For each model defined, the trainer object can call predict_latent_variable.
        This will return the variable deemed the most important. There is no reason
        that this cannot be a dict/list etc.

        Parameters
        ----------
        results_list : dict
            The list of predicted results from the result_combination method.
        one_hot : bool
            Return the variable after one-hot encoding.

        Returns
        -------

        """
        inference_mu = results_list["inference_mu"]
        if one_hot:
            return get_one_hot(inference_mu.argmax(axis=1), n_states=self.n_states)
        return inference_mu
