"""Implementation of the Fisher kernel for prediction studies.

See the `HMM description <https://osl-dynamics.readthedocs.io/en/latest/models\
/hmm.html>`_ for further details.

See Also
--------
`Example script <https://github.com/OHBA-analysis/osl-dynamics/blob/main\
/examples/simulation/hmm_hmm-mvn_fisher-kernel.py>`_ applying the Fisher kernel
to simulated HMM data.
"""

import logging


import numpy as np
from tqdm.auto import trange

_logger = logging.getLogger("osl-dynamics")


class FisherKernel:
    """Class for computing the Fisher kernel matrix given a generative model.

    Parameters
    ----------
    model : osl-dynamics model
        Model. Currently only the :code:`HMM`, :code:`DyNeMo` and
        :code:`M-DyNeMo` are implemented.
    """

    def __init__(self, model):
        compatible_models = ["HMM", "DyNeMo", "M-DyNeMo"]
        if model.config.model_name not in compatible_models:
            raise NotImplementedError(
                f"{model.config.model_name} was not found."
                + f"Options are {compatible_models}."
            )

        self.model = model

    def get_features(self, dataset, batch_size=None):
        """Get the Fisher features.

        Parameters
        ----------
        dataset : osl_dynamics.data.Data
            Data.
        batch_size : int, optional
            Batch size. If :code:`None`, we use :code:`model.config.batch_size`.

        Returns
        -------
        features : np.ndarray
            Fisher kernel matrix. Shape is (n_sessions, n_features).
        """
        _logger.info("Getting Fisher features")

        n_sessions = dataset.n_sessions
        if batch_size is not None:
            self.model.config.batch_size = batch_size

        dataset = self.model.make_dataset(
            dataset,
            concatenate=False,
            shuffle=False,
        )

        # Initialise list to hold features for each session
        features = []
        for i in trange(n_sessions, desc="Getting features"):
            session_data = dataset[i]

            # Initialise dictionary for holding gradients
            d_model = dict()

            if self.model.config.model_name == "HMM":
                d_model["d_initial_distribution"] = []
                d_model["d_trans_prob"] = []

            trainable_variable_names = [
                var.name for var in self.model.trainable_weights
            ]
            # Get only variable in the generative model
            for name in trainable_variable_names:
                if (
                    "mod" in name
                    or "alpha" in name
                    or "gamma" in name
                    or "means" in name
                    or "covs" in name
                    or "stds" in name
                    or "fcs" in name
                ):
                    d_model[name] = []

            # Loop over data for each session
            for inputs in session_data:
                if self.model.config.model_name == "HMM":
                    x = inputs["data"]
                    gamma, xi = self.model.get_posterior(x)
                    d_initial_distribution, d_trans_prob = self._d_HMM(gamma, xi)
                    d_model["d_initial_distribution"].append(d_initial_distribution)
                    d_model["d_trans_prob"].append(d_trans_prob)
                    inputs = np.concatenate(
                        [x, gamma.reshape(x.shape[0], x.shape[1], -1)], axis=2
                    )

                gradients = self._get_tf_gradients(inputs)

                for name in d_model.keys():
                    if name == "d_initial_distribution" or name == "d_trans_prob":
                        continue
                    d_model[name].append(gradients[name])

            # Concatenate the flattened gradients
            session_features = np.concatenate(
                [np.sum(grad, axis=0).flatten() for grad in d_model.values()]
            )
            features.append(session_features)

        features = np.array(features)  # shape=(n_sessions, n_features)

        # Normalise the features to l2-norm of 1
        features_l2_norm = np.sqrt(np.sum(np.square(features), axis=-1, keepdims=True))
        features /= features_l2_norm
        return features

    def get_kernel_matrix(self, dataset, batch_size=None):
        """Get the Fisher kernel matrix.

        Parameters
        ----------
        dataset : osl_dynamics.data.Data
            Data.
        batch_size : int, optional
            Batch size. If :code:`None`, we use :code:`model.config.batch_size`.

        Returns
        -------
        kernel_matrix : np.ndarray
            Fisher kernel matrix. Shape is (n_sessions, n_sessions).
        """
        _logger.info("Getting Fisher kernel matrix")

        features = self.get_features(dataset, batch_size=batch_size)

        # Compute the kernel matrix with inner product
        kernel_matrix = features @ features.T

        return kernel_matrix

    def _d_HMM(self, gamma, xi):
        """Get the derivative of free energy with respect to
        transition probability, initial distribution of HMM.

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data.
            Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states given the data.
            Shape is (batch_size*sequence_length-1, n_states*n_states).

        Returns
        -------
        d_initial_distribution : np.ndarray
            Derivative of free energy with respect to the initial distribution.
            Shape is (n_states,).
        d_trans_prob : np.ndarray
            Derivative of free energy with respect to the transition
            probability. Shape is (n_states, n_states).
        """
        initial_distribution = self.model.state_probs_t0
        initial_distribution = np.maximum(initial_distribution, 1e-6)
        initial_distribution /= np.sum(initial_distribution)

        trans_prob = self.model.trans_prob
        n_states = trans_prob.shape[0]
        xi = np.reshape(xi, (xi.shape[0], n_states, n_states), order="F")

        trans_prob = np.maximum(trans_prob, 1e-6)
        trans_prob /= np.sum(trans_prob, axis=1, keepdims=True)

        d_initial_distribution = gamma[0] / initial_distribution
        d_trans_prob = np.sum(xi, axis=0) / trans_prob
        return d_initial_distribution, d_trans_prob

    def _get_tf_gradients(self, inputs):
        """Get the gradient with respect to means and covariances.

        Parameters
        ----------
        inputs : tf.data.Dataset
            Model inputs.

        Returns
        -------
        gradients : dict
            Gradients with respect to the trainable variables.
        """
        import tensorflow as tf  # avoid slow imports

        with tf.GradientTape() as tape:
            loss = self.model.model(inputs)
            trainable_weights = {var.name: var for var in self.model.trainable_weights}
            gradients = tape.gradient(loss, trainable_weights)

        return gradients
