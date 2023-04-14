import numpy as np
import tensorflow as tf
import logging
from tqdm.auto import trange

_logger = logging.getLogger("osl-dynamics")


class FisherKernel:
    """Class for computing the Fisher kernel matrix given a generative model.

    Parameters
    ----------
    model : osl_dynamics.models.Model
        Model.
    """

    def __init__(self, model):
        compatible_models = ["HMM", "DyNeMo", "M-DyNeMo"]
        if model.config.model_name not in compatible_models:
            raise NotImplementedError(
                f"{model.config.model_name} was not found."
                + f"Options are {compatible_models}."
            )
        self.model = model

    def get_kernel_matrix(self, dataset, batch_size=None):
        """Get the Fisher kernel matrix.

        Parameters
        ----------
        dataset : osl_dynamics.data.Data
            Data.
        batch_size : int
            Batch size. If None, use the model's batch size.

        Returns
        -------
        kernel_matrix : np.ndarray
            Fisher kernel matrix. Shape is (n_subjects, n_subjects).
        """
        _logger.info("Getting Fisher kernel matrix")
        n_subjects = dataset.n_subjects
        if batch_size is not None:
            self.model.config.batch_size = batch_size
        dataset = self.model.make_dataset(dataset, concatenate=False, shuffle=False)

        # Initialise list to hold subject features
        features = []
        for i in trange(n_subjects, desc="Getting subject features"):
            subject_data = dataset[i]

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

            # Loop over data for each subject
            for data in subject_data:
                if self.model.config.model_name == "HMM":
                    x = data["data"]
                    gamma, xi = self.model._get_state_probs(x)
                    d_model["d_initial_distribution"].append(
                        self._d_initial_distribution(gamma)
                    )
                    d_model["d_trans_prob"].append(self._d_trans_prob(xi))

                    inputs = np.concatenate(
                        [x, gamma.reshape(x.shape[0], x.shape[1], -1)], 2
                    )
                else:
                    inputs = data

                gradients = self._get_tf_gradients(inputs)

                for name in d_model.keys():
                    if name == "d_initial_distribution" or name == "d_trans_prob":
                        continue
                    d_model[name].append(gradients[name])

            # Concatenate the flattened gradients to get the subject_features
            subject_features = np.concatenate(
                [np.sum(grad, axis=0).flatten() for grad in d_model.values()]
            )
            features.append(subject_features)

        features = np.array(features)  # shape=(n_subjects, n_features)

        # Normalise the features to l2-norm of 1
        features_l2_norm = np.sqrt(np.sum(np.square(features), axis=-1, keepdims=True))
        features /= features_l2_norm

        # Compute the kernel matrix with inner product
        kernel_matrix = features @ features.T
        return kernel_matrix

    def _d_trans_prob(self, xi):
        """Get the derivative of free energy with respect to
        the transition probability in HMM.

        Parameters
        ----------
        xi : np.ndarray
            Shape is (batch_size*sequence_length-1, n_states*n_states).

        Returns
        -------
        d_trans_prob : np.ndarray
            Derivative of free energy wrt the transition probability.
            Shape is (n_states, n_states).
        """
        trans_prob = self.model.trans_prob
        n_states = trans_prob.shape[0]
        # Reshape xi
        xi = np.reshape(xi, (xi.shape[0], n_states, n_states), order="F")
        # truncate at 1e-6 for numerical stability
        trans_prob = np.maximum(trans_prob, 1e-6)
        trans_prob /= np.sum(trans_prob, axis=-1, keepdims=True)

        d_trans_prob = np.sum(xi, axis=0) / trans_prob
        return d_trans_prob

    def _d_initial_distribution(self, gamma):
        """Get the derivative of free energy with respect to
        the initial distribution in HMM.

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data.
            Shape is (batch_size*sequence_length, n_states).

        Returns
        -------
        d_initial_distribution : np.ndarray
            Derivative of free energy wrt the initial distribution.
            Shape is (n_states,).
        """
        # truncate at 1e-6 for numerical stability
        initial_distribution = self.model.state_probs_t0
        initial_distribution = np.maximum(initial_distribution, 1e-6)
        initial_distribution /= np.sum(initial_distribution)

        d_initial_distribution = gamma[0] / initial_distribution
        return d_initial_distribution

    def _get_tf_gradients(self, inputs):
        """Get the gradient with respect to means and covariances.

        Parameters
        ----------
        inputs
            Input to the tensorflow models.

        Returns
        -------
        gradients : dict
            Gradients with respect to trainable variables.
        """
        with tf.GradientTape() as tape:
            loss = self.model.model(inputs)
            trainable_weights = {var.name: var for var in self.model.trainable_weights}
            gradients = tape.gradient(loss, trainable_weights)

        return gradients
