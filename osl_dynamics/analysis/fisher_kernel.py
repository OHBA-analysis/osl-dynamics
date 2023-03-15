import numpy as np
import tensorflow as tf
import logging
from tqdm import trange

_logger = logging.getLogger("osl-dynamics")


class FisherKernel:
    def __init__(self, model):
        compatible_models = ["HMM"]
        if model.config.model_name not in compatible_models:
            raise NotImplementedError(
                f"{model.config.model_name} was not found."
                + f"Options are {compatible_models}."
            )
        self.model = model

    def get_kernel_matrix(self, dataset, means, covariances):
        """Get the Fisher kernel matrix.

        Parameters
        ----------
        dataset : osl_dynamics.data.Data
            Data.
        means : np.ndarray
            Means for each subject.
            Shape is (n_subjects, n_modes, n_channels).
        covariances : np.ndarray
            Covariances for each subject.
            Shape is (n_subjects, n_modes, n_channels, n_channels).

        Returns
        -------
        kernel_matrix : np.ndarray
            Fisher kernel matrix. Shape is (n_subjects, n_subjects).
        """
        _logger.info("Getting Fisher kernel matrix")
        n_subjects = dataset.n_subjects
        dataset = self.model.make_dataset(dataset, concatenate=False, shuffle=False)

        # Initialise list to hold subject features
        features = []
        for i in trange(n_subjects, desc="Getting subject features"):
            subject_data = dataset[i]

            # Initialise list to hold gradients from different batches
            # Gradients specific to HMM
            d_initial_distribution, d_trans_prob = [], []

            # Gradients wrt the observation model parameters
            d_means, d_covariances = [], []

            for data in subject_data:
                x = data["data"]
                gamma, xi = self.model._get_state_probs(x)

                # Derivative wrt the temporal model parameters
                d_initial_distribution.append(self._d_initial_distribution(gamma))
                d_trans_prob.append(self._d_trans_prob(xi))

                # Get the gradient wrt the observation model parameters
                self.model.set_means(means[i])
                self.model.set_covariances(covariances[i])
                x_and_gamma = np.concatenate(
                    [x, gamma.reshape(x.shape[0], x.shape[1], -1)], 2
                )
                with tf.GradientTape() as tape:
                    loss = self.model.model(x_and_gamma)
                    trainable_weights = {
                        var.name: var for var in self.model.trainable_weights
                    }
                    gradients = tape.gradient(loss, trainable_weights)

                if self.model.config.learn_means:
                    d_means.append(gradients["means/means_kernel/tensor:0"])
                if self.model.config.learn_covariances:
                    d_covariances.append(gradients["covs/covs_kernel/tensor:0"])

            # Concatenate the flattened gradients to get the subject_features
            subject_features = np.concatenate(
                [
                    np.sum(d_initial_distribution, axis=0).flatten(),
                    np.sum(d_trans_prob, axis=0).flatten(),
                    np.sum(d_means, axis=0).flatten()
                    if self.model.config.learn_means
                    else [],
                    np.sum(d_covariances, axis=0).flatten()
                    if self.model.config.learn_covariances
                    else [],
                ]
            )
            features.append(subject_features)

        features = np.array(features)  # shape=(n_subjects, n_features)

        # Normalise the features to l2-norm of 1.
        features_l2_norm = np.sqrt(np.sum(np.square(features), axis=-1, keepdims=True))
        features /= features_l2_norm

        # Computet the kernel matrix with inner product
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

        d_initial_distribution = -gamma[0] / initial_distribution
        return d_initial_distribution
