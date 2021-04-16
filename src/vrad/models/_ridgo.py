"""Model class for a generative model with Gaussian observations.

"""

import logging
from operator import lt
from typing import Tuple, Union

import numpy as np
from tensorflow import Variable
from tensorflow.keras import Model, layers, optimizers
from tqdm import trange
from vrad import models
from vrad.inference import callbacks, initializers, losses
from vrad.models.layers import (
    DirichletKLDivergenceLayer,
    InferenceRNNLayers,
    LogLikelihoodLayer,
    MeansCovsLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    NormalizationLayer,
    SampleDirichletDistributionLayer,
)
from vrad.utils.misc import check_arguments, replace_argument

_logger = logging.getLogger("VRAD")


class RIDGO(models.GO):
    """RNN Inference/model network, Dirichlet samples and Gaussian Observations (RIDGO).

    Parameters
    ----------
    config : vrad.models.Config
    """

    def __init__(self, config: models.Config):

        # KL annealing
        self.kl_annealing_factor = (
            Variable(0.0) if config.do_kl_annealing else Variable(1.0)
        )

        # Initialise the observation model
        # This will inherit the base model, build and compile the model
        super().__init__(config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def compile(self):
        """Wrapper for the standard keras compile method."""

        # Loss function
        ll_loss = losses.ModelOutputLoss()
        kl_loss = losses.ModelOutputLoss(self.kl_annealing_factor)
        loss = [ll_loss, kl_loss]

        # Compile
        self.model.compile(optimizer=self.config.optimizer, loss=loss)

    def fit(
        self,
        *args,
        kl_annealing_callback=True,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        kl_annealing_callback : bool
            Should we NOT update the annealing factor during training?
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : tqdm
            Class for the tqdm progress bar.
        use_tensorboard : bool
            Should we use TensorBoard?
        tensorboard_dir : str
            Path to the location to save the TensorBoard log files.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.

        Returns
        -------
        history
            The training history.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        additional_callbacks = []

        if kl_annealing_callback is None:
            kl_annealing_callback = self.config.do_kl_annealing

        if kl_annealing_callback:
            kl_annealing_callback = callbacks.KLAnnealingCallback(
                kl_annealing_factor=self.kl_annealing_factor,
                annealing_sharpness=self.config.kl_annealing_sharpness,
                n_epochs_annealing=self.config.n_epochs_kl_annealing,
            )
            additional_callbacks.append(kl_annealing_callback)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
                additional_callbacks,
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

    def reset_weight(self):
        """Reset the model as if you've built a new model.

        Resets the model weights, optimizer and annealing factor.
        """
        self.compile()
        initializers.reinitialize_model_weights(self.model)
        if self.config.do_kl_annealing:
            self.kl_annealing_factor.assign(0.0)

    def predict(self, *args, **kwargs) -> dict:
        """Wrapper for the standard keras predict method.

        Returns
        -------
        dict
            Dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, *kwargs)
        return_names = ["ll_loss", "kl_loss", "alpha"]
        predictions_dict = dict(zip(return_names, predictions))
        return predictions_dict

    def predict_states(
        self, inputs, *args, concatenate: bool = False, **kwargs
    ) -> Union[list, np.ndarray]:
        """State mixing factors, alpha.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.
        concatenate : bool
            Should we concatenate alpha for each subject? Optional, default
            is False.

        Returns
        -------
        np.ndarray
            State mixing factors with shape (n_subjects, n_samples, n_states) or
            (n_samples, n_states).
        """
        inputs = self._make_dataset(inputs)
        outputs = []
        for dataset in inputs:
            alpha = self.predict(dataset, *args, **kwargs)["alpha"]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)
        if len(outputs) == 1:
            outputs = outputs[0]
        elif concatenate:
            outputs = np.concatenate(outputs)
        return outputs

    def losses(self, dataset, return_mean: bool = False) -> Tuple[float, float]:
        """Calculates the log-likelihood and KL loss for a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate losses for.
        return_mean : bool
            Should we return the mean loss over batches? Otherwise we return
            the sum. Optional, default is False.

        Returns
        -------
        ll_loss : float
            Negative log-likelihood loss.
        kl_loss : float
            KL divergence loss.
        """
        if return_mean:
            mean_or_sum = np.mean
        else:
            mean_or_sum = np.sum
        if isinstance(dataset, list):
            predictions = [self.predict(subject) for subject in dataset]
            ll_loss = mean_or_sum([mean_or_sum(p["ll_loss"]) for p in predictions])
            kl_loss = mean_or_sum([mean_or_sum(p["kl_loss"]) for p in predictions])
        else:
            predictions = self.predict(dataset)
            ll_loss = mean_or_sum(predictions["ll_loss"])
            kl_loss = mean_or_sum(predictions["kl_loss"])
        return ll_loss, kl_loss

    def free_energy(self, dataset, return_mean: bool = False) -> float:
        """Calculates the variational free energy of a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate the variational free energy for.
        return_mean : bool
            Should we return the mean free energy over batches? Otherwise
            we return the sum. Optional, default is False.

        Returns
        -------
        float
            Variational free energy for the dataset.
        """
        ll_loss, kl_loss = self.losses(dataset, return_mean=return_mean)
        free_energy = ll_loss + kl_loss
        return free_energy

    def sample(self, n_samples: int) -> np.ndarray:
        """Uses the model RNN to sample a state time course.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.

        Returns
        -------
        np.ndarray
            Sampled state time course.
        """
        # Get layers
        samples_norm_layer = self.model.get_layer("samples_norm")
        rnn_layer = self.model.get_layer("mod_rnn")
        alpha_layer = self.model.get_layer("mod_alpha")

        # Activate the first state and sample from the model RNN
        stc = np.zeros([n_samples, self.config.n_states], dtype=np.float32)
        stc[0, 0] = 1
        for i in trange(1, n_samples, desc="Sampling state time course", ncols=98):
            samples = stc[np.newaxis, max(0, i - self.config.sequence_length) : i]
            samples_norm = samples_norm_layer(samples)
            model_rnn_output = rnn_layer(samples_norm)
            alpha = alpha_layer(model_rnn_output)[0, -1]
            stc[i] = np.random.dirichlet(alpha)

        return stc


def _model_structure(config: models.Config):

    # Layer for input
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )

    # Inference RNN:
    # - Learns q(samples) ~ Dir(samples | inference_alpha), where
    #     - inference_alpha = softplus(RNN(inputs_<=t))

    # Definition of layers
    inference_rnn_output_layers = InferenceRNNLayers(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout_rate,
        name="inf_rnn",
    )
    inference_alpha_layer = layers.Dense(
        config.n_states, activation=config.alpha_xform, name="inf_alpha"
    )
    samples_layer = SampleDirichletDistributionLayer(name="samples")

    # Data flow
    inference_rnn_output = inference_rnn_output_layers(inputs)
    inference_alpha = inference_alpha_layer(inference_rnn_output)
    samples = samples_layer(inference_alpha)

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each state as the observation model.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

    # Definition of layers
    means_covs_layer = MeansCovsLayer(
        config.n_states,
        config.n_channels,
        learn_means=False,
        learn_covariances=config.learn_covariances,
        normalize_covariances=config.normalize_covariances,
        initial_means=None,
        initial_covariances=config.initial_covariances,
        name="means_covs",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        config.n_states,
        config.n_channels,
        config.learn_alpha_scaling,
        name="mix_means_covs",
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    mu, D = means_covs_layer(inputs)  # inputs not used
    m, C = mix_means_covs_layer([samples, mu, D])
    ll_loss = ll_loss_layer([inputs, m, C])

    # Model RNN:
    # - Learns p(sample | samples_<t) ~ Dir(sample | model_alpha), where
    #     - model_alpha = softplus(RNN(samples_<t))

    # Definition of layers
    samples_norm_layer = NormalizationLayer(
        config.theta_normalization, name="samples_norm"
    )
    model_rnn_output_layers = ModelRNNLayers(
        config.model_rnn,
        config.model_normalization,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout_rate,
        name="mod_rnn",
    )
    model_alpha_layer = layers.Dense(
        config.n_states, activation=config.alpha_xform, name="mod_alpha"
    )
    kl_loss_layer = DirichletKLDivergenceLayer(name="kl")

    # Data flow
    samples_norm = samples_norm_layer(samples)
    model_rnn_output = model_rnn_output_layers(samples_norm)
    model_alpha = model_alpha_layer(model_rnn_output)
    kl_loss = kl_loss_layer([inference_alpha, model_alpha])

    return Model(inputs=inputs, outputs=[ll_loss, kl_loss, samples])
