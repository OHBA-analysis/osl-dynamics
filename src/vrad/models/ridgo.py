"""Model class for a generative model with Gaussian observations.

"""


import numpy as np
from tensorflow.keras import Model, layers
from tqdm import trange
from vrad.models.inf_mod_base import InferenceModelBase
from vrad.models.go import GO
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
from vrad.utils.misc import replace_argument


class RIDGO(InferenceModelBase, GO):
    """RNN Inference/model network, Dirichlet samples and Gaussian Observations (RIDGO).

    Parameters
    ----------
    config : vrad.models.Config
    """

    def __init__(self, config):
        InferenceModelBase.__init__(self, config)
        GO.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

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


def _model_structure(config):

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
