import numpy as np
from tensorflow.keras import Model, layers, optimizers
from tensorflow.python import Variable, zeros
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from vrad.inference.callbacks import AnnealingCallback, BurninCallback
from vrad.inference.layers import (
    InferenceRNNLayers,
    KLDivergenceLayer,
    LogLikelihoodLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    MultivariateNormalLayer,
    ReparameterizationLayer,
    TrainableVariablesLayer,
)
from vrad.utils.misc import listify


def create_model(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    learn_means: bool,
    learn_covariances: bool,
    initial_mean: np.ndarray = None,
    initial_covariances: np.ndarray = None,
    n_layers_inference: int = 1,
    n_layers_model: int = 1,
    n_units_inference: int = 64,
    n_units_model: int = 64,
    dropout_rate_inference: float = 0.3,
    dropout_rate_model: float = 0.3,
    alpha_xform: str = "softmax",
    do_annealing: bool = True,
    annealing_sharpness: float = 5.0,
    n_epochs_annealing: int = 80,
    n_epochs_burnin: int = 10,
    learning_rate: float = 0.01,
    clip_normalization: float = None,
    multi_gpu: bool = False,
    strategy: str = None,
    generative_only: bool = False,
):
    annealing_factor = Variable(0.0) if do_annealing else Variable(1.0)

    # Setup optimizer
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, clipnorm=clip_normalization,
    )

    # Stretegy for distributed learning
    if multi_gpu:
        strategy = MirroredStrategy()
    elif strategy is None:
        strategy = get_strategy()

    with strategy.scope():

        # Model to generate observations from states
        generative_model = _generative_model_structure(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=sequence_length,
            n_layers=n_layers_model,
            n_units=n_units_model,
            dropout_rate=dropout_rate_model,
            learn_means=learn_means,
            learn_covariances=learn_covariances,
            initial_mean=initial_mean,
            initial_covariances=initial_covariances,
            alpha_xform=alpha_xform,
        )

        if not generative_only:
            # Model to infer latent states
            inference_model = _inference_model_structure(
                n_states=n_states,
                n_channels=n_channels,
                sequence_length=sequence_length,
                n_layers=n_layers_inference,
                n_units=n_units_inference,
                dropout_rate=dropout_rate_inference,
            )

            # Build the full model: inference + generative model
            inputs = inference_model.input
            inference_output = inference_model(inputs)
            outputs = generative_model(inference_output)
            model = Model(inputs=inputs, outputs=outputs)

        else:
            model = generative_model

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=[_ll_loss_fn, _create_kl_loss_fn(annealing_factor=annealing_factor)],
        )

    # Override original summary method
    if not generative_only:
        model.original_summary = model.summary

        def full_summary(*args, **kwargs):
            # Display summary for the inference model
            inference_model = model.get_layer("inference_model")
            inference_model.summary(*args, **kwargs)

            # Display summary for the generative model
            generative_model = model.get_layer("generative_model")
            generative_model.summary(*args, **kwargs)

        model.summary = full_summary

    # Callbacks
    burnin_callback = BurninCallback(epochs=n_epochs_burnin)

    annealing_callback = AnnealingCallback(
        annealing_factor=annealing_factor,
        annealing_sharpness=annealing_sharpness,
        n_epochs_annealing=n_epochs_annealing,
    )

    # Override original fit method
    model.original_fit_method = model.fit

    def anneal_burnin_fit(*args, **kwargs):
        args = list(args)
        if len(args) > 5:
            args[5] = listify(args[5]) + [annealing_callback, burnin_callback]
        if "callbacks" in kwargs:
            kwargs["callbacks"] = listify(kwargs["callbacks"]) + [
                annealing_callback,
                burnin_callback,
            ]
        return model.original_fit_method(*args, **kwargs)

    model.fit = anneal_burnin_fit

    # Override original predict method
    model.original_predict_function = model.predict

    def named_predict(*args, **kwargs):
        prediction = model.original_predict_function(*args, **kwargs)
        return_names = [
            "ll_loss",
            "kl_loss",
            "theta_t",
            "m_theta_t",
            "log_s2_theta_t",
            "mu_theta_jt",
            "log_sigma2_theta_j",
        ]
        prediction_dict = dict(zip(return_names, prediction))
        return prediction_dict

    model.predict = named_predict

    # Method to predict the inferred state time course
    if not generative_only:

        def predict_states(*args, **kwargs):
            m_theta_t = model.predict(*args, **kwargs)["m_theta_t"]
            return np.concatenate(m_theta_t)

        model.predict_states = predict_states

    # Method to calculate the free energy (log likelihood + KL divergence)
    def free_energy(*args, **kwargs):
        ll_loss = model.predict(*args, **kwargs)["ll_loss"]
        kl_loss = model.predict(*args, **kwargs)["kl_loss"]
        return np.mean(ll_loss + kl_loss)

    # Method to get the learned means and covariances for each state
    def state_means_covariances():
        for layer in model.layers:
            if hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, MultivariateNormalLayer):
                        mvn_layer = sub_layer

                        means = mvn_layer.get_means()
                        covariances = mvn_layer.get_covariances()
                        return means, covariances

    model.state_means_covariances = state_means_covariances

    # Method to get the alpha scaling of each state
    def alpha_scaling():
        mix_means_covs_layer = model.get_layer("mix_means_covs")
        alpha_scaling = mix_means_covs_layer.get_alpha_scaling()
        return alpha_scaling

    model.alpha_scaling = alpha_scaling

    return model


def _ll_loss_fn(y_true, ll_loss):
    """The first output of the model is the negative log likelihood
       so we just need to return it."""
    return ll_loss


def _create_kl_loss_fn(annealing_factor):
    def _kl_loss_fn(y_true, kl_loss):
        """Second output of the model is the KL divergence loss.
        We multiply with an annealing factor."""
        return annealing_factor * kl_loss

    return _kl_loss_fn


def _inference_model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    n_layers: int,
    n_units: int,
    dropout_rate: float,
):
    """Inference RNN
       - q(theta_t)     ~ N(m_theta_t, s2_theta_t)
       - m_theta_t      ~ affine(RNN(Y_<=t))
       - log_s2_theta_t ~ affine(RNN(Y_<=t))
    """
    # Definition of layers
    inputs = layers.Input(shape=(sequence_length, n_channels), name="data")
    input_normalisation_layer = layers.LayerNormalization()
    input_dropout_layer = layers.Dropout(dropout_rate)
    output_layers = InferenceRNNLayers(n_layers, n_units, dropout_rate)
    m_theta_t_layer = layers.Dense(n_states, activation="linear", name="m_theta_t")
    log_s2_theta_t_layer = layers.Dense(
        n_states, activation="linear", name="log_s2_theta_t"
    )
    theta_t_layer = ReparameterizationLayer(name="theta_t")

    # Inference RNN data flow
    inputs_norm = input_normalisation_layer(inputs)
    inputs_norm_dropout = input_dropout_layer(inputs_norm)
    output = output_layers(inputs_norm_dropout)
    m_theta_t = m_theta_t_layer(output)
    log_s2_theta_t = log_s2_theta_t_layer(output)
    theta_t = theta_t_layer([m_theta_t, log_s2_theta_t])

    # Inference model outputs
    outputs = [inputs, theta_t, m_theta_t, log_s2_theta_t]

    return Model(inputs=inputs, outputs=outputs, name="inference_model")


def _generative_model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    n_layers: int,
    n_units: int,
    dropout_rate: float,
    learn_means: bool,
    learn_covariances: bool,
    initial_mean: np.ndarray,
    initial_covariances: np.ndarray,
    alpha_xform: str,
):
    """Model RNN
       - p(theta_t|theta_<t) ~ N(mu_theta_jt, sigma2_theta_j)
       - mu_theta_jt         ~ affine(RNN(theta_<t))
       - log_sigma2_theta_j  = trainable constant
    """
    # Definition of layers
    inputs = layers.Input(shape=(sequence_length, n_channels), name="data")
    theta_t = layers.Input(shape=(sequence_length, n_states), name="theta_t")
    m_theta_t = layers.Input(shape=(sequence_length, n_states), name="m_theta_t")
    log_s2_theta_t = layers.Input(
        shape=(sequence_length, n_states), name="log_s2_theta_t"
    )

    theta_t_dropout_layer = layers.Dropout(dropout_rate)
    output_layers = ModelRNNLayers(n_layers, n_units, dropout_rate)
    mu_theta_jt_layer = layers.Dense(n_states, activation="linear", name="mu_theta_jt")
    log_sigma2_theta_j_layer = TrainableVariablesLayer(
        [n_states],
        initial_values=zeros(n_states),
        trainable=True,
        name="log_sigma2_theta_j",
    )

    observation_means_covs_layer = MultivariateNormalLayer(
        n_states,
        n_channels,
        learn_means=learn_means,
        learn_covariances=learn_covariances,
        initial_means=initial_mean,
        initial_covariances=initial_covariances,
        name="mvn",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        n_states, n_channels, alpha_xform, name="mix_means_covs"
    )

    log_likelihood_layer = LogLikelihoodLayer(name="ll")
    kl_loss_layer = KLDivergenceLayer(name="kl")

    # Model RNN data flow
    theta_t_dropout = theta_t_dropout_layer(theta_t)
    output = output_layers(theta_t_dropout)
    mu_theta_jt = mu_theta_jt_layer(output)
    log_sigma2_theta_j = log_sigma2_theta_j_layer(inputs)  # inputs not used
    mu_j, D_j = observation_means_covs_layer(inputs)  # inputs not used
    m_t, C_t = mix_means_covs_layer([theta_t, mu_j, D_j])
    ll_loss = log_likelihood_layer([inputs, m_t, C_t])
    kl_loss = kl_loss_layer(
        [m_theta_t, log_s2_theta_t, mu_theta_jt, log_sigma2_theta_j]
    )

    # Generative model inputs and outputs
    inputs = [inputs, theta_t, m_theta_t, log_s2_theta_t]
    outputs = [
        ll_loss,
        kl_loss,
        theta_t,
        m_theta_t,
        log_s2_theta_t,
        mu_theta_jt,
        log_sigma2_theta_j,
    ]

    return Model(inputs=inputs, outputs=outputs, name="generative_model")
