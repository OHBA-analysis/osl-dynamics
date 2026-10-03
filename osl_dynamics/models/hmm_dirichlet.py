"""HMM-Dirichlet for compositional mode time courses.

The sequence length is measured in hidden-state windows. Data contains raw
samples. The emission uses the mean per-sample log-density.
"""

import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from osl_dynamics import data
from osl_dynamics.inference import optimizers
from osl_dynamics.inference.initializers import WeightInitializer
from osl_dynamics.inference.layers import (
    HiddenMarkovStateInferenceLayer,
    SumLogLikelihoodLossLayer,
    DirichletConcentrationLayer,
    WindowedDirichletLogLikelihoodLayer,
)
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.models.inf_mod_base import (
    MarkovStateInferenceModelConfig,
    MarkovStateInferenceModelBase,
)


@dataclass
class Config(BaseModelConfig, MarkovStateInferenceModelConfig):
    """HMM-Dirichlet settings.

    ``window_size`` is raw samples per state; ``sequence_length`` is states per
    sequence. For the thesis these are 100 and 500, respectively. The model's
    posterior sampling frequency is the input frequency divided by window_size.
    ``optimizer_kwargs`` supports the old scripts' explicit Adam settings.
    """

    model_name: str = "HMM-Dirichlet"
    window_size: int = 100
    learn_concentration: bool = True
    initial_concentration: np.ndarray = None
    concentration_epsilon: float = 1e-9
    optimizer_kwargs: dict = None
    init_method: str = "random_subset"
    n_init: int = 3
    n_init_epochs: int = 1
    init_take: float = 1.0

    def __post_init__(self):
        if self.model_name != "HMM-Dirichlet":
            raise ValueError("model_name must be HMM-Dirichlet.")
        for key in ["window_size", "sequence_length", "n_channels", "n_states"]:
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{key} must be a positive integer.")
        if self.n_states < 2 or self.n_channels < 2 or self.sequence_length < 2:
            raise ValueError(
                "At least two states, channels and windows per sequence are required."
            )
        if not isinstance(self.learn_concentration, (bool, np.bool_)):
            raise ValueError("learn_concentration must be boolean.")
        if (
            not np.isfinite(self.concentration_epsilon)
            or self.concentration_epsilon <= 0
        ):
            raise ValueError("concentration_epsilon must be finite and positive.")
        if self.optimizer_kwargs is not None and not isinstance(
            self.optimizer_kwargs, dict
        ):
            raise ValueError("optimizer_kwargs must be a dictionary.")
        if isinstance(self.initial_concentration, (str, os.PathLike)):
            self.initial_concentration = np.load(self.initial_concentration)
        if self.initial_concentration is not None:
            self.initial_concentration = np.asarray(
                self.initial_concentration, dtype=np.float32
            )
            if self.initial_concentration.shape != (self.n_states, self.n_channels):
                raise ValueError(
                    "initial_concentration must have shape (n_states, n_channels)."
                )
            if not np.isfinite(self.initial_concentration).all() or np.any(
                self.initial_concentration <= self.concentration_epsilon
            ):
                raise ValueError(
                    "initial_concentration must be finite and exceed epsilon."
                )
        self.validate_hmm_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()
        for name, shape in [
            ("initial_trans_prob", (self.n_states, self.n_states)),
            ("initial_state_probs", (self.n_states,)),
        ]:
            value = getattr(self, name)
            if value is not None and (
                value.shape != shape
                or not np.isfinite(value).all()
                or np.any(value < 0)
            ):
                raise ValueError(f"Invalid {name}.")


class Model(MarkovStateInferenceModelBase):
    """Dirichlet HMM whose posterior has one row per raw-sample window."""

    config_type = Config

    @property
    def raw_sequence_length(self):
        return self.config.sequence_length * self.config.window_size

    def build_model(self):
        c = self.config
        x = layers.Input((self.raw_sequence_length, c.n_channels), name="data")
        concentration = DirichletConcentrationLayer(
            c.n_states,
            c.n_channels,
            c.learn_concentration,
            c.initial_concentration,
            c.concentration_epsilon,
            name="concentration",
        )(x)
        ll = WindowedDirichletLogLikelihoodLayer(c.window_size, name="ll")(
            [x, concentration]
        )
        inference = HiddenMarkovStateInferenceLayer(
            c.n_states,
            c.sequence_length,
            c.initial_trans_prob,
            c.trans_prob_prior,
            c.initial_state_probs,
            c.learn_trans_prob,
            c.learn_initial_state_probs,
            implementation=c.baum_welch_implementation,
            dtype="float64",
            name="hid_state_inf",
        )
        gamma, xi = inference(ll)
        loss = SumLogLikelihoodLossLayer(c.loss_calc, name="ll_loss")([ll, gamma])
        self.model = tf.keras.Model(
            {"data": x}, {"ll_loss": loss, "gamma": gamma, "xi": xi}, name=c.model_name
        )

    def compile(self, optimizer=None, **kwargs):
        if optimizer is None:
            c = self.config
            options = dict(c.optimizer_kwargs or {})
            options.setdefault("learning_rate", c.learning_rate)
            if c.gradient_clip is not None:
                options.setdefault("clipnorm", c.gradient_clip)
            base = tf.keras.optimizers.get(
                {"class_name": c.optimizer.lower(), "config": options}
            )
            decay = (1 + c.trans_prob_update_delay) ** -c.trans_prob_update_forget
            ema = optimizers.ExponentialMovingAverage(c.learning_rate, decay)
            optimizer = optimizers.MarkovStateModelOptimizer(
                base,
                ema,
                self.model.get_layer("hid_state_inf").trainable_variables,
                learning_rate=c.learning_rate,
            )
        ModelBase.compile(self, optimizer=optimizer, **kwargs)

    def make_dataset(
        self,
        inputs,
        shuffle=False,
        concatenate=False,
        step_size=None,
        drop_last_batch=False,
    ):
        """Convert Data into a TensorFlow Dataset using the original batching.

        step_size is in raw samples. Prebuilt datasets are passed through;
        a single dataset is wrapped in a list when concatenate=False.
        """
        if isinstance(inputs, str) or isinstance(inputs, np.ndarray):
            # str or numpy array -> Data object
            inputs = data.Data(inputs)

        if isinstance(inputs, data.Data):
            sequence_length = self.config.sequence_length * self.config.window_size
            if inputs.use_tfrecord:
                outputs = inputs.tfrecord_dataset(
                    sequence_length,
                    self.config.batch_size,
                    shuffle=shuffle,
                    concatenate=concatenate,
                    step_size=step_size,
                    drop_last_batch=drop_last_batch,
                    overwrite=True,
                )
            else:
                outputs = inputs.dataset(
                    sequence_length,
                    self.config.batch_size,
                    shuffle=shuffle,
                    concatenate=concatenate,
                    step_size=step_size,
                    drop_last_batch=drop_last_batch,
                )

        elif isinstance(inputs, tf.data.Dataset) and not concatenate:
            outputs = [inputs]

        else:
            outputs = inputs

        return outputs

    def get_training_time_series(self, training_data, prepared=True, concatenate=False):
        return training_data.trim_time_series(
            self.raw_sequence_length, prepared=prepared, concatenate=concatenate
        )

    def get_concentration(self):
        return self.model.get_layer("concentration")(tf.constant(1)).numpy()

    def get_observation_model_parameters(self):
        return self.get_concentration()

    def set_concentration(self, concentration, update_initializer=True):
        value = np.asarray(concentration, dtype=np.float32)
        c = self.config
        if value.shape != (c.n_states, c.n_channels) or not np.isfinite(value).all():
            raise ValueError(
                "Concentration must be finite with shape (n_states, n_channels)."
            )
        if np.any(value <= c.concentration_epsilon):
            raise ValueError("Concentrations must exceed epsilon.")
        layer = self.model.get_layer("concentration")
        kernel = layer.bijector.inverse(value - c.concentration_epsilon).numpy()
        tensor = layer.layers[0]
        tensor.tensor.assign(kernel)
        if update_initializer:
            tensor.tensor_initializer = WeightInitializer(kernel)

    def set_observation_model_parameters(
        self, observation_model_parameters, update_initializer=True
    ):
        self.set_concentration(observation_model_parameters, update_initializer)

    def get_log_likelihood(self, x):
        """Return (batch, windows, states) mean-sample log-densities."""
        return self.model.get_layer("ll")(
            [tf.convert_to_tensor(x), tf.convert_to_tensor(self.get_concentration())]
        ).numpy()

    def get_alpha(
        self, dataset, concatenate=False, remove_edge_effects=False, **kwargs
    ):
        if self.is_multi_gpu:
            raise ValueError("Load with single_gpu=True before inference.")
        length = self.config.sequence_length
        if remove_edge_effects and length % 4:
            raise ValueError(
                "Overlapping inference requires sequence_length divisible by four."
            )
        if remove_edge_effects and (
            isinstance(dataset, tf.data.Dataset)
            or (
                isinstance(dataset, list)
                and dataset
                and isinstance(dataset[0], tf.data.Dataset)
            )
        ):
            raise ValueError(
                "Pass raw arrays or Data for overlapping inference, not prebatched datasets."
            )
        datasets = self.make_dataset(
            dataset,
            step_size=self.raw_sequence_length // 2 if remove_edge_effects else None,
        )
        result = []
        for ds in datasets:
            batches = [self.predict(batch, **kwargs)["gamma"] for batch in ds]
            if not batches:
                raise ValueError("A session contains no complete sequence.")
            a = np.concatenate(batches)
            if remove_edge_effects and len(a) > 1:
                trim = length // 4
                pieces = [a[0, :-trim], *list(a[1:-1, trim:-trim]), a[-1, trim:]]
                result.append(np.concatenate(pieces))
            else:
                result.append(a.reshape(-1, self.config.n_states))
        return np.concatenate(result) if concatenate or len(result) == 1 else result

    def evidence(self, dataset):
        """Log forward score per window (or sequence for loss_calc='sum').

        The thesis's mean log-density is a tempered emission, not the normalized
        joint density of all raw samples in a window.
        """
        from scipy.special import logsumexp

        with np.errstate(divide="ignore"):
            log_trans = np.log(self.get_trans_prob())
            log_initial = np.log(self.get_initial_state_probs())
        scores = []
        for batch in self.make_dataset(dataset, concatenate=True):
            ll = self.get_log_likelihood(batch["data"])
            forward = ll[:, 0] + log_initial
            for t in range(1, ll.shape[1]):
                forward = ll[:, t] + logsumexp(forward[:, :, None] + log_trans, axis=1)
            scores.extend(logsumexp(forward, axis=-1))
        if not scores:
            raise ValueError("No complete sequences.")
        return float(
            np.mean(scores)
            / (self.config.sequence_length if self.config.loss_calc == "mean" else 1)
        )

    def get_n_params_generative_model(self):
        c = self.config
        return (
            int(c.learn_concentration) * c.n_states * c.n_channels
            + int(c.learn_trans_prob) * c.n_states * (c.n_states - 1)
            + int(c.learn_initial_state_probs) * (c.n_states - 1)
        )

    def set_regularizers(self, training_dataset):
        raise NotImplementedError("No data-dependent Dirichlet regularizer is defined.")

    def set_random_state_time_course_initialization(self, training_dataset):
        """Fit concentrations to raw samples assigned by a window-level chain.

        Accumulate per-state sufficient statistics, avoiding the old initializer's
        broadcast of the last state's log mean into all states.
        """
        from scipy.optimize import minimize
        from scipy.special import digamma, gammaln

        c = self.config
        count = np.zeros(c.n_states)
        sums = np.zeros((c.n_states, c.n_channels))
        squares, logs = np.zeros_like(sums), np.zeros_like(sums)
        for batch in training_dataset:
            x = np.asarray(batch["data"], dtype=float).reshape(
                -1, c.window_size, c.n_channels
            )
            if not np.isfinite(x).all() or np.any(x < 0) or np.any(x.sum(-1) <= 0):
                raise ValueError("Invalid compositional data.")
            x = np.clip(x, 1e-9, 1)
            x /= x.sum(-1, keepdims=True)
            state = self.sample_state_time_course(len(x)).argmax(-1)
            for k in range(c.n_states):
                values = x[state == k].reshape(-1, c.n_channels)
                count[k] += len(values)
                sums[k] += values.sum(0)
                squares[k] += (values**2).sum(0)
                logs[k] += np.log(values).sum(0)
        if np.any(count == 0):
            raise ValueError(
                "Random initialization missed a state; use more windows or random_subset."
            )
        mean = sums / count[:, None]
        var = np.maximum(squares / count[:, None] - mean**2, 1e-10)
        log_mean = logs / count[:, None]
        concentration = np.empty_like(sums)
        for k in range(c.n_states):
            precision = max(np.mean(mean[k] * (1 - mean[k]) / var[k] - 1), 1e-2)
            start = np.maximum(mean[k] * precision, 1e-3)

            def objective(log_c):
                value = np.exp(log_c)
                ll = (
                    gammaln(value.sum())
                    - gammaln(value).sum()
                    + ((value - 1) * log_mean[k]).sum()
                )
                gradient = (digamma(value.sum()) - digamma(value) + log_mean[k]) * value
                return -ll, -gradient

            fit = minimize(
                objective,
                np.log(start),
                jac=True,
                method="L-BFGS-B",
                bounds=[(np.log(max(c.concentration_epsilon * 2, 1e-8)), np.log(1e8))]
                * c.n_channels,
            )
            if not fit.success:
                raise ValueError(
                    f"Dirichlet initialization failed for state {k}: {fit.message}"
                )
            concentration[k] = np.exp(fit.x)
        if c.learn_concentration:
            self.set_concentration(concentration)
