"""
Example to show DyNeMo's ability to learn non-categorical descriptions of latent dynamics,
using non-overlapping Gaussian Kernels as the covariance matrix elements.

This example demonstrates what can sometimes happen when the underlying ground truth
covariances have a similar structure - modes can be "knocked out" and/or combined
during the inference stage. See the first two modes (as they are plotted) here.

Ryan Timms and Chetan Gohil, 2021.
"""

print("Setting up")
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dynemo import data, simulation
from dynemo.inference import modes, tf_ops
from dynemo.models.rigo import Config, Model

# GPU settings
tf_ops.gpu_growth()


def gaussian_heatmap(center=(2, 2), image_size=(10, 10), sig=1):
    """
    Produces a single gaussian at expected center

    Parameters
    ----------
    center: tuple of ints
        the mean position (X, Y) - where high value expected
    image_size: tuple of ints
        The total image size (width, height)
    sig: int
        The sigma value
    """
    x_axis = np.linspace(0, image_size[0] - 1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1] - 1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel


def set_all_seeds(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)


print("Setting up")

# Set all seeds for repro.
seed_value = 666
set_all_seeds(seed_value)


# GPU settings
tf_ops.gpu_growth()

n_modes = 6
n_chans = 80

# Settings
config = Config(
    n_modes=n_modes,
    sequence_length=200,
    inference_rnn="lstm",
    inference_n_units=64,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_units=64,
    model_normalization="layer",
    theta_normalization=None,
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=16,
    learning_rate=0.001,
    n_epochs=500,
)

# Generate ground truth covariance matrix for each mode,
# shape should be (n_modes,n_channels, n_channels)

GT_covz = np.zeros((n_modes, n_chans, n_chans))
for i in range(n_modes):
    tmp = (
        gaussian_heatmap(
            center=(
                np.random.randint(1, n_chans - 1),
                np.random.randint(1, n_chans - 1),
            ),
            image_size=(n_chans, n_chans),
            sig=5,
        )
        + np.eye(n_chans, n_chans)
    )
    GT_covz[i, :, :] = tmp @ tmp.T

# Simulate data
print("Simulating data")
sim = simulation.MixedSine_MVN(
    n_samples=25600 * 2,
    n_modes=6,
    n_channels=80,
    relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.1],
    amplitudes=[6, 5, 4, 3, 2, 1],
    frequencies=[1, 2, 3, 4, 6, 8],
    sampling_frequency=250,
    means="zero",
    covariances=GT_covz,
    random_seed=123,
)
sim.standardize()
sim_stc = sim.mode_time_course
meg_data = data.Data(sim.time_series)

config.n_channels = meg_data.n_channels

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length, config.batch_size, shuffle=True
)
prediction_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="model/weights",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Compare the inferred mode time course to the ground truth
alpha = model.get_alpha(prediction_dataset)
inferred_covz = model.get_covariances()

# Match inferred and ground truth covariances/alphas for visulaisation
sim_stc, inf_stc = modes.match_modes(sim_stc, alpha)
GT_covz, inferred_covz = modes.match_covariances(GT_covz, inferred_covz)

for i in range(n_modes):
    plt.figure()
    plt.plot(inf_stc[:, i])
    plt.plot(sim_stc[:, i])
    plt.xlim(0, 500)
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.imshow(inferred_covz[i, :, :])
    plt.title("Inferred")
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(GT_covz[i, :, :])
    plt.title("Ground truth")
    plt.colorbar()

# Delete temporary directory
meg_data.delete_dir()
