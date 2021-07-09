"""Example script for running inference on data with a soft mixture states.

- Demonstrates VRAD's ability to infer a soft mixture of states.
"""

print("Setting up")
from vrad import data, simulation
from vrad.inference import metrics, states, tf_ops
from vrad.models import Config, Model
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_states=8,
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
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
)

# Simulate data
print("Simulating data")
sim = simulation.MixedSine_MVN(
    n_samples=25600,
    n_states=8,
    n_channels=80,
    relative_activation=[1, 1, 1, 1, 1, 0.5, 0.5, 0.5],
    amplitudes=[8, 7, 6, 5, 4, 3, 2, 1],
    frequencies=[1, 2, 3, 4, 5, 6, 6, 6],
    sampling_frequency=250,
    means="zero",
    covariances="random",
)
meg_data = data.Data(sim.time_series)

config.n_channels = meg_data.n_channels

# Prepare dataset
training_dataset = meg_data.training_dataset(config.sequence_length, config.batch_size)
prediction_dataset = meg_data.prediction_dataset(
    config.sequence_length, config.batch_size
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="tmp/weights",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Compare the inferred state time course to the ground truth
alpha = model.predict_states(prediction_dataset)
matched_sim_stc, matched_alpha = states.match_states(sim.state_time_course, alpha)
plotting.plot_separate_time_series(
    matched_alpha, matched_sim_stc, n_samples=2000, filename="stc.png"
)

corr = metrics.alpha_correlation(matched_alpha, matched_sim_stc)
print("Correlation (VRAD vs Simulation):", corr)

# Delete the temporary folder holding the data
meg_data.delete_dir()
