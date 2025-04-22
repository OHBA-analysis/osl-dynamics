"""Example script for demonstrating DyNeStE's ability to learn long-range dependependcies.

- An HSMM is simulated.
- The output of this script will vary slightly due to random sampling.
"""

print("Setting up")
import os
from osl_dynamics import data, simulation
from osl_dynamics.inference import tf_ops, modes, metrics
from osl_dynamics.models.dyneste import Config, Model
from osl_dynamics.utils import plotting

# Make directory for results and plots
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_states=3,
    n_channels=11,
    sequence_length=200,
    inference_n_units=128,
    inference_normalization="layer",
    model_n_units=128,
    model_normalization="layer",
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    do_gs_annealing=True,
    gs_annealing_curve="exp",
    initial_gs_temperature=1.0,
    final_gs_temperature=0.06,
    gs_annealing_slope=0.04,
    n_gs_annealing_epochs=120,
    batch_size=16,
    learning_rate=5e-3,
    lr_decay=0.01,
    n_epochs=120,
)

# Simulate data
print("Simulating data")
sim = simulation.HSMM_MVN(
    n_samples=25600,
    n_channels=config.n_channels,
    n_states=config.n_states,
    means="zero",
    covariances="random",
    observation_error=0.0,
    gamma_shape=10,
    gamma_scale=5,
)
sim.standardize()
training_data = data.Data(sim.time_series)

# Plot the transition probability matrix for state switching in the HSMM
plotting.plot_matrices(
    sim.off_diagonal_trans_prob, filename="figures/sim_trans_prob.png"
)

# Create tensorflow datasets for training and model evaluation
training_dataset = training_data.dataset(
    config.sequence_length, config.batch_size, shuffle=True
)
prediction_dataset = training_data.dataset(
    config.sequence_length, config.batch_size, shuffle=False
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(training_dataset)

# Save model
model_dir = "results/model"
model.save(model_dir)

# Free energy = Negative Log Likelihood + KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state probabilities
inf_alp = model.get_alpha(prediction_dataset)

# State time courses
sim_stc = sim.mode_time_course
inf_stc = modes.argmax_time_courses(inf_alp)

# Calculate the dice coefficient between state time courses
orders = modes.match_modes(sim_stc, inf_stc, return_order=True)
inf_stc = inf_stc[:, orders[1]]
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

plotting.plot_alpha(
    sim_stc,
    inf_stc,
    y_labels=["Ground Truth", "DyNeStE"],
    filename="figures/compare.png",
)

plotting.plot_state_lifetimes(
    sim_stc, x_label="Lifetime", y_label="Occurrence", filename="figures/sim_lt.png"
)
plotting.plot_state_lifetimes(
    inf_stc, x_label="Lifetime", y_label="Occurrence", filename="figures/inf_lt.png"
)

# Ground truth vs inferred covariances
sim_cov = sim.covariances
inf_cov = model.get_covariances()[orders[1]]

plotting.plot_matrices(sim_cov, filename="figures/sim_cov.png")
plotting.plot_matrices(inf_cov, filename="figures/inf_cov.png")

# Sample from model RNN
sam_alp = model.sample_alpha(25600)
sam_stc = modes.argmax_time_courses(sam_alp)

plotting.plot_state_lifetimes(
    sam_stc,
    x_label="Lifetime",
    x_range=[0, 150],
    y_label="Occurrence",
    filename="figures/sam_lt.png",
)
