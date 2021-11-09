"""Example script for running multi-scale inference on resting-mode MEG data for one subject.

- The data is stored on the BMRC cluster: /well/woolrich/projects/uk_meg_notts
- The time course for variance component is fixed, but the variance component is still
  trainable.

- The time courses for mean and functional connectivity (fc) are learned.

- Observation model: 
    y_t = N (m_t, C_t)

    where m_t = \sum_j \alpha_jt \mu_j, C_t = G F_t G
    G - diagonal with variances, constant over time and modes
    F_t - correlation matrix

    F_t = \sum_j \gamma_jt D_j


"""



print("Setting up")
from dynemo.data import OSL_HMM, Data, manipulation
from dynemo.inference import metrics, modes, tf_ops
from dynemo.models import Config, Model

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.moment_helpers import cov2corr

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    multiple_scale=True,
    n_modes=6,
    sequence_length=200,
    inference_rnn="lstm",
    inference_n_units=64,
    inference_n_layers=1,
    inference_normalization="layer",
    inference_dropout_rate=0.0,
    model_rnn="lstm",
    model_n_units=64,
    model_n_layers=1,
    model_normalization="layer",
    model_dropout_rate=0.0,
    theta_normalization="layer",
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=True,
    learn_vars=True,
    learn_fcs=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=64,
    learning_rate=0.01,
    n_epochs=200,
    fix_variance=True,
)

# Read MEG data
print("Reading MEG data")
prepared_data = Data(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/prepared_data/subject1.mat",
    sampling_frequency=250,
    n_embeddings=15,
)

config.n_channels = prepared_data.n_channels

# Prepare dataset
training_dataset = prepared_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
)
prediction_dataset = prepared_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

# Initialise variances and fc with the final HMM covariances
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-1_K-6/hmm.mat"
)
initial_covariances = hmm.covariances

initial_variances = np.zeros([config.n_modes, config.n_channels], dtype=np.float32)
for i in range(config.n_modes):
    initial_variances[i] = np.sqrt(np.diag(initial_covariances[i]))
# enforce positive variance
config.initial_vars = np.maximum(initial_variances, 1e-6)

initial_fcs = np.zeros([config.n_modes, config.n_channels, config.n_channels], dtype=np.float32)
for i in range(config.n_modes):
    initial_fcs[i] = cov2corr(initial_covariances[i])
config.initial_fcs = initial_fcs

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="tmp/model",
    verbose=1,
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred mode mixing factors and mode time course
inf_alpha, inf_beta, inf_gamma = model.get_alpha(prediction_dataset)

inf_stc_alpha = modes.time_courses(inf_alpha)
inf_stc_gamma = modes.time_courses(inf_gamma)

hmm_stc = manipulation.trim_time_series(
    time_series=hmm.mode_time_course(),
    sequence_length=config.sequence_length,
)

# Dice coefficient
print("Dice coefficient for alpha:", metrics.dice_coefficient(hmm_stc, inf_stc_alpha))
print("Dice coefficient for gamma:", metrics.dice_coefficient(hmm_stc, inf_stc_gamma))



plt.figure()
fig, ax = plt.subplots(config.n_modes, figsize=(17,config.n_modes))
for i in range(config.n_modes):
    ax[i].plot(hmm_stc[:2000,i], label="hmm_stc")
    ax[i].plot(inf_alpha[:2000,i], label="inferred alpha")
    ax[i].set_ylim([-0.05,1.05])
ax[0].legend()
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.title("inferred alpha versus inferred stc from hmm (first 2000 time points).")
plt.ylabel("mixing probability")
plt.xlabel("time")
plt.tight_layout()
fig.savefig("figures/real_data_inf_alpha.png")

plt.figure()
fig1, ax1 = plt.subplots(config.n_modes, figsize=(17,config.n_modes))
for i in range(config.n_modes):
    ax1[i].plot(hmm_stc[:2000,i], label="hmm_stc")
    ax1[i].plot(inf_gamma[:2000,i], label="inferred gamma")
    ax1[i].set_ylim([-0.05,1.05])
ax1[0].legend()
fig1.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.title("inferred gamma versus inferred stc from hmm (first 2000 time points).")
plt.ylabel("mixing probability")
plt.xlabel("time")
plt.tight_layout()
fig1.savefig("figures/real_data_inf_gamma.png")

# Delete temporary directory
prepared_data.delete_dir()
