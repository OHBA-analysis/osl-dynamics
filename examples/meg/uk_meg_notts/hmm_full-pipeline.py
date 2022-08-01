"""Example script for fitting an HMM to data from the Nottingham site of the
MEG UK Partnership dataset.
"""

import os
import numpy as np
from osl_dynamics import analysis, data
from osl_dynamics.inference import tf_ops, modes, metrics
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting

# Directory for plots
os.makedirs("figures", exist_ok=True)

# Load an HMM inferred with MATLAB HMM-MAR for comparison
hmmmar = data.OSL_HMM("/well/woolrich/projects/mrc_meguk/notts22/results/K-6/hmm.mat")

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_states=6,
    n_channels=38,
    sequence_length=100,
    learn_means=False,
    learn_covariances=True,
    learn_transprob=True,
    batch_size=64,
    learning_rate=0.001,
    n_epochs=10,
)

# Use the final HMM covariances for initialisation
config.initial_covariances = hmmmar.covariances

# Load dataset
training_data = data.Data(
    [
        f"/well/woolrich/projects/mrc_meguk/notts22/prepared_data/subject{i}.mat"
        for i in range(1, 3)
    ]
)
training_data.prepare()

# Create tensorflow datasets for training and model evaluation
training_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
)

# Build model
model = Model(config)
model.summary()

# Train the model
history = model.fit(training_dataset)

# Get inferred state probabilities
prediction_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)
inf_alp = model.get_alpha(prediction_dataset)

# Get the inferred state probability from HMM-MAR
# and trim data points lost due to separating the data into sequences
hmm_gam = data.processing.trim_time_series(
    hmmmar.gamma,
    discontinuities=hmmmar.discontinuities,
    sequence_length=config.sequence_length,
    concatenate=True,
)

_, order = modes.match_modes(inf_alp, hmm_gam, return_order=True)
hmm_gam = hmm_gam[:, order]

plotting.plot_alpha(
    inf_alp,
    hmm_gam,
    n_samples=2000,
    y_labels=["Inferred", "HMM-MAR"],
    filename="figures/alp.png",
)

dice = metrics.dice_coefficient(inf_alp, hmm_gam)
print("Dice coefficient:", dice)

inf_fo = modes.fractional_occupancies(inf_alp)
hmm_fo = modes.fractional_occupancies(hmm_gam)
print("Fractional occupancies (Inferred):", inf_fo)
print("Fractional occupancies (HMM-MAR):", hmm_fo)

# Transition probability matrices
inf_tp = model.get_transprob()
inf_tp = inf_tp[np.ix_(order, order)]
np.fill_diagonal(inf_tp, 0)

hmm_tp = hmmmar.trans_prob
hmm_tp = hmm_tp[np.ix_(order, order)]
np.fill_diagonal(hmm_tp, 0)

plotting.plot_matrices([inf_tp, hmm_tp], filename="figures/transprob.png")

# Inferred covariances
inf_cov = model.get_covariances()
inf_cov = inf_cov[order]

hmm_cov = hmmmar.covariances
hmm_cov = hmm_cov[order]

plotting.plot_matrices(inf_cov, group_color_scale=False, filename="figures/inf_cov.png")
plotting.plot_matrices(hmm_cov, group_color_scale=False, filename="figures/hmm_cov.png")

# Compute partial covariances of state time courses regressing onto data,
# where data is the non-whitened hilbert envelope data
prediction_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
    concatenate=False,  # gives a list of indivudual subject data
)
ts = training_data.time_series()
ts = data.processing.trim_time_series(
    ts,
    discontinuities=hmmmar.discontinuities,
    sequence_length=config.sequence_length,
)
alp = model.get_alpha(prediction_dataset)  # subject-specific alpha
pcov = analysis.modes.partial_covariances(ts, alp)
pcov = np.mean(pcov, axis=0)  # average over subjects

analysis.power.save(
    power_map=pcov,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    filename="figures/pcov_.png",
)
