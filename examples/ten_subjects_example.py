"""Example script for running inference on resting-state MEG data for ten subjects.

- The data is stored on the BMRC cluster: /well/woolrich/shared/vrad
- Uses the final covariances inferred by an HMM fit from OSL for the covariance of each
  state.
- Covariances are NOT trainable.
- Achieves a dice coefficient of ~0.76 (when compared to the OSL HMM state time course).
- Achieves a free energy of ~27,900,000.
"""

print("Setting up")
import numpy as np
from vrad import data
from vrad.analysis import maps, spectral
from vrad.inference import metrics, states, tf_ops
from vrad.models import RNNGaussian

# GPU settings
tf_ops.gpu_growth()
multi_gpu = True

# Settings
n_states = 6
sequence_length = 400
batch_size = 128

do_annealing = True
annealing_sharpness = 5

n_epochs = 200
n_epochs_annealing = 100

rnn_type = "lstm"
normalization_type = "layer"

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 64
n_units_model = 96

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

learn_means = False
learn_covariances = False

alpha_xform = "categorical"
learn_alpha_scaling = False
normalize_covariances = False

learning_rate = 0.01

# Read MEG data
print("Reading MEG data")
prepared_data = data.Data(
    [f"/well/woolrich/shared/vrad/prepared_data/subject{i}.mat" for i in range(1, 11)]
)
n_channels = prepared_data.n_channels

# Prepare dataset
training_dataset = prepared_data.training_dataset(sequence_length, batch_size)
prediction_dataset = prepared_data.prediction_dataset(sequence_length, batch_size)

# Initialise covariances with the final HMM covariances
hmm = data.OSL_HMM("/well/woolrich/shared/vrad/hmm_fits/nSubjects-10_K-6/hmm.mat")
initial_covariances = hmm.covariances

# Build model
model = RNNGaussian(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=learn_means,
    learn_covariances=learn_covariances,
    initial_covariances=initial_covariances,
    rnn_type=rnn_type,
    n_layers_inference=n_layers_inference,
    n_layers_model=n_layers_model,
    n_units_inference=n_units_inference,
    n_units_model=n_units_model,
    dropout_rate_inference=dropout_rate_inference,
    dropout_rate_model=dropout_rate_model,
    normalization_type=normalization_type,
    alpha_xform=alpha_xform,
    learn_alpha_scaling=learn_alpha_scaling,
    normalize_covariances=normalize_covariances,
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
    learning_rate=learning_rate,
    multi_gpu=multi_gpu,
)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=n_epochs,
    save_best_after=n_epochs_annealing,
    save_filepath="/well/woolrich/shared/vrad/trained_models/ten_subjects_example/weights",
)

# Free energy = Log Likelihood + KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state mixing factors and state time courses
alpha = model.predict_states(prediction_dataset)
inf_stc = np.concatenate(states.time_courses(alpha), axis=0)
hmm_stc = np.concatenate(
    data.manipulation.trim_time_series(
        time_series=hmm.state_time_course,
        discontinuities=prepared_data.discontinuities,
        sequence_length=sequence_length,
    ),
    axis=0,
)

# Dice coefficient
print("Dice coefficient:", metrics.dice_coefficient(hmm_stc, inf_stc))

# Load preprocessed data to calculate spatial power maps
preprocessed_data = data.PreprocessedData(
    [
        f"/well/woolrich/shared/vrad/preprocessed_data/subject{i}.mat"
        for i in range(1, 11)
    ]
)
preprocessed_time_series = preprocessed_data.trim_raw_time_series(
    n_embeddings=13, sequence_length=sequence_length
)

# Compute spectra for states
f, psd, coh = spectral.state_spectra(
    data=preprocessed_time_series,
    state_mixing_factors=alpha,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
)

# Perform spectral decomposition (into 2 components) based on coherence spectra
components = spectral.decompose_spectra(coh, n_components=2)

# Calculate spatial maps
p_map, c_map = maps.state_maps(psd, coh, components)

# Save the power map for the first component as NIFTI file
# (The second component is noise)
maps.save_nii_file(
    mask_file="files/MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="files"
    + "/fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    power_map=p_map,
    filename="power_map.nii.gz",
    component=0,
)

# Delete the temporary folder holding the data
prepared_data.delete_dir()
preprocessed_data.delete_dir()
