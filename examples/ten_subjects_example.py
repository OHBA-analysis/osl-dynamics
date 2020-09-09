"""Example script for running inference on real MEG data for ten subjects.

- The data is stored on the BMRC cluster: /well/woolrich/shared/vrad
"""

print("Setting up")
from vrad import array_ops, data
from vrad.analysis import spectral, maps
from vrad.inference import metrics, states, tf_ops
from vrad.models import RNNGaussian
from vrad.utils import plotting

# GPU settings
tf_ops.suppress_messages()
tf_ops.gpu_growth()

# Settings
n_states = 6
sequence_length = 800
batch_size = 32

do_annealing = True
annealing_sharpness = 5

n_epochs = 250
n_epochs_annealing = 125

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 128
n_units_model = 128

learn_means = False
learn_covariances = True

alpha_xform = "softmax"
learn_alpha_scaling = True
normalize_covariances = True

n_initializations = 4
n_epochs_initialization = 35

# Read MEG data
print("Reading MEG data")
meg_data = data.Data(
    [f"/well/woolrich/shared/vrad/preprocessed_data/subject{i}.mat" for i in range(10)]
)
n_channels = meg_data.n_channels

# Build model
model = RNNGaussian(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=learn_means,
    learn_covariances=learn_covariances,
    n_layers_inference=n_layers_inference,
    n_layers_model=n_layers_model,
    n_units_inference=n_units_inference,
    n_units_model=n_units_model,
    dropout_rate_inference=dropout_rate_inference,
    dropout_rate_model=dropout_rate_model,
    alpha_xform=alpha_xform,
    learn_alpha_scaling=learn_alpha_scaling,
    normalize_covariances=normalize_covariances,
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
)

model.summary()

# Prepare dataset
training_dataset = meg_data.training_dataset(sequence_length, batch_size)
prediction_dataset = meg_data.prediction_dataset(sequence_length, batch_size)

# Initialise means and covariances
model.initialize_means_covariances(
    n_initializations, n_epochs_initialization, training_dataset, use_tqdm=True
)

# Train the model
print("Training model")
history = model.fit(training_dataset, epochs=n_epochs, use_tqdm=True)

# Inferred state probabilities and state time course
alpha = model.predict_states(prediction_dataset)
stc = states.time_courses(alpha)

# Find correspondance between HMM and inferred state time courses
hmm = data.OSL_HMM("/well/woolrich/shared/vrad/hmm_fits/ten_subjects.mat")
matched_hmm_stc, *matched_inf_stc = states.match_states(hmm.state_time_course, *stc)

# Dice coefficient
for miv in matched_inf_stc:
    print("Dice coefficient:", metrics.dice_coefficient(matched_hmm_stc, miv))

# Free energy = Log Likelihood + KL Divergence
for subject_dataset in prediction_dataset:
    free_energy = model.free_energy(subject_dataset)
    print(f"Free energy: {free_energy}")

# Compute spectra for states
f, psd, coh = spectral.state_spectra(
    data=meg_data.raw_data,
    state_probabilities=alpha,
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
