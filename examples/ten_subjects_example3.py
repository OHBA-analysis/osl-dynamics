"""Example script for running inference on real MEG data for ten subjects.

- The data is stored on the BMRC cluster: /well/woolrich/shared/vrad
- Data preparation is performed within V-RAD.
- Uses the BigData class to manage the data.
- Initialises the covariances with the identity matrix.
- Achieves a free energy of ~25,900,000.
"""

print("Setting up")
from vrad import array_ops, data
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

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

n_layers_inference = 1
n_layers_model = 1

normalization_type = "layer"

n_units_inference = 128
n_units_model = 128

learn_means = False
learn_covariances = True

alpha_xform = "softmax"
learn_alpha_scaling = True
normalize_covariances = True

n_initializations = 4
n_epochs_initialization = 15

# Read MEG data
print("Reading MEG data")
meg_data = data.BigData(
    [
        f"/well/woolrich/shared/vrad/preprocessed_data/subject{i}.mat"
        for i in range(1, 11)
    ]
)
meg_data.prepare(n_embeddings=13, n_pca_components=80, whiten=True)
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
    normalization_type=normalization_type,
    alpha_xform=alpha_xform,
    learn_alpha_scaling=learn_alpha_scaling,
    normalize_covariances=normalize_covariances,
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
    multi_gpu=multi_gpu,
)

model.summary()

# Prepare dataset
training_dataset = meg_data.training_dataset(sequence_length, batch_size)
prediction_dataset = meg_data.prediction_dataset(sequence_length, batch_size)

# Initialise means and covariances
model.initialize_means_covariances(
    n_initializations, n_epochs_initialization, training_dataset
)

# Train the model
print("Training model")
history = model.fit(training_dataset, epochs=n_epochs)

# Save trained model
model.save_weights("/well/woolrich/shared/vrad/trained_models/ten_subjects/example3")

# Free energy = Log Likelihood + KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state probabilities and state time course
alpha = model.predict_states(prediction_dataset)

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
