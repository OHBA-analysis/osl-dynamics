"""Example script for running inference on real MEG data for ten subjects.

- The data is stored on the BMRC cluster: /well/woolrich/shared/vrad
- Uses the final covariances inferred by an HMM fit from OSL.
- Achieves a dice coefficient of ~0.4 (when compared to the OSL HMM state time course).
- Achieves a free energy of ~26,600,000.
"""

print("Setting up")
from vrad import data
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

normalization_type = "layer"

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 128
n_units_model = 128

learn_means = False
learn_covariances = True

alpha_xform = "softmax"
learn_alpha_scaling = True
normalize_covariances = True

# Read MEG data
print("Reading MEG data")
prepared_data = data.Data("/well/woolrich/shared/vrad/prepared_data/ten_subjects.mat")
n_channels = prepared_data.n_channels

# Priors: we use the covariance matrices inferred by fitting an HMM with OSL
hmm = data.OSL_HMM("/well/woolrich/shared/vrad/hmm_fits/ten_subjects.mat")
initial_covariances = hmm.covariances

# Build model
model = RNNGaussian(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=learn_means,
    learn_covariances=learn_covariances,
    initial_covariances=initial_covariances,
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
training_dataset = prepared_data.training_dataset(sequence_length, batch_size)
prediction_dataset = prepared_data.prediction_dataset(sequence_length, batch_size)

print("Training model")
history = model.fit(training_dataset, epochs=n_epochs)

# Save trained model
model.save_weights("/well/woolrich/shared/vrad/trained_models/ten_subjects/example1")

# Free energy = Log Likelihood + KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state probabilities and state time course
alpha = model.predict_states(prediction_dataset)[0]
stc = states.time_courses(alpha)

# Find correspondance between HMM and inferred state time courses
matched_hmm_stc, matched_inf_stc = states.match_states(hmm.state_time_course, stc)

# Dice coefficient
print("Dice coefficient:", metrics.dice_coefficient(matched_hmm_stc, matched_inf_stc))
