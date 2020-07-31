"""Example script for running inference on real MEG data for one subject.

- The data is stored on the BMRC cluster: /well/woolrich/shared/vrad
- Initialises the covariances with the identity matrix. This is in constrast to
  example2.py, which uses the final covariances from an HMM fit using OSL.
- Takes approximately 4 minutes to train (on compG017).
- Achieves a dice coefficient of ~0.4 (when compared to the OSL HMM state time course).
- Achieved a free energy of ~240,000.
"""

print("Importing packages")
import numpy as np
from vrad import array_ops, data
from vrad.inference import metrics, tf_ops
from vrad.inference.models import create_model
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()

multi_gpu = True
strategy = None

# Settings
n_states = 6
sequence_length = 400
batch_size = 32

learning_rate = 0.01
clip_normalization = None

do_annealing = True
annealing_sharpness = 5

n_epochs = 200
n_epochs_annealing = 150
n_epochs_burnin = 0

dropout_rate_inference = 0.4
dropout_rate_model = 0.4

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 64
n_units_model = 64

learn_means = False
learn_covariances = True

alpha_xform = "softmax"
learn_alpha_scaling = False
normalize_covariances = False

n_initializations = 5
n_epochs_initialization = 25

# Read MEG data
print("Reading MEG data")
meg_data = data.Data("/well/woolrich/shared/vrad/prepared_data/one_subject.mat")
n_channels = meg_data.n_channels

# Use defaults for the initial means and covariances
means = None
covariances = None

# Build model
model = create_model(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=learn_means,
    learn_covariances=learn_covariances,
    initial_means=means,
    initial_covariances=covariances,
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
    n_epochs_burnin=n_epochs_burnin,
    n_initializations=n_initializations,
    n_epochs_initialization=n_epochs_initialization,
    learning_rate=learning_rate,
    clip_normalization=clip_normalization,
    multi_gpu=multi_gpu,
    strategy=strategy,
)

model.summary()

# Prepare dataset
training_dataset = meg_data.training_dataset(sequence_length, batch_size)
prediction_dataset = meg_data.prediction_dataset(sequence_length, batch_size)

# Train the model
print("Training model")
history = model.fit(training_dataset, epochs=n_epochs, verbose=0, use_tqdm=True)

# Inferred covariance matrices
int_means, inf_cov = model.get_means_covariances()
# plotting.plot_matrices(inf_cov, filename="covariances.png")

# Inferred state time courses
inf_stc = model.predict_states(prediction_dataset)
inf_stc = inf_stc.argmax(axis=1)
inf_stc = array_ops.get_one_hot(inf_stc)

# Find correspondance between state time courses
hmm = data.OSL_HMM("/well/woolrich/shared/vrad/hmm_fits/one_subject/hmm.mat")
matched_stc, matched_inf_stc = array_ops.match_states(hmm.viterbi_path, inf_stc)
# plotting.compare_state_data(matched_stc, matched_inf_stc, filename="compare.png")

# Dice coefficient
print("Dice coefficient:", metrics.dice_coefficient(matched_stc, matched_inf_stc))

# Free energy
free_energy, ll_loss, kl_loss = model.free_energy(prediction_dataset, return_all=True)
print(f"Free energy: {ll_loss} + {kl_loss} = {free_energy}")
