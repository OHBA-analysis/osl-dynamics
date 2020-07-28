"""Example script for running inference on real MEG data for ten subjects.

- The data is stored on the BMRC cluster: /well/woolrich/shared/vrad
- Samples from a BasicHMMSimulation to initialise the covariances.
"""

print("Importing packages")
import mat73
import numpy as np
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from vrad import array_ops, data
from vrad.inference import metrics, priors, tf_ops
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
n_epochs_burnin = 30

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

# Read MEG data
print("Reading MEG data")
meg_data = data.Data("/well/woolrich/shared/vrad/prepared_data/ten_subjects.mat")
n_channels = meg_data.n_channels

# Priors
means, covariances = priors.hmm(
    meg_data,
    n_states,
    stay_prob=0.95,
    learn_means=learn_means,
    n_initialisations=10,
    simulation="basic",
)

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
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
    n_epochs_burnin=n_epochs_burnin,
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
history = model.fit(
    training_dataset,
    callbacks=[TqdmCallback(tqdm_class=tqdm, verbose=0)],
    epochs=n_epochs,
    verbose=0,
)

# Inferred covariance matrices
int_means, inf_cov = model.get_means_covariances()

# Plot covariance matrices
# plotting.plot_matrices(inf_cov, filename="covariances.png")

# Inferred state time courses
inf_stc = model.predict_states(prediction_dataset)
inf_stc = inf_stc.argmax(axis=1)
inf_stc = array_ops.get_one_hot(inf_stc)

# State time course from HMM
hmm = data.OSL_HMM("/well/woolrich/shared/vrad/hmm_fits/ten_subjects/hmm.mat")
hmm_stc = hmm.viterbi_path

# Find correspondance between state time courses
matched_stc, matched_inf_stc = array_ops.match_states(hmm_stc, inf_stc)

# Compare state time courses
# plotting.compare_state_data(matched_stc, matched_inf_stc, filename="compare.png")

# Dice coefficient
dc = metrics.dice_coefficient(matched_stc, matched_inf_stc)
print(f"Dice coefficient: {dc}")

# Free energy
free_energy, ll_loss, kl_loss = model.free_energy(prediction_dataset, return_all=True)
print(f"Free energy: {ll_loss} + {kl_loss} = {free_energy}")
