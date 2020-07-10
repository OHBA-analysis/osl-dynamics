"""Example script for running inference on real MEG data.

- The data is stored on the BMRC cluster: /well/woolrich/shared/vrad
- Takes approximately 4 minutes to train (on compG017).
- Achieves a dice coefficient of ~0.7 (when compared to the OSL HMM state time course).
- Line 106, 127, 128 can be uncommented to produce a plot of the inferred
  covariances and state time courses.
"""
import mat73
import numpy as np
from tensorflow.python.data import Dataset
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from vrad import array_ops, data
from vrad.inference import metrics, tf_ops
from vrad.inference.models.variational_rnn_autoencoder import create_model
from vrad.utils import plotting

print("Importing packages")

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

# Read MEG data
print("Reading MEG data")
meg_data = data.Data("/well/woolrich/shared/vrad/prepared_data/one_subject.mat")
n_channels = meg_data.n_channels

# Priors: we use the covariance matrices inferred by fitting an HMM with OSL
covariances = mat73.loadmat("/well/woolrich/shared/vrad/hmm_fits/one_subject/Covs.mat")
covariances = covariances["Covs"].astype(np.float32)
means = np.zeros([n_states, n_channels], dtype=np.float32)

# Prepare dataset
training_dataset = (
    meg_data.dataset(sequence_length).batch(32).cache().shuffle(10000).prefetch(-1)
)
prediction_dataset = (
    Dataset.from_tensor_slices(meg_data.time_series)
    .batch(sequence_length, drop_remainder=True)
    .batch(batch_size, drop_remainder=True)
)

# Build autoencoder model
model = create_model(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=learn_means,
    learn_covariances=learn_covariances,
    initial_mean=means,
    initial_covariances=covariances,
    n_layers_inference=n_layers_inference,
    n_layers_model=n_layers_model,
    n_units_inference=n_units_inference,
    n_units_model=n_units_model,
    dropout_rate_inference=dropout_rate_inference,
    dropout_rate_model=dropout_rate_model,
    alpha_xform=alpha_xform,
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

# Train the model
print("Training model")
history = model.fit(
    training_dataset,
    callbacks=[TqdmCallback(tqdm_class=tqdm, verbose=0)],
    epochs=n_epochs,
    verbose=0,
)

# Inferred covariance matrices
int_means, inf_cov = model.state_means_covariances()

# Plot covariance matrices
# plotting.plot_matrices(inf_cov, filename="covariances.png")

# Inferred state probabilities
inf_stc = model.predict_states(prediction_dataset)

# Read file containing state probabilities inferred by OSL
hmm_stc = mat73.loadmat("/well/woolrich/shared/vrad/hmm_fits/one_subject/gamma.mat")
hmm_stc = hmm_stc["gamma"]

# Hard classify
inf_stc = inf_stc.argmax(axis=1)
hmm_stc = hmm_stc.argmax(axis=1)

# One hot encode
inf_stc = array_ops.get_one_hot(inf_stc)
hmm_stc = array_ops.get_one_hot(hmm_stc)

# Find correspondance between state time courses
matched_stc, matched_inf_stc = array_ops.match_states(hmm_stc, inf_stc)

# Compare state time courses
# plotting.compare_state_data(matched_stc, matched_inf_stc, filename="compare.png")
plotting.plot_state_time_courses(matched_stc, filename="stc.png")
plotting.plot_state_time_courses(matched_inf_stc, filename="inf_stc.png")

print("Dice coefficient:", metrics.dice_coefficient(matched_stc, matched_inf_stc))
