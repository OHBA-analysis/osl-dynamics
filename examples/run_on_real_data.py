"""Example script for running inference on real MEG data.

- The data is stored on the BMRC cluster: /well/woolrich/shared/vrad
- Takes approximately 4 minutes to train (on compG017).
- Achieves a dice coefficient of ~0.7 (when compared to the OSL HMM state time course).
- Line 121 can be uncommented to produce a plot of the simulated and inferred
  state time courses for comparison.
"""

print("Importing packages")
import numpy as np
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from vrad import array_ops, data
from vrad.inference import metrics, tf_ops
from vrad.inference.models.variational_rnn_autoencoder import create_model
from vrad.utils import plotting
import mat73

# GPU settings
tf_ops.gpu_growth()

multi_gpu = False
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

n_units_inference = 64
n_units_model = 64

learn_means = False
learn_covariances = True

activation_function = "softmax"

# Read MEG data
print('Reading MEG data')
meg_data = data.Data('/well/woolrich/shared/vrad/prepared_data/one_subject.mat')
n_channels = meg_data.shape[1]

# Priors: we use the covariance matrices inferred by fitting an HMM with OSL
covariances = mat73.loadmat('/well/woolrich/shared/vrad/hmm_fits/one_subject/Covs.mat')
covariances = covariances['Covs'].astype(np.float32)
means = np.zeros([n_states, n_channels], dtype=np.float32)

# Prepare dataset
training_dataset, prediction_dataset = tf_ops.train_predict_dataset(
    time_series=meg_data, sequence_length=sequence_length, batch_size=batch_size,
)

# Build autoecoder model
rnn_vae = create_model(
    n_channels=n_channels,
    n_states=n_states,
    sequence_length=sequence_length,
    learn_means=learn_means,
    learn_covariances=learn_covariances,
    initial_mean=means,
    initial_covariances=covariances,
    n_units_inference=n_units_inference,
    n_units_model=n_units_model,
    dropout_rate_inference=dropout_rate_inference,
    dropout_rate_model=dropout_rate_model,
    activation_function=activation_function,
    do_annealing=do_annealing,
    annealing_sharpness=annealing_sharpness,
    n_epochs_annealing=n_epochs_annealing,
    n_epochs_burnin=n_epochs_burnin,
    learning_rate=learning_rate,
    clip_normalization=clip_normalization,
    multi_gpu=multi_gpu,
    strategy=strategy,
)

rnn_vae.summary()

# Train the model
print("Training model")
history = rnn_vae.fit(
    training_dataset,
    callbacks=[TqdmCallback(tqdm_class=tqdm, verbose=0)],
    epochs=n_epochs,
    verbose=0,
)

# Inferred state probabilities
inf_stc = np.concatenate(rnn_vae.predict(prediction_dataset)["m_theta_t"])

# Read file containing state probabilities inferred by OSL
hmm_stc = mat73.loadmat('/well/woolrich/shared/vrad/hmm_fits/one_subject/gamma.mat')
hmm_stc = hmm_stc['gamma']

# Hard classify
inf_stc = inf_stc.argmax(axis=1)
hmm_stc = hmm_stc.argmax(axis=1)

# One hot encode
inf_stc = array_ops.get_one_hot(inf_stc)
hmm_stc = array_ops.get_one_hot(hmm_stc)

# Find correspondance between state time courses
matched_stc, matched_inf_stc = array_ops.match_states(hmm_stc, inf_stc)

# Compare state time courses
#plotting.compare_state_data(matched_stc, matched_inf_stc, filename="compare.png")

print("Dice coefficient:", metrics.dice_coefficient(matched_stc, matched_inf_stc))
