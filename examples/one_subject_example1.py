"""Example script for running inference on real MEG data for one subject.

- The data is stored on the BMRC cluster: /well/woolrich/shared/vrad
- Uses the final covariances inferred by an HMM fit from OSL.
- Takes approximately 4 minutes to train (on compG017).
- Achieves a dice coefficient of ~0.7 (when compared to the OSL HMM state time course).
- Achieves a free energy of ~490,000.
"""

print("Importing packages")
import numpy as np
from vrad import array_ops, data
from vrad.inference import metrics, tf_ops
from vrad.models import RNNGaussian
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()
multi_gpu = True

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
learn_alpha_scaling = True
normalize_covariances = True

# Read MEG data
print("Reading MEG data")
meg_data = data.Data("/well/woolrich/shared/vrad/prepared_data/one_subject.mat")
n_channels = meg_data.n_channels

# Priors: we use the covariance matrices inferred by fitting an HMM with OSL
hmm = data.OSL_HMM("/well/woolrich/shared/vrad/hmm_fits/one_subject/hmm.mat")
covariances = hmm.covariances
means = np.zeros([n_states, n_channels], dtype=np.float32)

# Build model
model = RNNGaussian(
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
    multi_gpu=multi_gpu,
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
matched_stc, matched_inf_stc = array_ops.match_states(hmm.viterbi_path, inf_stc)
# plotting.compare_state_data(matched_stc, matched_inf_stc, filename="compare.png")

# Dice coefficient
print("Dice coefficient:", metrics.dice_coefficient(matched_stc, matched_inf_stc))

# Free energy
free_energy, ll_loss, kl_loss = model.free_energy(prediction_dataset, return_all=True)
print(f"Free energy: {ll_loss} + {kl_loss} = {free_energy}")
