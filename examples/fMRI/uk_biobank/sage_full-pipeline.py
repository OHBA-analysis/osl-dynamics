"""A script to run SAGE on the UKB data. This example is composed of three main steps:

1) Running SAGE on the subset of UKB group-level data.
2) Running SAGE on the subject-level data using dual-estimation technique.
3) Predicting a non-imaging variable (age) using subject-level FCs.

This should give a prediction correlation around 0.4.
"""

print("Setting up")
import os, random, scipy
import numpy as np
from glob import glob
from scipy import io
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import ElasticNetCV

from osl_dynamics import analysis, data, inference
from osl_dynamics.models.sage import Config, Model
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.utils import plotting
from osl_dynamics.models.dynemo_obs import Config as DynemoConfig
from osl_dynamics.models.dynemo_obs import Model as DynemoModel

# GPU settings
tf_ops.gpu_growth()

# Output id and folders
output_dir = "temp"
os.makedirs(output_dir, exist_ok=True)

# Settings
config = Config(
    n_modes=12,
    n_channels=25,
    sequence_length=100,
    inference_n_units=64,
    inference_n_layers=2,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    discriminator_n_units=64,
    discriminator_normalization="layer",
    learn_means=False,
    learn_covariances=True,
    batch_size=32,
    learning_rate=0.0003,
    n_epochs=50,
)

# ------------------------------- #
#           UKB dataset           #
# ------------------------------- #

# Training (Subset) data
# For Training: Dataset containing a subset of resting-state fMRI data (location of training data in BMRC)
print("Reading resting-fMRI data")
dataset_dir = "/gpfs3/well/win-biobank/projects/imaging/data/data3/subjectsAll/2*/fMRI/rfMRI_25.dr/dr_stage1.txt"

files = random.sample(glob(dataset_dir), len(glob(dataset_dir)))[
    :2000
]  # subset of 3k subjects for training

training_data = data.Data(
    [path for path in glob(dataset_dir) if path in files],
    load_memmaps=False,
    keep_memmaps_on_close=True,
)

training_data.prepare(load_memmaps=False)

# Create tensorflow datasets for training the model
training_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
    concatenate=True,
)

# Prediction (full) data
# For Prediction: Dataset containing full resting-state fMRI data
prediction_files = [f"{path}" for path in glob(dataset_dir)]
prediction_data = data.Data(
    prediction_files,
    load_memmaps=False,
    keep_memmaps_on_close=True,
)

# Standardise the data
prediction_data.prepare(load_memmaps=False)
prediction_dataset = prediction_data.dataset(
    config.sequence_length, config.batch_size, shuffle=False, concatenate=False
)

# ------------------------- #
# Build the main SAGE model #
# ------------------------- #
model = Model(config)

# ------------------------- -------------#
# Train on the Training dataset (Subset) #
# ------------------------- -------------#
print("Training SAGE model")
history = model.fit(training_dataset)


# ----------------------------------------------------- #
# State-level Analysis on the Prediction dataset (Full) #
# ----------------------------------------------------- #

# Temporal Analysis: State-time courses

# Alpha time course for each subject
a = model.get_alpha(prediction_dataset, concatenate=False)

# Order modes with respect to mean alpha values
mean_a = np.mean(np.concatenate(a), axis=0)
order = np.argsort(mean_a)[::-1]

mean_a = mean_a[order]
a = [alp[:, order] for alp in a]

print("mean_a:", mean_a)

plotting.plot_alpha(a[0], filename=f"{output_dir}/alpha.png")

# Inferred means and covariances
means, covariances = model.get_means_covariances()
means = means[order]
covariances = covariances[order]

# Trace of mode covariances
tr_covariances = np.trace(covariances, axis1=1, axis2=2)

# Normalised weighted alpha
a_NW = [tr_covariances[np.newaxis, ...] * alp for alp in a]
a_NW = [alp_NW / np.sum(alp_NW, axis=1)[..., np.newaxis] for alp_NW in a_NW]

plotting.plot_alpha(a_NW[0], filename=f"{output_dir}/alpha_NW.png")

# Mean normalised weighted alpha
mean_a_NW = np.mean(np.concatenate(a_NW), axis=0)

print("mean_a_NW:", mean_a_NW)

# Create a state time course from the alpha
argmax_a_NW = inference.modes.argmax_time_courses(a_NW)

plotting.plot_alpha(
    argmax_a_NW[0],
    filename=f"{output_dir}/argmax_alpha_NW.png",
)

# State statistics

# State FO
fo = np.array(inference.modes.fractional_occupancies(argmax_a_NW))
mean_fo = np.mean(fo, axis=0)
print("mean_fo:", mean_fo)

# State lifetimes
lt = inference.modes.lifetimes(argmax_a_NW)
mean_lt = np.array([np.mean(lifetimes) for lifetimes in lt])
print("mean_lt:", mean_lt)

# State Intervals
intv = inference.modes.intervals(argmax_a_NW)
mean_intv = np.array([np.mean(interval) for interval in intv])
print("mean_intv:", mean_intv)

# Plotting FO, lifetimes and Intervals
plotting.plot_violin(
    [l for l in lt],
    y_range=[0, 20],
    x_label="Mode",
    y_label="Lifetime",
    filename=f"{output_dir}/lt.png",
)
plotting.plot_violin(
    intv,
    y_range=[0, 200],
    x_label="Mode",
    y_label="Interval (s)",
    filename=f"{output_dir}/intv.png",
)
plotting.plot_violin(
    [f for f in fo.T],
    x_label="Mode",
    y_label="Fractional Occupancy",
    filename=f"{output_dir}/fo.png",
)


plotting.plot_matrices(covariances, filename=f"{output_dir}/cov.png")
plotting.plot_matrices(means, filename=f"{output_dir}/mean.png")

# ------------------------------------------------------- #
# Subject-level Analysis on the Prediction dataset (Full) #
# ------------------------------------------------------- #
means, covariances = model.get_means_covariances()

config = DynemoConfig(
    n_modes=12,
    n_channels=25,
    sequence_length=100,
    learn_means=False,
    learn_covariances=True,
    batch_size=32,
    learning_rate=0.01,
    n_epochs=60,
)

obs_model = DynemoModel(config)
obs_model.set_covariances(covariances)

covariances_subject = []
subject_ids = []

for idx in range(len(prediction_files)):
    subject_data = data.Data(
        prediction_files[idx],
        load_memmaps=False,
        keep_memmaps_on_close=True,
    )
    subject_data.prepare(load_memmaps=False)
    subject_dataset = subject_data.dataset(
        config.sequence_length,
        config.batch_size,
        shuffle=False,
        alpha=[a[idx]],  # alphas are in order wrt to the prediction_files
    )
    IDs = prediction_files[idx][64:72]
    print("Training Subject-Level model - Subject {}".format(IDs))

    try:
        history = obs_model.fit(subject_dataset, epochs=config.n_epochs, verbose=0)
        # Inferred covariances
        covariances_temp = obs_model.get_covariances()
    except:  # if there is an error in inference - equalize subject-level to group-level
        covariances_temp = covariances

    covariances_subject.append(covariances_temp)
    subject_ids.append(IDs)

np.save(f"{output_dir}/covariances.npy", covariances_subject, allow_pickle=True)
np.save(f"{output_dir}/ids.npy", subject_ids, allow_pickle=True)

# --------------------------------------#
# Subject-level Behavioural Variability #
# --------------------------------------#
# Loading the non-imaging variable age (i.e., age is just used an example here, any non-imaging variable can be loaded)
age = io.loadmat("/gpfs3/well/win-fmrib-analysis/users/nhx531/ukb_46k/age.mat")["age"]

# Loading the original IDs of the subject that are in accordance with the order in which non-imaging variable (i.e., age) is loaded
ids_orig = io.loadmat("/gpfs3/well/win-fmrib-analysis/users/nhx531/ukb_46k/ids.mat")[
    "ids"
]

# Loading the subject-level state fcs and subject IDs in accordance with which the fcs are loaded
covariances_subject = np.load(
    f"/well/win/users/nhx531/osl-dynamics/examples/fMRI/{output_dir}/covariances.npy"
)
subject_ids = np.load(
    f"/well/win/users/nhx531/osl-dynamics/examples/fMRI/{output_dir}/ids.npy"
)

intersect, ind_a, ind_b = np.intersect1d(subject_ids, ids_orig, return_indices=True)
y, X = age[ind_b], covariances_subject[ind_a]
X = X.reshape(*X.shape[:-3], -1)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

# define model
ratios = np.arange(0, 1, 0.1)
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0]
model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1, precompute=False)

# Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model using the training sets
model.fit(X_train, y_train)

# summarize chosen configuration
print("alpha: %f" % model.alpha_)
print("l1_ratio_: %f" % model.l1_ratio_)

# Make predictions using the testing set
predict_y = model.predict(X_test)
print(
    pearsonr(np.squeeze(y_test), np.squeeze(predict_y))
)  # prediction correlation on the testing set.
