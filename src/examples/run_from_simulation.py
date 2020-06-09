import logging
import pathlib

import yaml
from taser import array_ops, plotting
from taser.callbacks import ComparisonCallback
from taser.data_manipulation import MEGData
from taser.inference.gmm import learn_mu_sigma
from taser.inference.models.inference_rnn import InferenceRNN
from taser.simulation import HiddenSemiMarkovSimulation
from taser.tf_ops import gpu_growth, train_predict_dataset
from taser.trainers import AnnealingTrainer

# Restrict GPU memory usage
gpu_growth()

script_dir = str(pathlib.Path(__file__).parent.absolute())

logger = logging.getLogger("TASER")
logger.setLevel(logging.INFO)

# Get all configuration options from a YAML file
logger.info("Reading configuration from 'run_from_simulation.yml'")
with open(script_dir + "/run_from_simulation.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

# Simulate data and store in a MEGData object
logger.info("Simulating data")
sim = HiddenSemiMarkovSimulation(**config["HiddenSemiMarkovSimulation"])
meg_data = MEGData(sim)
state_time_course = sim.state_time_course
n_states = sim.n_states

# Perform standard scaling/PCA
meg_data.standardize(pre_scale=True, do_pca=False, post_scale=False, n_components=1)

# Create TensorFlow Datasets
logger.info("Creating datasets")
training_dataset, prediction_dataset = train_predict_dataset(
    time_series=meg_data, **config["dataset"]
)

# Model states using a GaussianMixtureModel
logger.info("Fitting Gaussian mixture model")
covariance, means = learn_mu_sigma(
    data=meg_data,
    n_states=n_states,
    take_random_sample=20000,
    retry_attempts=5,
    learn_means=False,
    gmm_kwargs={
        "n_init": 1,
        "verbose": 2,
        "verbose_interval": 50,
        "max_iter": 10000,
        "tol": 1e-6,
    },
)

# Create model
logger.info("Creating InferenceRNN")
config["model"]["n_channels"] = meg_data.shape[1]
model = InferenceRNN(
    **config["model"], mus_initial=means, covariance_initial=covariance
)

# Create trainer and callback for checking dice coefficient
logger.info("Creating trainer")
trainer = AnnealingTrainer(model=model, **config["trainer"])
dice_callback = ComparisonCallback(trainer, state_time_course, prediction_dataset)

# Train
n_epochs = 100
logger.info(f"Training for {n_epochs} epochs")
trainer.train(training_dataset, n_epochs=n_epochs, callbacks=dice_callback)
logger.info("Training complete")

# Analysis

inf_stc = trainer.predict_latent_variable(prediction_dataset)

aligned_stc, aligned_inf_stc = array_ops.align_arrays(state_time_course, inf_stc)
matched_stc, matched_inf_stc = array_ops.match_states(aligned_stc, aligned_inf_stc)

dice_callback.plot_loss_dice()

print(f"Dice coefficient is {array_ops.dice_coefficient(matched_stc, matched_inf_stc)}")

plotting.compare_state_data(matched_inf_stc, matched_stc)

plotting.plot_state_sums(matched_stc)
plotting.plot_state_sums(matched_inf_stc)

plotting.confusion_matrix(matched_stc, matched_inf_stc)

plotting.plot_state_highlighted_data(meg_data, matched_inf_stc, n_time_points=10000)

plotting.plot_state_lifetimes(matched_inf_stc)
