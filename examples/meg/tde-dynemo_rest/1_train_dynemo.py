"""Train DyNeMo on time-delay embedded/PCA data.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of modes and run id, e.g. python 1_train_dynemo.py 8 1")
    exit()
n_modes = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")
import pickle
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model

#%% Load data

data = Data("training_data/networks", n_jobs=8)
methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
    "standardize": {},
}
data.prepare(methods)

#%% Setup model

config = Config(
    n_modes=n_modes,
    n_channels=data.n_channels,
    sequence_length=100,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=10,
    batch_size=64,
    learning_rate=1e-3,
    n_epochs=20,
)
model = Model(config)
model.summary()
model.set_regularizers(data)

#%% Training

init_history = model.random_subset_initialization(data, n_init=5, n_epochs=2, take=0.5)
history = model.fit(data)

# Save trained model
model_dir = f"results/{n_modes}_modes/run{run:02d}/model"
model.save(model_dir)

# Calculate the free energy
free_energy = model.free_energy(data)
history["free_energy"] = free_energy

# Save training history and free energy
pickle.dump(init_history, open(f"{model_dir}/init_history.pkl", "wb"))
pickle.dump(history, open(f"{model_dir}/history.pkl", "wb"))
