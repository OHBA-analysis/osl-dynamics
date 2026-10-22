"""Train an HMM with a given initialization method.

Run several times (different run ids) for each method, then compare
with 3_summarise.py.
"""

import os
import pickle
import time
from sys import argv

if len(argv) != 3:
    print(
        "Please pass the initialization method and run id, "
        "e.g. python 1_train_hmm.py random_state_time_course 1"
    )
    exit()

method = argv[1]  # random_state_time_course or random_subset
run = int(argv[2])

# Directory containing the TFRecord dataset (written with
# Data.save_tfrecord_dataset, see e.g. meg_analysis/projects/camcan)
tfrecord_dir = "dataset"

output_dir = f"results/{method}/run{run:02d}"
os.makedirs(output_dir, exist_ok=True)

print("Importing packages")
import numpy as np
from osl_dynamics.data import load_tfrecord_dataset, load_tfrecord_config
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import set_random_seed

# The run id is the seed, so every method sees the same data order and
# the only difference between runs of a method is the initialization
set_random_seed(run)

tfrecord_config = load_tfrecord_config(tfrecord_dir)

config = Config(
    n_states=8,
    n_channels=tfrecord_config["n_channels"],
    sequence_length=tfrecord_config["sequence_length"],
    learn_means=False,
    learn_covariances=True,
    batch_size=32,
    learning_rate=0.001,
    n_epochs=15,
    init_method=method,
    n_init=3,
    n_init_epochs=1,
)

dataset = load_tfrecord_dataset(tfrecord_dir, config.batch_size, buffer_size=1000)

model = Model(config)
model.summary()

# Initialization
start = time.time()
init_history = model.initialization(dataset)
init_time = time.time() - start
init_covs = model.get_covariances()

# Full training
print("Training model")
start = time.time()
history = model.fit(dataset)
train_time = time.time() - start

model.save(f"{output_dir}/model")

# Free energy on the full dataset
free_energy = model.free_energy(dataset)

# Inferred parameters
covs = model.get_covariances()
pred_dataset = load_tfrecord_dataset(
    tfrecord_dir, config.batch_size, shuffle=False, concatenate=False
)
alpha = model.get_alpha(pred_dataset)

np.save(f"{output_dir}/init_covs.npy", init_covs)
np.save(f"{output_dir}/covs.npy", covs)
pickle.dump(alpha, open(f"{output_dir}/alp.pkl", "wb"))
pickle.dump(init_history, open(f"{output_dir}/init_history.pkl", "wb"))
pickle.dump(history, open(f"{output_dir}/history.pkl", "wb"))
pickle.dump(
    {
        "method": method,
        "run": run,
        "free_energy": free_energy,
        "init_loss": init_history["loss"][-1] if init_history else None,
        "final_loss": history["loss"][-1],
        "init_time": init_time,
        "train_time": train_time,
    },
    open(f"{output_dir}/summary.pkl", "wb"),
)
print(f"free_energy = {free_energy}, init_time = {init_time:.0f} s")
