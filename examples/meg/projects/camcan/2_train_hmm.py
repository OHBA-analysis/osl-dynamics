"""Train an HMM.

"""

from sys import argv

if len(argv) != 3:
    print(
        "Please pass the number of states and run id, e.g. python 2_train_hmm.py 8 1"
    )
    exit()

n_states = int(argv[1])
run = int(argv[2])

output_dir = f"results/models/{n_states:02d}_states/run{run:02d}"

print("Importing packages")
import pickle
from osl_dynamics.data import load_tfrecord_dataset
from osl_dynamics.models.hmm import Config, Model

# Settings
config = Config(
    n_states=n_states,
    n_channels=120,
    sequence_length=400,
    learn_means=False,
    learn_covariances=True,
    learn_trans_prob=True,
    batch_size=32,
    learning_rate=0.001,
    n_epochs=10,
)

# Load training data
dataset = load_tfrecord_dataset(
    "training_dataset",
    config.batch_size,
    buffer_size=5000,
)

# Build model
model = Model(config)
model.summary()

# Initialization
model.random_state_time_course_initialization(
    dataset,
    n_init=3,
    n_epochs=1,
)

# Full training
print("Training model")
history = model.fit(dataset)

# Save the trained model
model.save(output_dir)

# Get free energy
free_energy = model.free_energy(dataset)
history["free_energy"] = free_energy

# Save training history
with open(f"{output_dir}/history.pkl", "wb") as file:
    pickle.dump(history, file)

with open(f"{output_dir}/loss.dat", "w") as file:
    file.write(f"ll_loss = {history['loss'][-1]}\n")
    file.write(f"free_energy = {free_energy}\n")
