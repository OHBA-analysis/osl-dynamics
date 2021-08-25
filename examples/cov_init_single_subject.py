"""Example script for training VRAD on a single subject to use the inferred covariances
as initialization for full training.

- Picks 1 of 46 subjects to train on at randomly.
- Trains VRAD using identity matrices to initialize the covariances.
- This is repeated 10 times.
- Then save the inferred covariances from the fit with the best free energy.
"""

print("Setting up")
import numpy as np
from vrad.data import Data
from vrad.inference import tf_ops
from vrad.models import Config, Model


# GPU settings
tf_ops.gpu_growth()

# Hyperparameters
config = Config(
    n_states=10,
    n_channels=80,
    sequence_length=200,
    inference_rnn="lstm",
    inference_n_units=64,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_units=64,
    model_normalization="layer",
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    n_init=10,
    n_init_epochs=20,
    batch_size=32,
    learning_rate=0.0025,
    gradient_clip=0.5,
    n_epochs=200,
    multi_gpu=False,
)

# Build model
model = Model(config)
model.summary()

# Choose subjects at random
n_runs = 10
subjects_used = np.random.choice(range(1, 46), n_runs, replace=False)

# Train the model a few times and keep the best one
best_loss = np.Inf
losses = []
for subject in subjects_used:
    print("Using subject", subject)

    # Load data
    training_data = Data(
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/prepared_data/subject{subject}.mat",
    )
    training_dataset = training_data.dataset(
        config.sequence_length, config.batch_size, shuffle=True
    )

    # Reset the model weights and train
    model.reset_weights()
    model.compile()
    history = model.fit(training_dataset, epochs=config.n_epochs, use_tqdm=True)
    loss = history.history["loss"][-1]
    losses.append(loss)
    print(f"loss: {loss}")

    if loss < best_loss:
        best_loss = loss
        subject_chosen = subject
        best_weights = model.get_weights()

# Restore best model
model.set_weights(best_weights)

# Get initialisation for covariances
init_cov = model.get_covariances()
np.save(f"init_cov.npy", init_cov)

# Save which subjects were used and which was chosen
with open(f"init_cov_runs.dat", "w") as file:
    file.write(f"used: {' '.join([str(s) for s in subjects_used])}\n")
    file.write(f"chose: {subject_chosen}\n")
    file.write(f"losses: {' '.join([str(s) for s in losses])}\n")

# Delete the temporary folder holding the data
training_data.delete_dir()
