"""Trains a DyNeMo model on data generated with prepare_data.py

"""

import pickle
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.dynemo import Config, Model

# Directory containing the prepared data
data_dir = "data"

# Directory to save the model to
model_dir = "model"

# GPU settings
tf_ops.gpu_growth()

# Settings
# - If you run out of memory you can reduce the sequence_length
#   and/or batch_size.
# - You might want to play around with the learning rate and
#   number of epochs.
# - Pick the parameters what give you the best final training loss.
# - You also want to show your results are robust to the choice
#   for n_modes.
config = Config(
    n_modes=8,
    n_channels=80,
    sequence_length=200,
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
    n_kl_annealing_epochs=25,
    batch_size=32,
    learning_rate=0.01,
    n_epochs=50,
)

# Load the prepared data
# - pass the path to the directory created by Data.save() in prepare_data.py
data = Data(data_dir)

# Build the model
model = Model(config)
model.summary()

# Train the model
print("Training model")
history = model.fit(training_data)

# Save the trained model
model.save(model_dir)

# Save the training history (contains the loss as a function of epochs)
pickle.dump(history, open(model_dir + "/history.pkl", "wb"))
