"""Assign the default settings for a sensor-space inference run-time to a pickle object.
"""
import pickle

default_settings = {
    "n_states": 10,
    "sequence_length": 200,
    "batch_size": 64,
    "n_layers_inference": 1,
    "n_layers_model": 1,
    "n_units_inference": 64,
    "n_units_model": 96,
    "do_annealing": True,
    "annealing_sharpness": 10,
    "n_epochs": 500,
    "n_epochs_annealing": 300,
    "rnn_type": "lstm",
    "rnn_normalization": "layer",
    "theta_normalization": "layer",
    "dropout_rate_inference": 0.0,
    "dropout_rate_model": 0.0,
    "learn_means": False,
    "learn_covariances": True,
    "alpha_xform": "categorical",
    "learn_alpha_scaling": False,
    "normalize_covariances": False,
    "learning_rate": 0.001,  # 0.001 for SP or categorical
    "alpha_temperature": 2,
}

with open("default_TABFER_settings.pkl", "wb") as pickle_file:
    pickle.dump(default_settings, pickle_file)

with open("default_TABFER_settings.pkl", "rb") as pickle_file:
    settings = pickle.load(pickle_file)

print(settings)
