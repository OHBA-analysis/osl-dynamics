"""Train a DyNeMo model on time-delay embedded data and calculate mode spectra.

"""

from osl_dynamics import run_pipeline

# Settings
#
# If you run in memory issues you can try reducing the batch_size
# first, then also try reducing the sequence length.
config = """
    data_prep:
      n_embeddings: 15
      n_pca_components: 80
    dynemo:
      n_modes: 8
      sequence_length: 200
      inference_n_units: 64
      inference_normalization: layer
      model_n_units: 64
      model_normalization: layer
      learn_alpha_temperature: True
      initial_alpha_temperature: 1.0
      learn_means: False
      learn_covariances: True
      do_kl_annealing: True
      kl_annealing_curve: tanh
      kl_annealing_sharpness: 10
      n_kl_annealing_epochs: 20
      batch_size: 64
      learning_rate: 0.01
      n_epochs: 40
      n_init: 3
      n_init_epochs: 1
    regression_spectra:
      window_length: 1000
      sampling_frequency: 250
      frequency_range: [1, 45]
      n_sub_windows: 8
      return_coef_int: True
"""

# Path to directory containing training data
#
# This directory should contain individual subject files containing
# continuous time series data in (time x channels) format. These
# files should be called: subject0.npy, subject1.npy, ...
inputs = "training_data"

# Run analysis
#
# This creates the following directories in savedir:
# - /trained_model, which contains the trained model.
# - /inf_params, which contains the inferred parameters.
# - /spectra, which contains the power/coherence spectra.
run_pipeline(config, inputs, savedir="tde-dyn_results")
