"""Example script for training an HMM using a simplified user interface.

"""

from osl_dynamics import run_pipeline

# Settings
config = """
    data_prep:
      n_embeddings: 15
      n_pca_components: 80
    hmm:
      n_states: 8
      n_channels: 80
      sequence_length: 2000
      learn_means: False
      learn_covariances: True
      batch_size: 16
      learning_rate: 0.01
      n_epochs: 20
    multitaper_spectra:
      sampling_frequency: 250
      time_half_bandwidth: 4
      n_tapers: 7
      frequency_range: [1, 45]
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
run_pipeline(config, inputs, savedir="output")
