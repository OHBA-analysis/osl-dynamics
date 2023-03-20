"""Train an AE-HMM.

"""

from osl_dynamics import run_pipeline

# Settings
#
# These settings have been tested on multiple independent datasets
# (in source space) and seem to work well.
#
# If you run in memory issues you can try reducing the batch_size
# first, then also try reducing the sequence length.
config = """
    data_prep:
      amplitude_envelope: True
      n_window: 6
    hmm:
      n_states: 8
      sequence_length: 2000
      learn_means: True
      learn_covariances: True
      batch_size: 16
      learning_rate: 0.01
      n_epochs: 20
      n_init: 3
      n_init_epochs: 1
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
run_pipeline(config, inputs, savedir="ae_results")
