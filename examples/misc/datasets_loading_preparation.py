"""Example script for loading data from a file and preparing it for training.

"""

print("Setting up")
from osl_dynamics.data import Data

# Load real MEG data
# - String can be the path to a .npy or .mat file, or the path to a folder containing .npy files
# - N.b. a list of strings can be passed.
meg_data = Data(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject1.mat"
)

# Prepare the preprocessed data
# - This includes time-delay embedding, PCA and standardisation
meg_data.prepare(n_embeddings=15, n_pca_components=80)

print(meg_data)

# Hyperparameters for training a model
sequence_length = 200
batch_size = 32

# Create tensorflow datasets for training and model evaluation
training_dataset, validation_dataset = meg_data.dataset(
    sequence_length,
    batch_size,
    shuffle=True,
    validation_split=0.1,  # if this is not passed, only the training_dataset is returned
)
prediction_dataset = meg_data.dataset(sequence_length, batch_size, shuffle=False)

print("Training dataset:")
print(training_dataset)

# We can save the prepared data
meg_data.save("prepared_data")
