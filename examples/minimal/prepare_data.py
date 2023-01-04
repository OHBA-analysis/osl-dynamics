"""Prepare data for training an osl-dynamics model.

"""

from osl_dynamics.data import Data

# Directory to save the prepared data to
data_dir = "data"

# Load your input data (this can be sensor-level or source reconstructed data)
#
# The data can be passed as numpy (.npy) files containing 2D numpy arrays
# or matlab (.mat) files with and 'X' field containing the data.
#
# The data must be in the format (time x channels). If you have the data
# in (channels x time) format, you can pass time_axis_first=False to Data
files = [f"subject{i}.npy" for i in range(5)]
data = Data(files)

# Prepare the data
#
# In this example we will perform time-delay embedding and principal
# component analysis
#
# Alternatively, we could prepare amplitude envelope data. See the Data.prepare()
# method for various the options.
data.prepare(n_embeddings=15, n_pca_components=80)

# Save the data (this will create a new directory containined the prepared data)
data.save(data_dir)
