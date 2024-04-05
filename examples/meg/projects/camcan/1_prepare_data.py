"""Prepare training dataset.

"""

from glob import glob

from osl_dynamics.data import Data

# Find files containing the parcel-level data
# Note, spring23 contains eyes closed resting-state data only
files = sorted(glob("/well/woolrich/projects/camcan/spring23/src/*/sflip_parc-raw.fif"))

# Load into osl-dynamics
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=16)

# Prepare TDE-PCA data
data.prepare(
    {
        "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
        "standardize": {},
    }
)

# Save a TFRecord dataset
data.save_tfrecord_dataset("training_dataset", sequence_length=400)

# Delete temporary directory
data.delete_dir()
