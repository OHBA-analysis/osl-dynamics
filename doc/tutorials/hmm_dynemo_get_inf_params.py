"""
HMM/DyNeMo: Get Inferred Parameters
===================================

In this tutorial we will get the inferred parameters from a dynamic network model (HMM/DyNeMo).

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/gq2zw>`_ for the expected output.
"""

#%%
# Download data and trained model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this tutorial, we'll download example data and a trained model from `OSF <https://osf.io/by2tc/>`_. Let's first download the (prepared) data we trained the model on.


import os

def get_data(name, rename):
    if rename is None:
        rename = name
    if os.path.exists(rename):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the dataset (approximately 1.7 GB)
get_data("notts_mrc_meguk_glasser_prepared", rename="prepared_data")

#%%
# Let's also download a model. In this tutorial, we will download a trained HMM, however, this can be subsituted with a DyNeMo model without any other changes being needed.


def get_model(name, rename=None):
    if rename is None:
        rename = name
    if os.path.exists(rename):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch trained_models/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Model downloaded to: {rename}"

get_model("hmm_notts_mrc_meguk_glasser", rename="results/model")

#%%
# Load Trained Model
# ^^^^^^^^^^^^^^^^^^
# First we need to load the trained model.


from osl_dynamics.models import load

model = load("results/model")

#%%
# Get the Inferred Parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Latent variables
# ****************
# Both the HMM and DyNeMo refer to the latent variable as `alpha`. In the HMM, `alpha` corresponds to the state probability time courses, and in DyNeMo, `alpha` corresponds to the mixing coefficient time courses. We can get these using the `get_alpha` method by passing the (prepared) training data.
#
# Let's first load the training data.


from osl_dynamics.data import Data

data = Data("prepared_data", n_jobs=4)

#%%
# Now we can  get the `alpha` for each subject.


alpha = model.get_alpha(data)

#%%
# `alpha` is a list of numpy arrays, one for each subject.


import pickle

os.makedirs("results/inf_params", exist_ok=True)
pickle.dump(alpha, open("results/inf_params/alp.pkl", "wb"))

#%%
# Observation model
# *****************
# We can get the inferred state/mode means and covariances with the `get_means_covariances` method. Note, if we passed `learn_means=False` in the config, the means will be zero.


means, covs = model.get_means_covariances()

np.save("results/inf_params/means.npy", means)
np.save("results/inf_params/covs.npy", covs)

