"""
Preparing fMRI Data
===================

In this tutorial we discuss how to prepare fMRI data. This tutorial covers:

1. Preparing group-ICA data
2. Preparing anatomically parcellated data
3. Saving and loading prepared data

fMRI data needs to be preprocessed in a particular way to train a dynamic network model (HMM/DyNeMo), see this `paper <https://www.sciencedirect.com/science/article/pii/S1053811922001550>`_ for further details. This is make sure we infer good dynamics (state switching) in the data. We advise training a model on the group-ICA time courses (which are prepared by standardizing the data) or using an anatomical parcellation (which is prepared using a full rank PCA).

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/p6f9c>`_ for the expected output.
"""

#%%
# Group-ICA time courses
# ^^^^^^^^^^^^^^^^^^^^^^
# **This is the recommended data to train an HMM/DyNeMo on.**
#
# The first step is to load the group-ICA (source) time courses into osl-dynamics using the Data object, see the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details. In this tutorial, we will simulate some random data (for 2 subjects). You should substitute this with your preprocessed fMRI data.


import numpy as np
from osl_dynamics.data import Data

X = [np.random.normal(size=(490,25)), np.random.normal(size=(490,25))]

data = Data(X)
print(data)

#%%
# To prepare the data, all we need to do is standardize it. Let's use the `Data.prepare` method to do this.


data.prepare({"standardize": {}})

#%%
# Note, this is equivalent to `data.standardize()`.
#
# Anatomically parcellated data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Alternatively, if you have defined regions of interest (ROIs) using an anatomical parcellation and loaded these into osl-dynamics instead of the group-ICA time courses, it is important to apply PCA to infer good dynamics (state switching). Below we apply a full rank PCA to the data.


data.prepare({
    "pca": {"n_pca_components": data.n_channels},
    "standardize": {},
})

#%%
# You can access the PCA components via `data.pca_components`.
#
# Saving and Loading Prepared Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Saving prepared data
# ********************
# For large datasets, preparing the data can sometimes be time consuming. In this case it is useful to save the data after preparing it. Then we can just load the prepared data before training a model. To save prepared data we can use the `save` method. We simply need to pass the output directory to write the data to.


data.save("prepared_data")


#%%
# This method has created a directory called `prepared_data`. Let's list its contents.


os.listdir('prepared_data')

#%%
# We can see each subject's data is saved as a numpy file and there is an additional pickle (`.pkl`) file which contains information regarding how the data was prepared.
#
# Loading prepared data
# *********************
# We can load the prepared data by simply passing the path to the directory to the Data class.


data = Data("prepared_data")
print(data)

#%%
# We can see from the number of samples it is the amplitude envelope data that we previously prepared.
#
# Note, if we saved data that included PCA in preparation. The `pca_components` attribute will be loaded from the pickle file when we load data using the Data class.
