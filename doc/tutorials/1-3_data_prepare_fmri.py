"""
Preparing fMRI Data
===================

In this tutorial we discuss how to prepare fMRI data. This tutorial covers:

1. Preparing group-ICA data
2. Preparing anatomically parcellated data
3. Saving and loading prepared data

fMRI data needs to be preprocessed in a particular way to train a dynamic network model (HMM/DyNeMo), see this `paper <https://www.sciencedirect.com/science/article/pii/S1053811922001550>`_ for further details. This is make sure we infer good dynamics (state switching) in the data. We advise training a model on the group-ICA time courses (which are prepared by standardizing the data) or using an anatomical parcellation (which is prepared using a full rank PCA).
"""

#%%
# Preparing group-ICA data
# ^^^^^^^^^^^^^^^^^^^^^^^^
# This is the recommended data to train an HMM/DyNeMo on.
# 
# Deriving subject-specific time series via dual regression
# *********************************************************
# This section describes how group ICA time courses can be extracted from preprocessed fMRI data using **dual regression**. This process is applicable to data in both volumetric space and surface space.
# 
# **1. Input Requirements**
# 
# Dual regression requires two inputs: the preprocessed subject data and a set of group-level spatial maps (Group ICA) to act as a template.
# 
# **A. Preprocessed Subject Data**
# 
# Ensure your subject data has been preprocessed in one of the following formats:
# 
# - **Volumetric Space:** Standard 4D fMRI nifti files.
# - **Surface Space:** Data processed using the Human Connectome Project (HCP) surface-based pipeline (CIFTI/GIFTI formats).
# 
# **B. Group ICA Maps (Spatial Templates)**
# 
# You must provide a set of group-level spatial maps. These are typically obtained by running group-level Independent Component Analysis (ICA) using FSL MELODIC. These maps function as a "soft parcellation" of the brain, identifying functional networks common across the group.
# 
# For technical details on generating these maps, refer to the `FSL MELODIC Documentation <https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/MELODIC.html>`_.
# 
# **2. Dual Regression**
# 
# To derive the final subject-specific time series, you must run dual Regression on your preprocessed data (Input A) using the Group ICA maps (Input B) as the spatial regressors.
# 
# **Understanding the Output Stages**
# 
# Dual regression is a two-stage process.
# 
# **Stage 1: Spatial Regression**
# 
# In this first stage, the group-level spatial maps (e.g. from UKB or HCP) are regressed into the subject's 4D dataset.
# 
#     **Key Output:** The result of Stage 1 is the **Group ICA time courses**. These represent the subject-specific temporal dynamics associated with each group-level spatial component.
# 
# **Stage 2: Temporal Regression**
# 
# The time courses from stage 1 are used in the second stage to regress against the subject's data again to find subject-specific *spatial* maps.
#
#     **Note:** if you only require the time series, Stage 1 is your endpoint.
# 
# **3. Further Reading & Resources**
# 
# For a comprehensive technical explanation of the mathematical framework behind dual regression, please consult the official FSL documentation and course materials:
# 
# - `FSL Wiki: Dual Regression Details <https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/DualRegression.html#Research_Overview_-_Dual_Regression>`_.
# - `ICA and Dual Regression (FSL Course) <https://fsl.fmrib.ox.ac.uk/fslcourse/2019_Beijing/lectures/ICA_and_resting_state/ICA_and_Dual_Regression.pdf>`_.

#%%
# Example data
# ************
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_ that we have already calculated dual regression for.

import os

def get_data(name):
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset (approximately 6 MB)
get_data("example_loading_data")

# List the contents of the downloaded directory containing the dataset
print("Contents of example_loading_data:")
os.listdir("example_loading_data")

#%%
# Now, let's load the example data into osl-dynamics. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.

from osl_dynamics.data import Data

data = Data("example_loading_data/txt_format")
print(data)

#%%
# To prepare the data, all we need to do is standardize it. Let's use the `Data.prepare` method to do this.

data.prepare({"standardize": {}})

#%%
# Note, this is equivalent to `data.standardize()`.
#
# Preparing anatomically parcellated data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

import os

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
# Note, if we saved data that included PCA in preparation. The `pca_components` attribute will be loaded from the pickle file when we load data using the Data class.
#
# Next steps
# ^^^^^^^^^^
# Once you have loaded and prepared your fMRI data, the next step is to train a dynamic network model. See:
#
# - `HMM: Training <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/3-2_hmm_training.html>`_
# - `DyNeMo: Training <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/3-3_dynemo_training.html>`_
