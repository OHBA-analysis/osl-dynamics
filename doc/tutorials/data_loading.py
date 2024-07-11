"""
Loading Data
============

In this tutorial we demonstrate the various options for loading data. This tutorial covers:

1. The Data Class
2. Getting Example Data
3. Loading Data in NumPy Format
4. Loading Data in MATLAB Format
5. Loading Data in fif Format

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/dvfku>`_ for the expected output.
"""

#%%
# The Data class
# ^^^^^^^^^^^^^^
# In osl-dynamics we typically load data using the `osl_dynamics.data.Data class <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_. The Data class has a lot of useful methods that can be used to modify the data.
#
# Inputs
# ******
# There is one mandatory argument that needs to be passed to the Data class: `inputs`. This can be:
#
# - A path to a directory containing .npy files. Each .npy file should be a subject or session.
# - A list of paths to .npy, .mat, or .fif files. Each file should be a subject or session.
# - A numpy array. The array will be treated as continuous data from the same subject.
# - A list of numpy arrays. Each numpy array should be the data for a subject or session.
#
# Data format
# ***********
# The data files or numpy arrays should be in the format `(n_samples, n_channels)`, i.e. time by channels. If your data is in `(n_channels, n_samples)` format, use should also pass `time_axis_first=False` to the Data class.
#
# The temporary store directory
# *****************************
# Note, there is an option to load the data as a `memory map <https://numpy.org/doc/stable/reference/generated/numpy.memmap.html>`_. This allows us to access the data without holding it in memory. To use this feature, pass `load_memmaps=True`. The Data class creates a directory called `tmp` which is used for storing temporary data (memory map files and prepared data). This directory can be safely deleted after you run your script. You can specify the name of the temporary directory by passing the `store_dir` argument.
#
# We will demonstate how the Data class is used with example data below.
#
# Getting Example Data
# ^^^^^^^^^^^^^^^^^^^^
#
# Download the dataset
# ********************
# We will download example data hosted on `OSF <https://osf.io/zxb6c/>`_.


import os

def get_data(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset (approximately 88 MB)
get_data("example_loading_data")

# List the contents of the downloaded directory containing the dataset
print("Contents of example_loading_data:")
os.listdir("example_loading_data")

#%%
# We can see there are three directories in `example_loading_data`:
#
# - `numpy_format`, which contains `.npy` files.
# - `matlab_format`, which contains `.mat` files.
# - `fif_format`, which contains directories with `.fif` files.
#
# We'll show how to load data in each of these data types.
#
# Loading Data in NumPy Format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's first list the `example_loading_data/numpy_format` directory.


os.listdir("example_loading_data/numpy_format")

#%%
# We can see there's two numpy files. These files contain 2D numpy array data. It is in time by channels format. If we wanted to load this data using the numpy package, we could do:


import numpy as np

# Just load one of the files
X = np.load("example_loading_data/numpy_format/subject0.npy")
print(X.shape)

#%%
# Importing a numpy array directly
# ********************************
# If we have already loaded a numpy array and just want to create an `osl_dynamics.data.Data` object, we can simply pass it to the class:


from osl_dynamics.data import Data

data = Data(X)
print(data)

#%%
# We normally like to keep the data for each subject separate. If we had multiple 2D numpy arrays (one for each subject), we can collate them into a python list and pass that to the Data class:


# Load numpy files
X0 = np.load("example_loading_data/numpy_format/subject0.npy")
X1 = np.load("example_loading_data/numpy_format/subject1.npy")

# Collate into a list
X = [X0, X1]

# Create a Data object
data = Data(X)
print(data)

#%%
# Loading from file
# *****************
# Rather than loading the data into memory then creating a Data object, we could load the data directly from the file.


# Just load one of the files
data = Data("example_loading_data/numpy_format/subject0.npy")
print(data)

#%%
# We can see the data loaded matches the array shape when we loaded it using numpy. To access the 2D numpy array we can use the `time_series()` method.


ts = data.time_series()
print(ts.shape)

#%%
# Normally, we would want to load the data for multiple subjects. We could do this in two ways if the data is in numpy format (i.e. `.npy`). We could pass a list of file paths:


files = [f"example_loading_data/numpy_format/subject{i}.npy" for i in [0, 1]]
data = Data(files)
print(data)

#%%
# or just pass the path to the directory containing the `.npy` files:


data = Data("example_loading_data/numpy_format")
print(data)

#%%
# Note, when we have multiple subjects, if we call the `time_series()` method, we will now get a list of numpy arrays. Each item in the list is the data for each subject.


ts = data.time_series()
print(len(ts))
print(ts[0].shape)
print(ts[1].shape)

#%%
# Loading Data in MATLAB Format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We will discuss two methods for loading MATLAB files. First, we will load the MATLAB files using public python packages (`scipy` and `mat73`), then we'll show how to pass MATLAB files to the Data class.
#
# Loading MATLAB files in Python
# ******************************
# The popular python package SciPy has a function for loading MATLAB files: `scipy.io.loadmat <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html>`_. Note, this function can only be used to load a newer version of MATLAB files, if you saved your files using `v7.3` format, then you need to use `mat73.loadmat <https://github.com/skjerns/mat7.3>`_ to load the file in python. Both of these packages are automatically installed when you install osl-dynamics.
#
# Let first see what files we have in the `example_loading_data/matlab_format` directory.


os.listdir("example_loading_data/matlab_format")

#%%
# Let's load the first subject's data using standard python function.


from scipy.io import loadmat

# Load the first subject
mat = loadmat("example_loading_data/matlab_format/subject0.mat")
print(mat)

#%%
# We can see the `loadmat` function returns a python dict. We can list the fields using:


mat.keys()

#%%
# The important field is `X`, which is the one that contains the 2D time series data for this subject. Note, MATLAB files created using the `HMM-MAR <https://github.com/OHBA-analysis/HMM-MAR>`_ toolbox come in the above format, i.e. with a `X` and `T` field. For us, only the `X` matters.
#
# Loading MATLAB data into the Data class
# ***************************************
# We can pass the numpy array contained in the `X` field of the dictionary directly to the Data class:


data = Data(mat["X"])
print(data)

#%%
# However, we would prefer to load the data directly from the file. We can do this by passing the file path to the `.mat` file and the `data_field` argument to the Data class.


data = Data("example_loading_data/matlab_format/subject0.mat", data_field="X")
print(data)

#%%
# Note, the default value for the `data_field` argument is `X`, so the Data class would still be able to load the data without it being passed. The `data_field` is useful if the data is contained in a MATLAB in a field with a different name.
#
# If we wanted to load multiple data files in MATLAB format we would need to pass a list of file paths.


files = [f"example_loading_data/matlab_format/subject{i}.mat" for i in [0, 1]]
data = Data(files)
print(data)

#%%
# Loading fif files
# *****************
# Another data format that can be loaded with the Data class is fif files. This format is commonly used in `MNE-Python <https://mne.tools/stable/index.html>`_ and is the data format used in `OSL <https://github.com/OHBA-analysis/osl>`_. Here, we will load source reconstruct (parcellated) data created with OSL. In OSL, we often have a separate directory for each subject. The `fif_format` directory contains two directories for different subjects.


os.listdir('example_loading_data/fif_format')

#%%
# Let's see what's inside `subj001_run01`.


os.listdir('example_loading_data/fif_format/subj001_run01')

#%%
# We have a fif file which contains the data for this subject. We could load this with MNE.


import mne

raw = mne.io.read_raw_fif("example_loading_data/fif_format/subj001_run01/sflip_parc-raw.fif")
print(raw.info)

#%%
# We can see this particular fif file contains 38 `misc` channels and 3 `stim` channels. We're interested in the `misc` channels. Let's load these into the Data class.


data = Data(
    "example_loading_data/fif_format/subj001_run01/sflip_parc-raw.fif",
    picks="misc",
    reject_by_annotation="omit",
)
print(data)

#%%
# The `reject_by_annotation="omit"` argument is used to make sure we don't include bad segments. This argument is passed to `Raw.get_data <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.get_data>`_ in MNE.
#
# To load multiple subjects we can do:


files =[
    f"example_loading_data/fif_format/subj{i:03d}_run01/sflip_parc-raw.fif"
    for i in range(1,3)
]
data = Data(files, picks="misc", reject_by_annotation="omit")
print(data)

