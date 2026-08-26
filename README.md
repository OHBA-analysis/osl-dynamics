# OHBA Software Library: Dynamics Toolbox

[![PyPI version](https://img.shields.io/pypi/v/osl-dynamics)](https://pypi.org/project/osl-dynamics/)
[![Documentation](https://readthedocs.org/projects/osl-dynamics/badge/?version=latest)](https://osl-dynamics.readthedocs.io)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://github.com/OHBA-analysis/osl-dynamics/blob/main/LICENSE)
[![Paper](https://img.shields.io/badge/paper-eLife-orange)](https://elifesciences.org/articles/91949)

osl-dynamics is a Python toolbox for studying brain dynamics using neuroimaging data: MEG, EEG and fMRI. It provides generative models that decompose data into brain networks (often called brain states or modes), including the Hidden Markov Model (HMM) and Dynamic Network Modes (DyNeMo), along with everything needed for a complete analysis: data loading and preparation, spectral estimation, network visualisation and statistical significance testing.

You can use osl-dynamics to:

- **Infer dynamic functional networks** from resting-state or task M/EEG and fMRI data using the HMM, DyNeMo and related models (M-DyNeMo, HIVE, DIVE, DyNeSTE and more).
- **Characterise brain states/modes** with summary statistics (fractional occupancy, lifetimes, intervals, switching rates), state-specific power maps, and functional connectivity.
- **Estimate spectra** using multitaper and regression-based methods, or wavelet transforms.
- **Detect oscillatory bursts**.
- **Test for statistical significance** using GLM permutation testing.
- **Preprocess and source reconstruct M/EEG data**: preprocessing, coregistration, beamforming and parcellation.
- **Simulate time series data** from HMMs, sinusoidal oscillators and autoregressive models.

osl-dynamics works with [MNE-Python](https://mne.tools): a typical M/EEG workflow preprocesses, source reconstructs and parcellates data first, then models the dynamics of the parcel time courses. Data can be loaded from NumPy (`.npy`), MATLAB (`.mat`), text (`.txt`) or MNE (`.fif`) files.

For a full description of the toolbox, see the [documentation](https://osl-dynamics.readthedocs.io).

## Quick example

Train a Time-Delay Embedded Hidden Markov Model (TDE-HMM) on parcellated MEG data to infer dynamic functional brain networks:

```python
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model

# Load data, e.g. parcel time courses
data = Data("training_data")

# Prepare the data: time-delay embedding + PCA captures spectral structure
data.prepare({
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
    "standardize": {},
})

# Train an HMM
config = Config(
    n_states=8,
    n_channels=data.n_channels,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=256,
    learning_rate=0.01,
    n_epochs=20,
)
model = Model(config)
model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
model.fit(data)

# Get inferred state probabilities
alpha = model.get_alpha(data)
```

See the [tutorials](https://osl-dynamics.readthedocs.io/en/latest/documentation.html) for complete walkthroughs and the [examples directory](https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples) for full analysis pipelines.

## Installation

We recommend installing osl-dynamics using the conda environment files in `/envs`, which can be installed using [Miniforge](https://conda-forge.org/download/).

### conda / mamba installation

Miniforge (`conda`/`mamba`) can be installed with:
```
curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
```

### osl-dynamics installation

Different computers have their own environment files. For more information see the envs [README](https://github.com/OHBA-analysis/osl-dynamics/tree/main/envs#readme).

#### Linux
```
curl -LO https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf.yml
mamba env create -f osld-tf.yml
rm osld-tf.yml
```

If you have a GPU, then use the `osld-tf-cuda.yml` environment instead:
```
curl -LO https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf-cuda.yml
mamba env create -f osld-tf-cuda.yml
rm osld-tf-cuda.yml
```

#### Mac

If you have an M-series (M1, M2, M3) chip use:
```
curl -LO https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf.yml
mamba env create -f osld-tf.yml
rm osld-tf.yml
```

Otherwise, if you have an Intel chip use:
```
curl -LO https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf-macos.yml
mamba env create -f osld-tf-macos.yml
rm osld-tf-macos.yml
```

#### Windows

If you are using a Windows computer, we recommend first installing Linux (Ubuntu) as a Windows Subsystem by following the instructions [here](https://ubuntu.com/wsl). Then follow the instructions for Linux above in the Ubuntu terminal.

#### hbaws (Oxford)

On the OHBA workstation (hbaws), install Miniforge and Mamba using the instructions above and install osl-dynamics using:
```
curl -LO https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/hbaws.yml
mamba env create -f hbaws.yml
rm hbaws.yml
```

#### BMRC (Oxford)

On the Biomedical Research Computing (BMRC) cluster, `conda` is available as a software module:
```
module load Miniforge3
```
and osl-dynamics can be installed with:
```
curl -LO https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/bmrc.yml
conda env create -f bmrc.yml
rm bmrc.yml
```
The above can be run on the login nodes (`clusterX.bmrc.ox.ac.uk`). On `compg017` you will need to set the following to use conda:
```
unset https_proxy http_proxy no_proxy HTTPS_PROXY HTTP_PROXY NO_PROXY
```

### Install the latest code from the GitHub repository (optional)

You should only need to do this if you need a feature or fix that has not been released on pip yet.

After you have created an `osld` environment you can install the latest code (development version) from the GitHub repository with:
```
conda activate osld
pip install git+https://github.com/OHBA-analysis/osl-dynamics.git
```

### Install the source code (optional)

After you have created an `osld` environment you can install an editable local copy of the source code on your computer with:
```
git clone https://github.com/OHBA-analysis/osl-dynamics.git
conda activate osld
cd osl-dynamics
pip install -e .
```
You will run your local copy of the code when you `import osl_dynamics`.

If you are a developer, you may wish to clone the repository using SSH rather than HTTPS to make pushing branches/commits easier:
```
git clone git@github.com:OHBA-analysis/osl-dynamics.git
```

### Test GPUs are working

You can use the following to check if TensorFlow is using any GPUs you have available:
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
This should return a list of GPUs.

### Removing osl-dynamics

Simply delete the conda environment:
```
conda env remove -n osld
conda clean --all
```
And remove the GitHub repository if you have cloned it:
```
rm -rf osl-dynamics
```

## Documentation

The read the docs page should be automatically updated whenever there's a new commit on the `main` branch.

The documentation is included as docstrings in the source code. The API reference documentation will only be automatically generated if the docstrings are written correctly. The documentation directory `/doc` also contains `.rst` files that provide additional info regarding installation, development, the models, etc.

To compile the documentation locally you need to install the required packages (sphinx, etc) in your conda environment:
```
cd osl-dynamics
conda activate osld
pip install -r doc/requirements.txt
```
To compile the documentation locally use:
```
sphinx-build -b html doc build
```
The local build of the documentation webpage can be found in `build/sphinx/html/index.html`.

To skip building the tutorials, comment out `"sphinx_gallery.gen_gallery"` [here](https://github.com/OHBA-analysis/osl-dynamics/blob/main/doc/conf.py#L36).

## Releases

To release a new version:

1. Check the latest commit on `main` has compiled successfully on [readthedocs](https://readthedocs.org/projects/osl-dynamics).

2. Create a new release using the 'Create a new release' link on the right of the GitHub repo webpage. Set the tag to the new version number with a `v` prefix (e.g. `v3.3.0`), write the release notes, the output of the following is a useful starting point:
```
git log --oneline <previous tag>..main
```
Select 'Latest' for the release label and click 'Publish release'.

3. Publishing the release triggers a GitHub Actions workflow (`.github/workflows/release.yml`) that builds the package and uploads it to [PyPI](https://pypi.org/project/osl-dynamics/). Check the workflow succeeded under the Actions tab of the GitHub repo.

Installations from a clone of the repo (`pip install -e .`) automatically get a development version number based on the latest tag, e.g. `3.3.1.dev12` if 12 commits have been made since `v3.3.0`.

## Citation

If you find this toolbox useful, please cite the [paper](https://elifesciences.org/articles/91949):

> **Gohil, C., Huang, R., Roberts, E., van Es, M. W., Quinn, A. J., Vidaurre, D., & Woolrich, M. W. (2024). osl-dynamics, a toolbox for modeling fast dynamic brain activity. Elife, 12, RP91949.**

