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

The recommended installation for osl-dynamics is:
```
conda create -n osld -c conda-forge osl-dynamics
conda activate osld
```

See the [installation page](https://osl-dynamics.readthedocs.io/en/latest/install.html) for more information.

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

2. Create a new release using the 'Create a new release' link on the right of the GitHub repo webpage:

    - Set the tag to the new version number with a `v` prefix (e.g. `v3.3.0`).
    - Write the release notes.
    - Select 'Latest' for the release label and click 'Publish release'.

3. Publishing the release triggers a GitHub Actions workflow (`.github/workflows/release.yml`) that builds the package and uploads it to [PyPI](https://pypi.org/project/osl-dynamics/). Check the workflow succeeded under the Actions tab of the GitHub repo.

## Citation

If you find this toolbox useful, please cite the [paper](https://elifesciences.org/articles/91949):

> **Gohil, C., Huang, R., Roberts, E., van Es, M. W., Quinn, A. J., Vidaurre, D., & Woolrich, M. W. (2024). osl-dynamics, a toolbox for modeling fast dynamic brain activity. Elife, 12, RP91949.**

