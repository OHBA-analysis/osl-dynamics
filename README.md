# OHBA Software Library: Dynamics Toolbox

See the read the docs page for a description of this project: [https://osl-dynamics.readthedocs.io](https://osl-dynamics.readthedocs.io).

## Citation

If you find this toolbox useful, please cite:

> **Chetan Gohil, Rukuang Huang, Evan Roberts, Mats WJ van Es, Andrew J Quinn, Diego Vidaurre, Mark W Woolrich (2024) osl-dynamics, a toolbox for modeling fast dynamic brain activity eLife 12:RP91949.**

## Installation

We recommend installing osl-dynamics using the conda environment files in `/envs`, which can be installed using [Miniforge](https://conda-forge.org/download/) (or [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install)/[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)) and [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

### conda / mamba installation

Miniforge (`conda`) can be installed with:
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
```

Mamba (`mamba`) can be installed with:
```
conda install -n base -c conda-forge mamba
```

### osl-dynamics installation

Different computers have their own environment files. For more information see the envs [readme](https://github.com/OHBA-analysis/osl-dynamics/tree/main/envs#readme).

#### Linux
```
wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf.yml
mamba env create -f osld-tf.yml
rm osld-tf.yml
```

If you have a GPU, then use the `osld-tf-cuda.yml` environment instead:
```
wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf-cuda.yml
mamba env create -f osld-tf-cuda.yml
rm osld-tf-cuda.yml
```

#### Mac

If you have an M-series (M1, M2, M3) chip use:
```
wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf.yml
mamba env create -f osld-tf.yml
rm osld-tf.yml
```

Otherwise, if you have an Intel chip use:
```
wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf-macos.yml
mamba env create -f osld-tf-macos.yml
rm osld-tf-macos.yml
```

#### Windows

If you are using a Windows computer, we recommend first installing Linux (Ubuntu) as a Windows Subsystem by following the instructions [here](https://ubuntu.com/wsl). Then follow the instructions for Linux above in the Ubuntu terminal.

#### hbaws (Oxford)

On the OHBA workstation (hbaws), install Miniforge and Mamba using the instructions above and install osl-dynamics using:
```
wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/hbaws.yml
mamba env create -f hbaws.yml
rm hbaws.yml
```

#### BMRC (Oxford)

On the Biomedical Research Computing (BMRC) cluster, `conda` and `mamba` are available as a software module:
```
module load Miniforge3
```
and osl-dynamics can be installed with:
```
wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/bmrc.yml
mamba env create -f bmrc.yml
rm bmrc.yml
```
The above can be run on the login nodes (`clusterX.bmrc.ox.ac.uk`). On `compg017` you will need to set the following to use conda:
```
unset https_proxy http_proxy no_proxy HTTPS_PROXY HTTP_PROXY NO_PROXY
```

See [here](https://github.com/OHBA-analysis/osl-dynamics/blob/main/doc/using_bmrc.rst) for useful information regarding how to use the BMRC cluster.

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

## Releases

A couple packages are needed to build and upload a project to PyPI, these can be installed in your conda environment with:

```
pip install build twine
```

The following steps can be used to release a new version:

1. Update the version on line 10 of `pyproject.toml` by removing `dev` from the version number.

2. Commit the updated `pyproject.toml` to the `main` branch of the GitHub repo.

3. Delete any old distributions that have been built (if there are any):
```
rm -r dist
```

4. Build a distribution in the osl-dynamics root directory with:
```
python -m build
```
This will create a new directory called `dist`.

5. Test the build by installing with
```
pip install dist/<build>.whl
```

6. Upload the distribution to PyPI with
```
twine upload dist/*
```
You will need to enter the username and password that you used to register with [https://pypi.org](https://pypi.org). You may need to setup 2FA and/or an API token, see API token instructions in your PyPI account settings.

7. Tag the commit uploaded to PyPI with the version number using the 'Create a new release' link on the right of the GitHub repo webpage. You will need to untick 'Set as a pre-release' and tick 'Set as the latest release'.

8. Change the version to `X.Y.devZ` in `pyproject.toml` and commit the new dev version to `main`.

The uploaded distribution will then be available to be installed with:
```
pip install osl-dynamics
```

9. Optional: draft a new release (click 'Releases' on the right panel on the GitHub homepage, then 'Draft a new release') to help keep note of changes for the next release.

10. Activate the new version in the [readthedocs](https://readthedocs.org/projects/osl-dynamics) project.
