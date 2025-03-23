# osl-dynamics

See the read the docs page for a description of this project: [https://osl-dynamics.readthedocs.io](https://osl-dynamics.readthedocs.io).

## Citation

If you find this toolbox useful, please cite:

> **Chetan Gohil, Rukuang Huang, Evan Roberts, Mats WJ van Es, Andrew J Quinn, Diego Vidaurre, Mark W Woolrich (2024) osl-dynamics, a toolbox for modeling fast dynamic brain activity eLife 12:RP91949.**

## Installation

We recommend using the conda environment files in `/envs`, which can be installed using [Miniforge](https://conda-forge.org/download/) (or [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install)/[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)) and [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

### Conda / Mamba

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

### Linux/Mac

```
wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/conda-envs/envs/osld-tf.yml
mamba env create -f osld-tf.yml
rm osld-tf.yml
```
If you have a GPU, then use the `osld-tf-cuda.yml` environment.

### Windows

If you are using a Windows computer, we recommend first installing Linux (Ubuntu) as a Windows Subsystem by following the instructions [here](https://ubuntu.com/wsl). Then follow the instructions above in the Ubuntu terminal.

### Source code

An editable local copy of the GitHub repo can be installed within the `osld` environment (created above):
```
conda activate osld
git clone https://github.com/OHBA-analysis/osl-dynamics.git
cd osl-dynamics
pip install -e .
```

### hbaws (Oxford)

On the OHBA workstation (hbaws), install Miniforge and Mamba using the instructions above and install osl-dynamics using:
```
git clone https://github.com/OHBA-analysis/osl-dynamics.git
cd osl-dynamics
mamba env create -f envs/hbaws.yml
conda activate osld
pip install -e .
```

### BMRC (Oxford)

On the Biomedical Research Computing (BMRC) cluster, `mamba` is available as a software module:
```
module load Miniforge3
```
and osl-dynamics can be installed with:
```
git clone https://github.com/OHBA-analysis/osl-dynamics.git
cd osl-dynamics
mamba env create -f envs/bmrc.yml
conda activate osld
pip install -e .
```
The above can be run on the login nodes (`clusterX.bmrc.ox.ac.uk`). On `compg017` you will need to set the following to use conda:
```
unset https_proxy http_proxy no_proxy HTTPS_PROXY HTTP_PROXY NO_PROXY
```

### Within an osl environment

If you have already installed [osl-ephys](https://github.com/OHBA-analysis/osl-ephys) you can install osl-dynamics in the `osl` environment with:
```
conda activate osl
cd osl-dynamics
pip install tensorflow==2.19
pip install tensorflow-probability[tf]==0.25
pip install -e .
```

If you want GPU support, install TensorFlow with
```
pip install tensorflow[and-cuda]==2.19
```

### Test GPUs are working

You can use the following to check if TensorFlow is using any GPUs you have available:
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
This should return a list of GPUs.

### Removing osl-dynamics

Simply delete the conda environment and repository:
```
conda env remove -n osld
rm -rf osl-dynamics
```

## Documentation

The read the docs page should be automatically updated whenever there's a new commit on the `main` branch.

The documentation is included as docstrings in the source code. The API reference documentation will only be automatically generated if the docstrings are written correctly. The documentation directory `/doc` also contains `.rst` files that provide additional info regarding installation, development, the models, etc.

To compile the documentation locally you need to install the required packages (sphinx, etc.) in your conda environment:
```
cd osl-dynamics
pip install -r doc/requirements.txt
```
To compile the documentation locally use:
```
python setup.py build_sphinx
```
The local build of the documentation webpage can be found in `build/sphinx/html/index.html`.

## Releases

A couple packages are needed to build and upload a project to PyPI, these can be installed in your conda environment with:

```
pip install build twine
```

The following steps can be used to release a new version:

1. Update the version on line 5 of `setup.cfg` by removing `dev` from the version number.

2. Commit the updated `setup.cfg` to the `main` branch of the GitHub repo.

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

8. Change the version to `X.Y.devZ` in `setup.cfg` and commit the new dev version to `main`.

The uploaded distribution will then be available to be installed with:
```
pip install osl-dynamics
```

9. Optional: draft a new release (click 'Releases' on the right panel on the GitHub homepage, then 'Draft a new release') to help keep note of changes for the next release.

10. Activate the new version in the [readthedocs](https://readthedocs.org/projects/osl-dynamics) project.

## Editing Source Code

See [here](https://github.com/OHBA-analysis/osl-dynamics/blob/main/doc/using_bmrc.rst) for useful info regarding how to use the Oxford BMRC cluster and how to edit the source code.
