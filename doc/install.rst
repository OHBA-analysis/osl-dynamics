Install
=======

The recommended installation is described below.

Conda Environments
------------------
We recommend installing osl-dynamics within a conda environment, see https://docs.conda.io/en/latest for further details.
A conda environment can be created and activated with:

::

    conda create --name osld python=3
    conda activate osld


NOTE: this conda environment must be activated everytime you would like to use osl-dynamics.

Pip Installation
----------------

osl-dynamics can be installed in your conda environment with:

::

    pip install osl-dynamics

This will install the latest version of TensorFlow compatible with your Python version.

For Developers
--------------

Developers will want to install from source in editable mode:

::

    git clone git@github.com:OHBA-analysis/osl-dynamics.git
    cd osl-dynamics
    conda env create -f envs/linux.yml
    conda activate osld
    pip install -e .

Additional packages needed for development can be installed with:

::

    pip install black
    pip install -r doc/requirements.txt
    pip install build twine

Depending on the machine you are installing on you may want to use the system specific conda environments: bmrc.yml, hbaws.yml, m1_mac.yml.
