Install
=======

The recommended installation is described below. We recommend using Python 3.8 and TensorFlow 2.4.

Conda Environments
------------------
We recommend installing osl-dynamics within a conda environment, see https://docs.conda.io/en/latest for further details.
A conda environment can be created and activated with:

::

    conda create --name osld python=3.8
    conda activate osld


NOTE: this conda environment must be activated everytime you would like to use osl-dynamics.

Pip Installation
----------------

osl-dynamics can be installed in your conda environment with:

::

    pip install osl-dynamics

This will install TensorFlow 2.4 by default.

For Developers
--------------

Developers will want to install from source in editable mode:

::

    conda create --name osld-tf24 python=3.8
    conda activate osld-tf24
    git clone git@github.com:OHBA-analysis/osl-dynamics.git
    cd osl-dynamics
    pip install -e .

To use the HMM you will also need to install armadillo:

::

    conda install -c conda-forge armadillo

Additional packages needed for development can be installed with:

::

    pip install black
    pip install -r doc/requirements.txt
    pip install build twine
