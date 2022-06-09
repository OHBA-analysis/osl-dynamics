Installing OSL Dynamics
=======================

Currently osl-dynamics can only be installed from source. Installation instructions are given below.

Conda Environments
------------------
We recommend installing osl-dynamics within a conda environment, see https://docs.conda.io/en/latest for further details.

Pip Installation
----------------
osl-dynamics can be installed from source with the following:

::
    
    git clone https://github.com/OHBA-analysis/osl-dynamics.git
    conda create --name osld python=3.8
    conda activate osld
    cd osl-dynamics
    pip install .

This will install TensorFlow 2.4 by default.

Developers will want to install in editable mode:

::

    pip install -e .

See the `contribution guide <https://github.com/OHBA-analysis/osl-dynamics/blob/main/CONTRIBUTION.md>`_ for further details.
