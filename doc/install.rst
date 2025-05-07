Installation
============

We recommend installing osl-dynamics using `Miniforge <https://conda-forge.org/download/>`_ (or `Anaconda <https://www.anaconda.com/docs/getting-started/anaconda/install>`_/`Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/install>`_) and `Mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_.

Conda / Mamba Installation
--------------------------

Miniforge (:code:`conda`) can be installed with:

.. code::

    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    rm Miniforge3-$(uname)-$(uname -m).sh

Mamba (:code:`mamba`) can be installed with:

.. code::

    conda install -n base -c conda-forge mamba

Linux Instructions
------------------

The following lines can be used to download a conda environment file and install osl-dynamics with its dependencies.

.. code::

    wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf.yml
    mamba env create -f osld-tf.yml
    rm osld-tf.yml

If you have a GPU, then use the :code:`osld-tf-cuda.yml` environment file:

.. code::

    wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf-cuda.yml
    mamba env create -f osld-tf-cuda.yml
    rm osld-tf-cuda.yml

Mac Instructions
----------------

If you have an M-series (M1, M2, M3) chip, the following lines can be used to download a conda environment file and install osl-dynamics with its dependencies.

.. code::

    wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf.yml
    mamba env create -f osld-tf.yml
    rm osld-tf.yml

If you have an Intel chip, then use the :code:`osld-tf-macos.yml` environment file:

.. code::

    wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld-tf-macos.yml
    mamba env create -f osld-tf-macos.yml
    rm osld-tf-macos.yml

Windows Instructions
--------------------

If you are using a Windows computer, we recommend first installing linux (Ubuntu) as a Windows Subsystem by following the instructions `here <https://ubuntu.com/wsl>`_. Then following the instructions above in the Ubuntu terminal.

Oxford-Specific Computers (hbaws, BMRC)
---------------------------------------

See the instructions on the GitHub `README <https://github.com/OHBA-analysis/osl-dynamics>`_.

TensorFlow Versions
-------------------

osl-dynamics v2.1+ requires tensorflow 2.19 and tensorflow-probability 0.25.

osl-dynamics v2.0.0-2.0.2 has been tested with the following versions:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - tensorflow
     - tensorflow-probability
   * - 2.11
     - 0.19
   * - 2.12
     - 0.19
   * - 2.13
     - 0.20
   * - 2.14
     - 0.22
   * - 2.15
     - 0.22

Test your GPUs are working
--------------------------

You can use the following to check if TensorFlow is using any GPUs you have available:

.. code::

    conda activate osld
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

This should print a list of the GPUs you have available (or an empty list :code:`[]` if there's none available).

Install without TensorFlow
--------------------------

If you want to install osl-dynamics without TensorFlow, then use the :code:`osld.yml` environment file:

.. code::

    wget https://raw.githubusercontent.com/OHBA-analysis/osl-dynamics/refs/heads/main/envs/osld.yml
    mamba env create -f osld.yml
    rm osld.yml

This environment file will only install the osl-dynamics package. You will have to install the dependencies (numpy, scipy, etc) yourself.

Install the latest development code (optional)
----------------------------------------------

You should only need to do this if you need a feature or fix that has not been released on pip yet.

Once you have created the :code:`osld` conda environment (see instructions above) you can install the latest development version on the `GitHub repository <https://github.com/OHBA-analysis/osl-dynamics>`_ with:

.. code::

    conda activate osld
    pip install git+https://github.com/OHBA-analysis/osl-dynamics.git

Install the source code (optional)
----------------------------------

Once you have created the :code:`osld` conda environment (see instructions above) you can install a local copy of the source code (`GitHub repository <https://github.com/OHBA-analysis/osl-dynamics>`_) into it.

.. code::

    git clone https://github.com/OHBA-analysis/osl-dynamics.git
    conda activate osld
    cd osl-dynamics
    pip install -e .

Now you can run osl-dynamics with your own local changes to the code.

Removing osl-dynamics
---------------------

To remove osl-dynamics simply delete the conda environment:

::

    conda env remove -n osld
    conda clean --all
