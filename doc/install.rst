Installation
============

osl-dynamics is available on `conda-forge <https://anaconda.org/conda-forge/osl-dynamics>`_ and `PyPI <https://pypi.org/project/osl-dynamics>`_. We recommend conda-forge.

Conda Installation (recommended)
--------------------------------

If you do not already have conda, install `Miniforge <https://conda-forge.org/download/>`_:

.. code::

    curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    rm Miniforge3-$(uname)-$(uname -m).sh

Then create an environment with osl-dynamics in it:

.. code::

    conda create -n osld -c conda-forge osl-dynamics
    conda activate osld

This installs osl-dynamics with everything it needs, TensorFlow included, on Linux and macOS (both Apple Silicon and Intel). There is no longer any need to download an environment file.

Windows Instructions
--------------------

conda-forge has no recent TensorFlow build for Windows, so the full package is not available there. We recommend installing linux (Ubuntu) as a Windows Subsystem (WSL) by following the instructions `here <https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/>`_, then following the Conda instructions above in the Ubuntu terminal.

If you only need to load, prepare and analyse data, :code:`osl-dynamics-base` (see below) does install natively on Windows.

Install without TensorFlow
--------------------------

If you do not need to train models, install :code:`osl-dynamics-base`:

.. code::

    conda create -n osld -c conda-forge osl-dynamics-base
    conda activate osld

You will be able to load, prepare and analyse data, but not build or train models. TensorFlow can be added later with :code:`conda install -c conda-forge tensorflow tensorflow-probability tf-keras`.

Test your GPUs are working
--------------------------

On linux, conda-forge builds TensorFlow both with and without CUDA and picks between them based on the drivers on your machine, so a GPU is used automatically where there is one. To check:

.. code::

    conda activate osld
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

This should print a list of the GPUs you have available (or an empty list :code:`[]` if there are none).

Pip Installation
----------------

osl-dynamics can also be installed from PyPI:

.. code::

    pip install osl-dynamics

This does not include TensorFlow. To install it as well:

.. code::

    pip install "osl-dynamics[tf]"

or, if you have an NVIDIA GPU:

.. code::

    pip install "osl-dynamics[tf-cuda]"

Note that pip will not install the non-Python libraries that some dependencies need, which is why we recommend conda.

Data files
----------

The parcellations, masks, surfaces, scanner layouts and Workbench scenes used by osl-dynamics are not shipped with the package. They live in `osl-files <https://github.com/OHBA-analysis/osl-files>`_ and are downloaded the first time something needs them, then cached.

If you will be working somewhere without network access, such as a cluster compute node, fetch everything up front from somewhere that does have access:

.. code::

    osl-dynamics-download-data

Files are cached in :code:`~/Library/Caches/osl-files` on macOS and :code:`~/.cache/osl-files` on linux. Set the :code:`OSL_DATA` environment variable to put them somewhere else, which is useful if your home directory has a quota, or to point a whole group at one shared copy:

.. code::

    export OSL_DATA=/path/to/shared/osl-files

Oxford-Specific Computers (hbaws, BMRC)
---------------------------------------

See the instructions on the GitHub `README <https://github.com/OHBA-analysis/osl-dynamics>`_.

Install the latest development code (optional)
----------------------------------------------

You should only need to do this if you need a feature or fix that has not been released yet.

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

The downloaded data files are cached separately, and can be removed with:

::

    rm -rf ~/.cache/osl-files        # linux
    rm -rf ~/Library/Caches/osl-files  # macOS
