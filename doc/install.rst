Installation
============

osl-dynamics is available on `conda-forge <https://anaconda.org/conda-forge/osl-dynamics>`_ and `PyPI <https://pypi.org/project/osl-dynamics>`_.

Conda installation (recommended)
--------------------------------

We recommend installing osl-dynamics with `Miniforge <https://conda-forge.org/download/>`_, which provides :code:`conda` and :code:`mamba`. If you do not already have it:

.. code::

    curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    rm Miniforge3-$(uname)-$(uname -m).sh

Then create an environment with osl-dynamics in it:

.. code::

    mamba create -n osld -c conda-forge osl-dynamics
    conda activate osld

This installs osl-dynamics with everything it needs. This includes `MNE-Python <https://mne.tools/stable/index.html>`_ and `TensorFlow <https://www.tensorflow.org/>`_.

On Windows, the command above will not work because conda-forge has no TensorFlow for Windows. See :ref:`windows-install` below.

Pip installation
----------------

osl-dynamics can be installed from PyPI with:

.. code::

    pip install osl-dynamics

This does not include TensorFlow. To install it as well:

.. code::

    pip install "osl-dynamics[tf]"

or, if you have an NVIDIA GPU:

.. code::

    pip install "osl-dynamics[tf-cuda]"

Note that pip will not install the non-Python libraries that some dependencies need, which is why we recommend conda.

.. _windows-install:

Windows
-------

conda-forge does not provide TensorFlow for Windows, so the full :code:`osl-dynamics` conda package is only available for Linux and macOS. There are two ways to use osl-dynamics on Windows.

**Option 1: WSL (recommended).** The `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ (WSL) runs Linux inside Windows. With it you can install the full package, including GPU support for training models. To set up WSL, open PowerShell as an administrator and run:

.. code::

    wsl --install

then restart your computer. Open the Ubuntu app from the Start menu and follow the `Conda installation (recommended)`_ instructions above in the Ubuntu terminal. If you have an NVIDIA GPU, install the latest NVIDIA driver for Windows; do not install a separate driver inside WSL. conda will install the GPU version of TensorFlow automatically.

**Option 2: native Windows.** Install `Miniforge <https://conda-forge.org/download/>`_ using the Windows installer, then open the Miniforge Prompt from the Start menu and run:

.. code::

    mamba create -n osld -c conda-forge osl-dynamics-base python=3.12
    conda activate osld
    pip install tensorflow==2.19.1 tensorflow-probability==0.25.0 tf-keras==2.19.0

The :code:`osl-dynamics-base` conda package contains everything except TensorFlow, which is installed from PyPI with pip in the last step. Install TensorFlow with the command above rather than :code:`pip install "osl-dynamics[tf]"`, which would replace the conda copy of osl-dynamics with the PyPI one. Python is set to 3.12 because this is the newest version TensorFlow 2.19.1 supports on Windows. Note that TensorFlow does not support GPUs on native Windows, so models will be trained on the CPU. If you need a GPU, use WSL.

TensorFlow is only needed to train models (e.g. the HMM and DyNeMo). If you only need the rest of osl-dynamics, such as the preprocessing, source reconstruction and static analysis, you can skip the :code:`pip install` step.

BMRC cluster (Oxford)
---------------------

On the Biomedical Research Computing (BMRC) cluster, `conda` is available as a software module:

.. code::

    module load Miniforge3

and osl-dynamics can be installed with:

.. code::

    mamba create -n osld -c conda-forge osl-dynamics
    conda activate osld

The above can be run on the login nodes (`clusterX.bmrc.ox.ac.uk`). On `compg017` you will need to set the following to use conda:

.. code::

    unset https_proxy http_proxy no_proxy HTTPS_PROXY HTTP_PROXY NO_PROXY

Install the latest development code (optional)
----------------------------------------------

You should only need to do this if you need a feature or fix that has not been released yet.

Once you have created the :code:`osld` conda environment (see instructions above) you can install the latest development version on the `GitHub repository <https://github.com/OHBA-analysis/osl-dynamics>`_ with:

.. code::

    conda activate osld
    pip install git+https://github.com/OHBA-analysis/osl-dynamics.git

Copy the source code (optional)
-------------------------------

If you want to make your own changes to osl-dynamics, we recommend you work on a fork of the `GitHub repository <https://github.com/OHBA-analysis/osl-dynamics>`_. A fork is your own copy of the repository on GitHub, so you can save your changes there (and share them with others) without needing write access to the main repository.

First, create a fork by clicking the 'Fork' button at the top right of the `GitHub repository <https://github.com/OHBA-analysis/osl-dynamics>`_ webpage. Then, once you have created the :code:`osld` conda environment (see instructions above), you can install a local copy of your fork into it:

.. code::

    git clone https://github.com/<your-username>/osl-dynamics.git
    conda activate osld
    cd osl-dynamics
    pip install -e .

where :code:`<your-username>` is your GitHub username. Now you can run osl-dynamics with your own local changes to the code.

To get the latest changes from the main repository, add it as a remote called :code:`upstream`:

.. code::

    git remote add upstream https://github.com/OHBA-analysis/osl-dynamics.git

Then you can update your copy with:

.. code::

    git pull upstream main

If you would like your changes to be included in osl-dynamics, push them to your fork and open a pull request on the `GitHub repository <https://github.com/OHBA-analysis/osl-dynamics>`_.

Test your GPUs are working
--------------------------

To check if your GPUs are working:

.. code::

    conda activate osld
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

This should print a list of the GPUs you have available (or an empty list :code:`[]` if there are none).

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
