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
