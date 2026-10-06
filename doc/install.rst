Installation
============

osl-dynamics is available on `conda-forge <https://anaconda.org/conda-forge/osl-dynamics>`_ and `PyPI <https://pypi.org/project/osl-dynamics>`_.

Conda (recommended)
-------------------

If you do not already have conda, install `Miniforge <https://conda-forge.org/download/>`_:

.. code::

    curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    rm Miniforge3-$(uname)-$(uname -m).sh

Then create an environment with osl-dynamics in it:

.. code::

    conda create -n osld -c conda-forge osl-dynamics
    conda activate osld

This installs osl-dynamics with everything it needs, TensorFlow included. Add :code:`jupyterlab` to that command if you want to work in notebooks.

Pip
---

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

BMRC Cluster (Oxford)
---------------------

See the instructions on the GitHub `README <https://github.com/OHBA-analysis/osl-dynamics>`_.

Install the latest development code (optional)
----------------------------------------------

You should only need to do this if you need a feature or fix that has not been released yet.

Once you have created the :code:`osld` conda environment (see instructions above) you can install the latest development version on the `GitHub repository <https://github.com/OHBA-analysis/osl-dynamics>`_ with:

.. code::

    conda activate osld
    pip install git+https://github.com/OHBA-analysis/osl-dynamics.git

Copy the source code (optional)
-------------------------------

Once you have created the :code:`osld` conda environment (see instructions above) you can install a local copy of the source code (`GitHub repository <https://github.com/OHBA-analysis/osl-dynamics>`_) into it.

.. code::

    git clone https://github.com/OHBA-analysis/osl-dynamics.git
    conda activate osld
    cd osl-dynamics
    pip install -e .

Now you can run osl-dynamics with your own local changes to the code.

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
