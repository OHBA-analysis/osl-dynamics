============
osl-dynamics
============

See the read the docs page for a description of this project: `https://osl-dynamics.readthedocs.io <https://osl-dynamics.readthedocs.io>`_.

Citation
========

If you find this toolbox useful, please cite:

    **Chetan Gohil, Rukuang Huang, Evan Roberts, Mats WJ van Es, Andrew J Quinn, Diego Vidaurre, Mark W Woolrich (2024) osl-dynamics, a toolbox for modeling fast dynamic brain activity eLife 12:RP91949.**

Installation
============

Conda
-----

We recommend installing osl-dynamics within a virtual environment. You can do this with `Anaconda <https://docs.anaconda.com/free/anaconda/install/index.html>`_ (or `miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_).

Below we describe how to install osl-dynamics from source. We recommend using the conda environment files in ``/envs``.

Linux
-----

.. code-block:: shell

    git clone https://github.com/OHBA-analysis/osl-dynamics.git
    cd osl-dynamics
    conda env create -f envs/linux.yml
    conda activate osld
    pip install -e .

Mac
---

For a Mac, the installation of TensorFlow is slightly different to a Linux computer. We recommend using the lines above replacing the Linux environment file ``envs/linux.yml`` with the Mac environment file ``envs/mac.yml``.

Windows
-------

If you are using a Windows computer, we recommend first installing Linux (Ubuntu) as a Windows Subsystem by following the instructions `here <https://ubuntu.com/wsl>`_. Then following the instructions above in the Ubuntu terminal.

Within an osl environment
-------------------------

If you have already installed `OSL <https://github.com/OHBA-analysis/osl>`_ you can install osl-dynamics in the ``osl`` environment with:

.. code-block:: shell

    conda activate osl
    cd osl-dynamics
    pip install tensorflow==2.11.0
    pip install tensorflow-probability==0.19.0
    pip install -e .

Note, if you're using a Mac computer you need to install TensorFlow with ``pip install tensorflow-macos==2.11.0`` instead of ``tensorflow==2.11.0``.

Removing osl-dynamics
---------------------

Simply delete the conda environment and repository:

.. code-block:: shell

    conda env remove -n osld
    rm -rf osl-dynamics

Documentation
=============

The read the docs page should be automatically updated whenever there's a new commit on the ``main`` branch.

The documentation is included as docstrings in the source code. Please write docstrings to any classes or functions you add following the `numpy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_. The API reference documentation will only be automatically generated if the docstrings are written correctly. The documentation directory ``/doc`` also contains ``.rst`` files that provide additional info regarding installation, development, the models, etc.

To compile the documentation locally you need to install the required packages (sphinx, etc.) in your conda environment:

.. code-block:: shell

    cd osl-dynamics
    pip install -r doc/requirements.txt

To compile the documentation locally use:

.. code-block:: shell

    python setup.py build_sphinx

The local build of the documentation webpage can be found in ``build/sphinx/html/index.html``.

Releases
========

A couple packages are needed to build and upload a project to PyPI, these can be installed in your conda environment with:

.. code-block:: shell

    pip install build twine

The following steps can be used to release a new version:

#. Update the version on line 5 of ``setup.cfg`` by removing ``dev`` from the version number.
#. Commit the updated setup.cfg to the ``main`` branch of the GitHub repo.
#. Delete any old distributions that have been built (if there are any): ``rm -r dist``.
#. Build a distribution in the osl-dynamics root directory with ``python -m build``. This will create a new directory called ``dist``.
#. Test the build by installing in a test conda environment with ``cd dist; pip install <build>.whl``.
#. Upload the distribution to PyPI with ``twine upload dist/*``. You will need to enter the username and password that you used to register with `https://pypi.org <https://pypi.org>`_.
#. Tag the commit uploaded to PyPI with the version number using the 'Create a new release' link on the right of the GitHub repo webpage.
#. Change the version to ``X.Y.devZ`` in ``setup.cfg`` and commit the new dev version to ``main``.

The uploaded distribution will then be available to be installed with:

.. code-block:: shell

    pip install osl-dynamics

Editing Source Code
===================

See `here <https://github.com/OHBA-analysis/osl-dynamics/blob/main/doc/using_bmrc.rst>`_ for useful info regarding how to use the BMRC cluster and how to edit the source code.
