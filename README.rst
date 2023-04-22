============
osl-dynamics
============

See the read the docs page for a description of this project: `https://osl-dynamics.readthedocs.io <https://osl-dynamics.readthedocs.io>`_.

Installation
============

Here, we describe how to install osl-dynamics from source. We recommend using the conda environment files in ``/envs``. For a generic linux machine, osl-dynamics can be installed in editable mode with:

.. code-block:: shell

    git clone https://github.com/OHBA-analysis/osl-dynamics.git
    cd osl-dynamics
    conda env create -f envs/linux.yml
    conda activate osld
    pip install -e .

If you're using a Mac then use the ``envs/mac.yml`` file to create the conda environment instead of ``envs/linux.yml``. Note, the conda environments use ``pip`` to install TensorFlow, you may need to load/install additional libraries (such as CUDA/cuDNN) if you have GPU support.

Developers might want to clone the repo using SSH instead of HTTPS:

.. code-block:: shell

    git clone git@github.com:OHBA-analysis/osl-dynamics.git

Oxford specific computers
-------------------------

If you're installing on the Oxford BMRC server, use ``envs/bmrc.yml``. If you're installing on the OHBA workstation, use ``envs/hbaws.yml``. Note, the ``hbaws.yml`` environment will automatically install spyder and jupyter notebooks.

Installing within an osl environment
------------------------------------

If you have already installed `OSL <https://github.com/OHBA-analysis/osl>`_ you can install osl-dynamics in the ``osl`` environment with:

.. code-block:: shell

    conda activate osl
    cd osl-dynamics
    pip install tensorflow
    pip install -e .

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

The process of packaging a python project is described here: `https://packaging.python.org/en/latest/tutorials/packaging-projects <https://packaging.python.org/en/latest/tutorials/packaging-projects>`_.

A couple packages are needed to build and upload a project to PyPI, these can be installed in your conda environment with:

.. code-block:: shell

    pip install build twine

The following steps can be used to release a new version:

#. Update the version number on line 5 of ``setup.cfg``.
#. Commit the updated setup.cfg to the ``main`` branch of the GitHub repo.
#. Delete any old distributions that have been built (if there are any): ``rm -r dist``.
#. Build a distribution in the osl-dynamics root directory with ``python -m build``. This will create a new directory called ``dist``.
#. Test the build by installing in a test conda environment with ``cd dist; pip install <build>.whl``.
#. Upload the distribution to PyPI with ``twine upload dist/*``. You will need to enter the username and password that you used to register with `https://pypi.org <https://pypi.org>`_.
#. Tag the commit uploaded to PyPI with the version number using the 'Create a new release' link on the right of the GitHub repo webpage.

The uploaded distribution will then be available to be installed with:

.. code-block:: shell

    pip install osl-dynamics

Editing Source Code
===================

See `here <https://github.com/OHBA-analysis/osl-dynamics/blob/main/doc/using_bmrc.rst>`_ for useful info regarding how to use the BMRC cluster and how to edit the source code.
