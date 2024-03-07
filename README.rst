============
osl-dynamics
============

See the read the docs page for a description of this project: `https://osl-dynamics.readthedocs.io <https://osl-dynamics.readthedocs.io>`_.

Citation
========

If you find this toolbox useful, please cite:

    **Gohil C., Huang R., Roberts E., van Es M.W.J., Quinn A.J., Vidaurre D., Woolrich M.W. (2023) osl-dynamics: A toolbox for modelling fast dynamic brain activity. eLife 12:RP91949 https://doi.org/10.7554/eLife.91949.2**

Installation
============

Conda
-----

We recommend installing osl-dynamics within a virtual environment. You can do this with `Anaconda <https://docs.anaconda.com/free/anaconda/install/index.html>`_ (or `miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_).

Linux
-----

Here, we describe how to install osl-dynamics from source. We recommend using the conda environment files in ``/envs``. For a generic linux machine, osl-dynamics can be installed in editable mode with:

.. code-block:: shell

    git clone https://github.com/OHBA-analysis/osl-dynamics.git
    cd osl-dynamics
    conda env create -f envs/linux.yml
    conda activate osld
    pip install -e .

Note, if you have a Mac you may want to use the ``envs/mac.yml`` environment file instead.

Windows
-------

If you are using a Windows computer, we recommend first installing linux (Ubuntu) as a Windows Subsystem by following the instructions `here <https://ubuntu.com/wsl>`_. Then following the instructions above in the Ubuntu terminal.

Oxford specific computers
-------------------------

If you're installing on the Oxford BMRC server, use ``envs/bmrc.yml``. If you're installing on the OHBA workstation, use ``envs/hbaws.yml``. Note, the ``hbaws.yml`` environment will automatically install spyder and jupyter notebooks.

Within an osl environment
-------------------------

If you have already installed `OSL <https://github.com/OHBA-analysis/osl>`_ you can install osl-dynamics in the ``osl`` environment with:

.. code-block:: shell

    conda activate osl
    cd osl-dynamics
    pip install tensorflow==2.9.1
    pip install tensorflow-probability==0.17
    pip install -e .

Developers
----------

Developers might want to clone the repo using SSH instead of HTTPS:

.. code-block:: shell

    git clone git@github.com:OHBA-analysis/osl-dynamics.git


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
