============
osl-dynamics
============

See the read the docs page for a description of this project: `https://osl-dynamics.readthedocs.io <https://osl-dynamics.readthedocs.io>`_.

Installation (for Developers)
=============================

Here, we describe how to install osl-dynamics from source. We recommend using the conda environment files in ``/envs``. For a generic linux machine, osl-dynamics can be installed in editable mode with:

.. code-block:: shell

    git clone https://github.com/OHBA-analysis/osl-dynamics.git
    cd osl-dynamics
    conda env create -f envs/linux.yml
    conda activate osld
    pip install -e .

If you're installing on the Oxford BMRC server, use ``envs/bmrc.yml``. If you're installing on the OHBA workstation, use ``envs/hbaws.yml``.

Note, the conda environments use ``pip`` to install TensorFlow, you may need to load/install additional libraries (such as CUDA/cuDNN) if you have GPU support.

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

Formatting and Conventions
--------------------------

We use the python code formatter ``black`` to give a consistent code layout in our source files. To install:

.. code-block:: shell

    conda activate <env>
    pip install black

To format a source file:

.. code-block:: shell

    black <filename>.py

Please run ``black`` on any edited files before commiting changes.

Git Workflow
------------

We use git for version control. There is one ``main`` branch. To add changes:

Create a feature branch for changes:

.. code-block:: shell

    git checkout main
    git pull
    git checkout -b <branch-name>

Make changes to file and commit it to the branch:

.. code-block:: shell

    git add <file>
    git commit -m "Short description of changes"

When writing commit messages please follow the conventions `here <https://www.conventionalcommits.org/en/v1.0.0-beta.2/#specification>`_.

Then either push the new branch to the remote repository:

.. code-block:: shell

    git push --set-upstream origin <branch-name>

and create a pull request (recommended), or merge branch into ``main`` and push:

.. code-block:: shell

    git checkout main
    git merge <branch-name>
    git push
