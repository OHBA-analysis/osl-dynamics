============
osl-dynamics
============

Models for analysing neuroimaging data developed by the Oxford Centre for Human Brain Activity (OHBA) group at the University of Oxford.

Models included:
    - Hidden Markov Model (HMM).
    - Dynamic Network Modes (DyNeMo).
    - Multi-Dynamic Network Modes (M-DyNeMo).
    - Dynamic Network States (DyNeSt).
    - Single-dynamic Adversarial Generator Encoder (SAGE).
    - Multi-dynamic Adversarial Generator Encoder (MAGE).

Installation
============
.. code-block:: shell

    git clone git@github.com:OHBA-analysis/osl-dynamics.git
    cd osl-dynamics
    pip install -e .

To use the HMM you also need to install armadillo:

.. code-block:: shell

    conda install -c conda-forge armadillo

See CONTRIBUTION.md for further details.

Documentation
=============

The documentation is hosted on Read the Docs: `https://osl-dynamics.readthedocs.io <https://osl-dynamics.readthedocs.io>`_. To compile locally use:

.. code-block:: shell

    cd osl-dynamics
    python setup.py build_sphinx


See CONTRIBUTION.md for further details.
