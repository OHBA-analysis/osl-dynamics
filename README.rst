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

See osl-dynamics/contribution.md for further details.

Build documentation
===================
.. code-block:: shell

    cd osl-dynamics
    python setup.py docs
