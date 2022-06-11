Development
===========

If you want to request new features or if you’re confident that you have found a bug, please create a new issue on the `GitHub issues <https://github.com/OHBA-analysis/osl-dynamics/issues>`_ page.

Current Status
--------------
This package is under active development. Some of the models and analysis functions are still a work in progress.
The completed models (i.e. those without outstanding development plans) are:

- Dynamic Network Modes (DyNeMo).
- Single-dynamic Adversarial Generator Encoder (SAGE).

The remaining models are still a work in progress, we summarise their status below:

- Hidden Markov Model (HMM).

    - The core model has been written.
    - Improvements in parameter intialisation and removal of the C library is planned.

- Dynamic Network States (DyNeSt).

    - Core model has been written.
    - Has not been validated or tested.

- Multi-Dynamic Network Modes (M-DyNeMo).

    - Core model has been written.
    - The model is currently being used to study MEG data. Results have not been finalised.

- Multi-dynamic Adversarial Generator Encoder (MAGE).

    - Core model has been added.
    - This model is currently being tested.
    - A full pipeline example scripts using the model to study fMRI data will be added.

Contribution Guide
------------------
Developers may find the `contribution guide <https://github.com/OHBA-analysis/osl-dynamics/blob/main/CONTRIBUTION.md>`_ helpful.

Development Team
----------------
The core development team is:

.. include:: ../AUTHORS.rst 

Please email chetan.gohil@psych.ox.ac.uk if you'd like to get in touch with us.
