Multi-Dynamic Network Modes (M-DyNeMo)
=======================================

Introduction
------------

M-DyNeMo extends :doc:`DyNeMo <dynemo>` with **separate temporal dynamics for power and functional connectivity (FC)** [1]. DyNeMo describes both with the same mixing coefficients; M-DyNeMo relaxes this, so that a region's power can change without a change in its connectivity, or vice versa.

Generative Model
----------------

M-DyNeMo decomposes the time-varying covariance matrix into separate **standard deviation** (power) and **correlation** (FC) components, each with their own set of mixing coefficients:

.. math::
    C_t = E_t R_t E_t,

where:

- :math:`E_t = \sum_j \alpha_{jt} E_j` is the time-varying standard deviation matrix (diagonal), controlling **power**.
- :math:`R_t = \sum_j \beta_{jt} R_j` is the time-varying correlation matrix, controlling **functional connectivity**.
- :math:`\alpha_{jt}` and :math:`\beta_{jt}` are separate sets of mixing coefficients for power and FC respectively.

As in DyNeMo, each set of mixing coefficients has a recurrent neural network as its temporal model.

Inference
---------

M-DyNeMo uses the same **amortised variational inference** approach as DyNeMo. An inference RNN takes the observed data and predicts posterior distributions for both sets of logits (power and FC). The cost function is the variational free energy, with **KL annealing** used to stabilize training.

The number of power modes (``n_modes``) and FC modes (``n_corr_modes``) can be set independently, allowing for different numbers of power and connectivity patterns.

Post-hoc Analysis
-----------------

After fitting M-DyNeMo, you obtain two sets of mixing coefficients:

- **Alpha** (:math:`\alpha_{jt}`): mixing coefficients for power modes.
- **Beta** (:math:`\beta_{jt}`): mixing coefficients for FC modes.

Each can be analysed with the summary statistics and spectral analysis used for :doc:`DyNeMo <dynemo>`.

References
----------

#. R Huang, C Gohil, M W Woolrich, Evidence for Transient, Uncoupled Power and Functional Connectivity Dynamics. `Human Brain Mapping, 2025 <https://doi.org/10.1002/hbm.70179>`_.
