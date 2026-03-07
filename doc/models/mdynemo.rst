:orphan:

Multi-Dynamic Network Modes (M-DyNeMo)
=======================================

Introduction
------------

M-DyNeMo extends `DyNeMo <dynemo.html>`_ by allowing **separate temporal dynamics for power and functional connectivity (FC)** [1]. Standard DyNeMo assumes that power and FC share the same dynamics — i.e. both are described by the same set of mixing coefficients. M-DyNeMo relaxes this assumption, enabling the model to capture situations where power and connectivity evolve independently.

This is motivated by the observation that power and FC dynamics can be **uncoupled** — they may change at different times or at different rates. For example, a brain region may increase in power without a corresponding change in its connectivity to other regions, or vice versa.

Generative Model
----------------

M-DyNeMo decomposes the time-varying covariance matrix into separate **standard deviation** (power) and **correlation** (FC) components, each with their own set of mixing coefficients:

.. math::
    C_t = E_t R_t E_t,

where:

- :math:`E_t = \sum_j \alpha_{jt} E_j` is the time-varying standard deviation matrix (diagonal), controlling **power**.
- :math:`R_t = \sum_j \beta_{jt} R_j` is the time-varying correlation matrix, controlling **functional connectivity**.
- :math:`\alpha_{jt}` and :math:`\beta_{jt}` are separate sets of mixing coefficients for power and FC respectively.

The key difference from DyNeMo is that :math:`\alpha_{jt}` and :math:`\beta_{jt}` are inferred independently, allowing power and FC to have different temporal dynamics.

Like DyNeMo, the temporal model for both sets of mixing coefficients uses recurrent neural networks (LSTMs) to capture long-range temporal dependencies.

Inference
---------

M-DyNeMo uses the same **amortised variational inference** approach as DyNeMo. An inference RNN takes the observed data and predicts posterior distributions for both sets of logits (power and FC). The cost function is the variational free energy, with **KL annealing** used to stabilize training.

The number of power modes (``n_modes``) and FC modes (``n_corr_modes``) can be set independently, allowing for different numbers of power and connectivity patterns.

M-DyNeMo in osl-dynamics
-------------------------

M-DyNeMo can be trained using the ``osl_dynamics.models.mdynemo`` module. Key configuration options that differ from DyNeMo:

- ``n_modes``: Number of modes for power (standard deviations).
- ``n_corr_modes``: Number of modes for FC (correlations). Defaults to ``n_modes`` if not specified.
- ``learn_means``, ``learn_stds``, ``learn_corrs``: Control which observation model parameters are trainable.

Post-hoc Analysis
-----------------

After fitting M-DyNeMo, you obtain two sets of mixing coefficients:

- **Alpha** (:math:`\alpha_{jt}`): mixing coefficients for power modes.
- **Beta** (:math:`\beta_{jt}`): mixing coefficients for FC modes.

These can be analysed separately to study the independent dynamics of power and connectivity. The same summary statistics (fractional occupancy, lifetimes, etc.) and spectral analysis approaches used with `DyNeMo <dynemo.html>`_ can be applied to each set of mixing coefficients independently.

References
----------

#. R Huang, C Gohil, M W Woolrich, Evidence for Transient, Uncoupled Power and Functional Connectivity Dynamics. `Human Brain Mapping, 2025 <https://pubmed.ncbi.nlm.nih.gov/40035183/>`_.
