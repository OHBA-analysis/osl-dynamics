:orphan:

Dynamic Network States (DyNeSte)
================================

Introduction
------------

DyNeSte combines elements of both the `HMM <hmm.html>`_ and `DyNeMo <dynemo.html>`_ to address their inherent trade-offs [1]. The HMM infers categorical brain network states that provide good interpretability but does not model long-range temporal structure due to its Markovian constraint. DyNeMo uses recurrent neural networks to model long-range temporal dependencies, but at the expense of interpretability (due to its mixture description of states).

DyNeSte addresses this by combining:

- **Discrete states** (like the HMM) for interpretability — we can directly calculate summary statistics such as fractional occupancies, lifetimes, and switching rates.
- **Non-Markovian temporal dynamics** (like DyNeMo) for capturing long-range dependencies — using a recurrent neural network to model the temporal evolution of states.

In both simulations and real resting-state magnetoencephalography (MEG) data, DyNeSte was able to recover plausible dynamic brain network states and showed superior performance over the HMM in capturing long-range temporal dependencies in network dynamics [1].

Generative Model
----------------

Mathematically, the generative model (joint probability distribution) is

.. math::
    p(x_{1:T}, s_{1:T}) = p(x_1 | s_1) p(s_1) \prod_{t=2}^T p(x_t | s_t) p(s_t | s_{1:t-1}),

where :math:`x_{1:T}` denotes a sequence of observed data (:math:`x_1, x_2, ..., x_T`) and :math:`s_{1:T}` denotes a sequence of hidden states (:math:`s_1, s_2, ..., s_T`).

The observation model is the same as the HMM — a multivariate normal distribution:

.. math::
    p(x_t | s_t = k) = \mathcal{N}(m_k, C_k),

where :math:`m_k` and :math:`C_k` are state means and covariances and :math:`k` indexes the active state.

The key difference from the HMM is the temporal model: :math:`p(s_t | s_{1:t-1})` depends on the **entire history** of states, not just the previous state. This is achieved using a recurrent neural network (**Model RNN**) that predicts a categorical distribution over states based on the full history:

.. math::
    p(s_t | s_{1:t-1}) = \mathrm{Cat}(\mathrm{softmax}(\theta^{\mathrm{mod}}_t)),

where :math:`\theta^{\mathrm{mod}}_t` are logits predicted by the Model RNN given the history of states :math:`s_{1:t-1}`.

Inference
---------

DyNeSte uses **amortised variational inference** to learn the model parameters. An **inference RNN** takes the observed data as input and predicts the posterior distribution over states at each time point:

.. math::
    q(s_t) = \mathrm{Cat}(\mathrm{softmax}(\theta^{\mathrm{inf}}_t)),

where :math:`\theta^{\mathrm{inf}}_t` are logits predicted by the inference RNN.

To handle the discrete nature of the states during training, DyNeSte uses the **Gumbel-Softmax** reparameterization trick, which provides a differentiable approximation to sampling from a categorical distribution. The temperature of the Gumbel-Softmax distribution can be annealed during training (starting high and decreasing) to produce increasingly discrete state assignments.

Cost Function
^^^^^^^^^^^^^

The cost function used to train DyNeSte is the variational free energy, consisting of:

- A **log-likelihood** term: how well the observation model explains the data given the inferred states.
- A **KL divergence** term: how close the inferred state probabilities (from the inference RNN) are to the prior (from the model RNN).

As with DyNeMo, **KL annealing** can be used to stabilize training — the KL term is slowly turned on over the first part of training.

DyNeSte in osl-dynamics
-----------------------

DyNeSte can be trained using the ``osl_dynamics.models.dyneste`` module. The key configuration options are similar to DyNeMo, with the addition of Gumbel-Softmax temperature annealing parameters:

- ``do_gs_annealing``: Whether to anneal the Gumbel-Softmax temperature.
- ``initial_gs_temperature``: Starting temperature (default: 1.0).
- ``final_gs_temperature``: Final temperature after annealing (default: 0.01).
- ``n_gs_annealing_epochs``: Number of epochs over which to anneal.

Post-hoc Analysis
-----------------

Because DyNeSte infers discrete states, the post-hoc analysis is the same as for the `HMM <hmm.html>`_. You can directly calculate summary statistics (fractional occupancy, mean lifetime, mean interval, switching rate) and use the multitaper approach for spectral analysis.

See the :doc:`HMM post-hoc analysis documentation <hmm>` for details on these analyses.

References
----------

#. S Cho, R Huang, C Gohil, O Parker Jones, M W Woolrich, Modelling Discrete States and Long-Term Dynamics in Functional Brain Networks. `bioRxiv, 2025 <https://www.biorxiv.org/content/10.1101/2025.09.25.678554>`_.
