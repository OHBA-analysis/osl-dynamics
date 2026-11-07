Dynamic Network States (DyNeSte)
================================

Introduction
------------

DyNeSte combines the strengths of the :doc:`HMM <hmm>` and :doc:`DyNeMo <dynemo>` [1]: the HMM's **discrete states**, which are interpretable and give summary statistics directly (fractional occupancies, lifetimes, switching rates), with DyNeMo's **non-Markovian temporal model**, a recurrent neural network that captures long-range dependencies the HMM's Markovian constraint cannot. On resting-state MEG data it recovers plausible network states and captures long-range temporal structure better than the HMM [1].

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

Post-hoc Analysis
-----------------

Because DyNeSte infers discrete states, the post-hoc analysis is the same as for the :doc:`HMM <hmm>`: summary statistics (fractional occupancy, mean lifetime, mean interval, switching rate) and multitaper spectral analysis.

References
----------

#. S Cho, R Huang, C Gohil, O Parker Jones, M W Woolrich, Modelling Discrete States and Long-Term Dynamics in Functional Brain Networks. `Imaging Neuroscience, 2026 <https://doi.org/10.1162/IMAG.a.1237>`_.
