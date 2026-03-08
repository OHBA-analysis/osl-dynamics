:orphan:

HIVE (HMM with Integrated Variability Estimation)
==================================================

Introduction
------------

HIVE extends the `HMM <hmm.html>`_ to explicitly model **inter-session variability** while inferring dynamic networks across a group [1]. The standard HMM assumes all sessions share exactly the same observation model parameters (state means and covariances). In practice, sessions from different subjects, scanners, or sites may have systematic differences in their functional networks. HIVE addresses this by learning an **embedding vector** (a "fingerprint") for each session, which modulates the group-level observation model parameters to produce session-specific parameters.

This approach has two key benefits:

- **Improved model fit**: By accounting for inter-session variability, HIVE can better describe the data compared to a standard HMM that forces all sessions to share identical parameters.
- **Meaningful embeddings**: The learnt embedding vectors can capture sources of variation across a population (e.g. demographics, scanner types, sites) in an unsupervised manner.

Generative Model
----------------

HIVE builds on the HMM generative model. The key addition is a **variability encoding block** that takes session-specific embedding vectors and produces deviations from the group-level observation model parameters.

For session :math:`n`, the observation model is:

.. math::
    p(x_t | s_t = k, n) = \mathcal{N}(m_k + \Delta m_{k,n}, ~ C_k + \Delta C_{k,n}),

where:

- :math:`m_k` and :math:`C_k` are the **group-level** state means and covariances (shared across all sessions).
- :math:`\Delta m_{k,n}` and :math:`\Delta C_{k,n}` are **session-specific deviations**, predicted by a multi-layer perceptron (MLP) that takes the session's embedding vector as input.

The temporal model (transition probability matrix) is the same as the standard HMM.

Inference
---------

HIVE uses the same Bayesian inference approach as the HMM (Expectation-Maximization) for the hidden states and transition probabilities. The embedding vectors and the deviation MLP parameters are learnt jointly during training. KL annealing can be used to stabilize the learning of the deviation parameters.

Key configuration options specific to HIVE:

- ``embeddings_dim``: Dimensionality of the embedding vectors.
- ``spatial_embeddings_dim``: Dimensionality of spatial embeddings.
- ``dev_n_layers``, ``dev_n_units``: Architecture of the MLP that predicts deviations.

Post-hoc Analysis
-----------------

After fitting HIVE, you can perform the same post-hoc analyses as with the `HMM <hmm.html>`_ (summary statistics, spectral analysis, etc.). Additionally, you can analyse the **embedding vectors** to study inter-session variability:

- Visualize the embedding space to see how sessions cluster.
- Correlate embeddings with external variables (age, sex, scanner type, etc.).
- Examine the session-specific deviations to understand how individual sessions differ from the group average.

References
----------

#. R Huang, C Gohil, M W Woolrich, Modelling variability in functional brain networks using embeddings. `bioRxiv, 2024 <https://www.biorxiv.org/content/10.1101/2024.01.29.577718>`_.
