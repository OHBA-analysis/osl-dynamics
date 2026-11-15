HIVE (HMM with Integrated Variability Estimation)
==================================================

Introduction
------------

HIVE extends the :doc:`HMM <hmm>` to model **inter-session variability** while inferring dynamic networks across a group [1]. The HMM gives every session the same state means and covariances, although sessions from different subjects, scanners or sites differ systematically. HIVE learns an **embedding vector** (a "fingerprint") for each session that modulates the group-level parameters into session-specific ones. This fits the data better, and the embeddings capture sources of variation across the population (demographics, scanner, site) without supervision.

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

HIVE infers the hidden states and transition probabilities as the HMM does, and learns the embedding vectors and the deviation MLP jointly during training, with KL annealing to stabilize the deviations. The embedding and deviation-network sizes are set in :class:`osl_dynamics.models.hive.Config`.

Post-hoc Analysis
-----------------

The post-hoc analyses of the :doc:`HMM <hmm>` apply. In addition, the **embedding vectors** can be visualized to see how sessions cluster, correlated with external variables (age, sex, scanner), and the session-specific deviations examined to see how a session differs from the group.

References
----------

#. R Huang, C Gohil, M W Woolrich, Modelling variability in functional brain networks using embeddings. `Imaging Neuroscience, 2026 <https://doi.org/10.1162/IMAG.a.1188>`_.
