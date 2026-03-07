Getting Started
===============

This page provides a quick introduction to osl-dynamics. For installation instructions, see :doc:`Install <install>`.

Overview
--------

osl-dynamics is a Python toolbox for studying dynamic brain networks from neuroimaging data (M/EEG and fMRI). It provides:

- **Generative models** for inferring dynamic brain states: the Hidden Markov Model (HMM), Dynamic Network Modes (DyNeMo), and Dynamic Network States (DyNeSte).
- **Post-hoc analysis tools**: spectral analysis, summary statistics, and network visualization.
- **Static analysis**: time-averaged power and connectivity analysis.
- **Statistical testing**: GLM-based permutation testing for group comparisons.
- **Simulation tools**: for generating synthetic time series data.

A Typical Workflow
------------------

A typical analysis with osl-dynamics follows these steps:

1. **Load data** — Load your preprocessed source-space M/EEG or fMRI data.
2. **Prepare data** — Standardize, and optionally apply time-delay embedding (TDE) and PCA.
3. **Train a model** — Fit an HMM, DyNeMo, or DyNeSte to infer dynamic brain states/modes.
4. **Post-hoc analysis** — Estimate state/mode spectra, compute summary statistics, and visualize networks.
5. **Group analysis** — Compare dynamics across groups or conditions using statistical testing.

Quick Example
-------------

Here is a minimal example of training an HMM using the config API::

    from osl_dynamics.config_api import wrappers

    # Train a TDE-HMM
    model, data = wrappers.train_hmm(
        "training_data",
        output_dir="results",
        config_kwargs={
            "n_states": 8,
            "learn_means": False,
            "learn_covariances": True,
        },
        prepare={
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
            "standardize": {},
        },
    )

Or equivalently, using a YAML config::

    config = """
        load_data:
            inputs: training_data
            prepare:
                tde_pca: {n_embeddings: 15, n_pca_components: 80}
                standardize: {}
        train_hmm:
            config_kwargs:
                n_states: 8
                learn_means: False
                learn_covariances: True
    """

    from osl_dynamics.config_api import run_pipeline
    run_pipeline(config, output_dir="results")

You can also use the command line::

    osl-dynamics config.yml results

For more details on the config API, see the :mod:`osl_dynamics.config_api` API reference.

Step-by-Step with the Data API
------------------------------

For more control, you can use the lower-level API directly::

    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config, Model

    # Load data
    data = Data(["subject1.npy", "subject2.npy", ...])

    # Prepare data
    data.prepare({"tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
                   "standardize": {}})

    # Create and train the model
    config = Config(
        n_states=8,
        n_channels=data.n_channels,
        sequence_length=200,
        learn_means=False,
        learn_covariances=True,
        batch_size=16,
        learning_rate=0.01,
        n_epochs=20,
    )
    model = Model(config)
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
    model.fit(data)

    # Get inferred state probabilities
    alpha = model.get_alpha(data)

Choosing a Model
----------------

The three main models in osl-dynamics are:

+-------------+--------------------+---------------------+-------------------+
| Model       | State type         | Temporal model      | Best for          |
+=============+====================+=====================+===================+
| **HMM**     | Discrete           | Markovian           | Resting-state     |
|             | (mutually          | (transition         | analysis;         |
|             | exclusive)         | probability matrix) | interpretable     |
|             |                    |                     | summary stats     |
+-------------+--------------------+---------------------+-------------------+
| **DyNeMo**  | Continuous         | Non-Markovian       | Task data;        |
|             | (linear mixture    | (RNN)               | overlapping       |
|             | of modes)          |                     | network activity  |
+-------------+--------------------+---------------------+-------------------+
| **DyNeSte** | Discrete           | Non-Markovian       | When you need     |
|             | (mutually          | (RNN)               | both discrete     |
|             | exclusive)         |                     | states and long-  |
|             |                    |                     | range dynamics    |
+-------------+--------------------+---------------------+-------------------+

See the model description pages for more details: :doc:`HMM <models/hmm>`, :doc:`DyNeMo <models/dynemo>`, :doc:`DyNeSte <models/dyneste>`.

Also see the :doc:`FAQ <faq>` for guidance on choosing hyperparameters and other common questions.

What Next?
----------

- **Tutorials**: Work through the :doc:`tutorials <documentation>` for detailed examples covering data loading, model training, and post-hoc analysis.
- **API reference**: Browse the :doc:`API docs <autoapi/index>` for detailed information on all classes and functions.
- **Examples**: See the `examples directory <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples>`_ for more analysis scripts.
- **Workshops**: Check out materials from past `OSL workshops <https://github.com/OHBA-analysis/osl-workshop-2025-dynamics>`_.
