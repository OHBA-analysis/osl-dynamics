Getting Started
===============

This page provides a quick introduction to osl-dynamics. For installation instructions, see :doc:`Install <install>`.

Overview
--------

osl-dynamics is a Python toolbox for studying dynamic brain networks from neuroimaging data (M/EEG and fMRI). It provides:

- **M/EEG processing**: preprocessing, source reconstruction (LCMV beamformer) and parcellation (see :doc:`M/EEG Processing <meeg>`).
- **Generative models** for inferring dynamic brain networks, e.g. the Hidden Markov Model (HMM) and Dynamic Network Modes (DyNeMo) (see :doc:`Models <models/index>`).
- **Post-hoc analysis tools**: spectral analysis, summary statistics, and network visualization.
- **Static analysis**: time-averaged power and connectivity analysis.
- **Statistical testing**: GLM permutation tests for group and condition comparisons, network responses and regressions, with maximum statistic, cluster-based and TFCE corrections for multiple comparisons, and effect sizes with confidence intervals (see :doc:`Statistical Significance Testing <stats>`).
- **Simulation tools**: for generating synthetic time series data.

M/EEG Processing
-----------------

If you are starting from raw M/EEG data, osl-dynamics includes a full processing pipeline covering preprocessing, surface extraction, coregistration, source reconstruction (LCMV beamformer), and parcellation. The :doc:`M/EEG Processing <meeg>` page describes the steps and the software behind them. See the :doc:`MEG Preprocessing tutorial <tutorials_build/0-1_meg_preprocessing>` for a step-by-step walkthrough, or the `batch processing scripts <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/meg_preproc>`_ for running multiple sessions in parallel.

A Typical Workflow
------------------

Once you have parcellated source-space data (or fMRI data), a typical analysis follows these steps:

1. **Load data**: Load your parcellated (or sensor) M/EEG or fMRI data (:doc:`Data tutorials <tutorial_sections/data>`).
2. **Prepare data**: Standardize, and optionally apply time-delay embedding (TDE) and PCA (:doc:`Data tutorials <tutorial_sections/data>`).
3. **Train a model**: Fit an HMM, DyNeMo, or DyNeSte to infer dynamic brain states/modes (:doc:`Dynamic Modelling tutorials <tutorial_sections/dynamic_modelling>`).
4. **Post-hoc analysis**: Estimate state/mode spectra, compute summary statistics, and visualize networks (:doc:`HMM <tutorial_sections/hmm_post_hoc>` and :doc:`DyNeMo <tutorial_sections/dynemo_post_hoc>` post-hoc tutorials).
5. **Group analysis**: Compare dynamics across groups or conditions using statistical testing (:doc:`Statistical Significance Testing tutorials <tutorial_sections/statistics>`).

Quick Example
-------------

Here is a minimal example of training an HMM using the Data API::

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
        batch_size=256,
        learning_rate=0.01,
        n_epochs=20,
    )
    model = Model(config)
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
    model.fit(data)

    # Get inferred state probabilities
    alpha = model.get_alpha(data)

Statistical significance testing works on any subject-level quantity. For example, to compare the fractional occupancy of each state between two groups of subjects::

    from osl_dynamics import stats

    # fo has shape (n_subjects, n_states), assignments is a group label per subject
    result = stats.group_diff(fo, assignments, n_perm=1000)
    print(result.pvalues)        # corrected for multiple comparisons across states
    print(result.effect_size())  # Cohen's d with confidence intervals

Using the Config API
--------------------

For a more streamlined approach, you can use the Config API with a YAML config::

    from osl_dynamics.config_api import run_pipeline

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

    run_pipeline(config, output_dir="results")

You can also run this from the command line::

    osl-dynamics config.yml results

where ``config.yml`` is a YAML file containing the pipeline configuration, e.g.:

.. code-block:: yaml

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

and ``results`` is the output directory.

The YAML config specifies a pipeline of functions to run. Each top-level key (e.g. ``load_data``, ``train_hmm``) corresponds to a function in :mod:`osl_dynamics.config_api.wrappers`, and each value holds its keyword arguments. Post-hoc analysis steps such as ``multitaper_spectra``, ``plot_group_tde_hmm_networks`` and ``compare_groups_hmm_summary_stats`` (a group comparison of the summary statistics) can be added in the same way. See the :mod:`wrappers API reference <osl_dynamics.config_api.wrappers>` for the full list of available functions and their arguments.

What Next?
----------

- **Tutorials**: Work through the :doc:`tutorials <tutorials>` for detailed examples covering M/EEG processing, data loading, model training, post-hoc analysis and statistical testing, with links to example scripts and workshop materials.
- **User Guide**: Read the :doc:`User Guide <documentation>` for how the :doc:`processing pipeline <meeg>`, the :doc:`models <models/index>` and the :doc:`statistical tests <stats>` work and when to use which.
- **FAQ**: Check the :doc:`FAQ <faq>` for common questions about preprocessing, choosing a model and its hyperparameters, and interpreting results.
- **API reference**: Browse the :doc:`API docs <autoapi/index>` for detailed information on all classes and functions.
