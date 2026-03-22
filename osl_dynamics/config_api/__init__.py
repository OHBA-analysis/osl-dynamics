"""Config API for specifying and running pipelines.

The config API allows you to define a full analysis pipeline (data loading,
preparation, and model training) as a YAML config and run it in a single
call. For example, to train a TDE-HMM::

    config = '''
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
    '''

    run_pipeline(config, output_dir="results")

Command line usage
------------------
The config API can also be used from the command line::

    osl-dynamics <config-file> <output-directory>

where ``<config-file>`` is a YAML file containing the config and
``<output-directory>`` is the output directory.

To restrict to a specific GPU::

    osl-dynamics <config-file> <output-directory> --restrict 0

Remember to activate the ``osld`` conda environment first.

Python example scripts
----------------------

Code to reproduce the `eLife paper <https://elifesciences.org/articles/91949>`_ results can be found `here <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/toolbox_paper>`_. The different pipelines are:

- `TDE-HMM burst analysis <https://github.com/OHBA-analysis/osl-dynamics/blob/main/examples/toolbox_paper/ctf_rest/tde_hmm_bursts.py>`_
- `AE-HMM network analysis <https://github.com/OHBA-analysis/osl-dynamics/blob/main/examples/toolbox_paper/elekta_task/ae_hmm.py>`_
- `TDE-HMM network analysis <https://github.com/OHBA-analysis/osl-dynamics/blob/main/examples/toolbox_paper/ctf_rest/tde_hmm_networks.py>`_
- `TDE-DyNeMo network analysis <https://github.com/OHBA-analysis/osl-dynamics/blob/main/examples/toolbox_paper/ctf_rest/tde_dynemo_networks.py>`_

Modules
-------
- ``pipeline.py`` — :py:func:`run_pipeline` for executing a config.
- ``wrappers.py`` — Wrapper functions for each pipeline step.
"""

from osl_dynamics.config_api import pipeline, wrappers
