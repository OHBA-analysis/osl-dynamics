"""Config API.

------

Specify a pipeline using a config, e.g. to train a TDE-HMM::

    config = ```
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
    ```

and run with::

    run_pipeline(config, output_dir="results")

------

See the `toolbox paper examples <https://github.com/OHBA-analysis/osl-dynamics\
/tree/main/examples/toolbox_paper>`_ for scripts that use the config API:

- `TDE-HMM burst analysis <https://github.com/OHBA-analysis/osl-dynamics\
  /blob/main/examples/toolbox_paper/ctf_rest/tde_hmm_bursts.py>`_.
- `AE-HMM network analysis <https://github.com/OHBA-analysis/osl-dynamics\
  /blob/main/examples/toolbox_paper/elekta_task/ae_hmm.py>`_.
- `TDE-HMM network analysis <https://github.com/OHBA-analysis/osl-dynamics\
  /blob/main/examples/toolbox_paper/ctf_rest/tde_hmm_networks.py>`_.
- `TDE-DyNeMo network analysis <https://github.com/OHBA-analysis/osl-dynamics\
  /blob/main/examples/toolbox_paper/ctf_rest/tde_dynemo_networks.py>`_.

------

Note
----
The config API can be used via the command line with::

    % osl-dynamics <config-file> <output-directory>

where

- :code:`<config-file>` is a yaml file containing the config.
- :code:`<output-directory>` is the output directory.

Optionally, you can specify a particular GPU to use with::

    % osl-dynamics <config-file> <output-directory> --restrict <restrict>

where :code:`<restrict>` is an integer specifying the GPU number. E.g. if you
would just like to use the first GPU, you can pass::

    % osl-dynamics <config-file> <output-directory> --restrict 0

Remember you need to activate the :code:`osld` conda environment to use the
command line interface.
"""

from osl_dynamics.config_api import pipeline, wrappers
