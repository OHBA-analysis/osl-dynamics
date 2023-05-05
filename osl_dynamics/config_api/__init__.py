"""Config API.

See the `toolbox examples
<https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/toolbox_paper>`_
for scripts that use the config API.

Note, the config API can be used via the command line with::

    % osld-pipeline <config-file> <output-directory> <restrict>

where

- :code:`<config-file>` is a yaml file containing the config.
- :code:`<output-directory>` is the output directory.

Optionally, you can specify a particular GPU to use with::

    % osld-pipeline <config-file> <output-directory> --restrict <restrict>

where :code:`<restrict>` is an integer specifying the GPU number. E.g. if you would
just like to use the first GPU, you can pass::

    % osld-pipeline <config-file> <output-directory> --restrict 0

Remember you need to activate the :code:`osld` conda environment to use the
command line interface.
"""

from osl_dynamics.config_api import pipeline, wrappers
