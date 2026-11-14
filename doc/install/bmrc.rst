Installing on BMRC (Oxford)
===========================

On the Biomedical Research Computing (BMRC) cluster, :code:`conda` is available as a software module:

.. code::

    module load Miniforge3

and osl-dynamics can be installed with:

.. code::

    mamba create -n osld -c conda-forge osl-dynamics
    conda activate osld

The above can be run on the login nodes (:code:`clusterX.bmrc.ox.ac.uk`). On :code:`compg017` you will need to set the following to use conda:

.. code::

    unset https_proxy http_proxy no_proxy HTTPS_PROXY HTTP_PROXY NO_PROXY
