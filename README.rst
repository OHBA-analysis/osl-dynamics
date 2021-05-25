================================
VariationalRNNAutoDecoder (VRAD)
================================

BMRC Installation
=================
.. code-block:: shell

    module load Anaconda3
    module load cudnn

    conda create --name myenv python=3.8
    conda activate myenv

    cd VRAD
    pip install -e .


Build documentation
===================
.. code-block:: shell

    cd VRAD
    python setup.py docs


Run tests
=========
.. code-block:: shell

    cd VRAD
    python setup.py test
