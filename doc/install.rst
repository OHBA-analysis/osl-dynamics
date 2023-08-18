Installation
============

Instructions
------------

OSL Dynamics can be installed in three steps. Open a Terminal and execute the following commands:

#. Create a virtual environment, we recommend using Anaconda: https://docs.anaconda.com/anaconda/install/index.html.

    Once you have installed Anaconda (or Miniconda) execute:

    ::

        conda create --name osld python=3.10
        conda activate osld

    Note, this environment must be activated every time you want to use osl-dynamics.

#. Install the deep learning library TensorFlow: https://www.tensorflow.org/overview (and the addon tensorflow-probability).

    To install TensorFlow use:

    ::

        pip install tensorflow==2.9.1

    If you have GPU resources you may need to install additional libraries (CUDA/cuDNN), see https://www.tensorflow.org/install/pip for detailed instructions.

    If you are using an Apple computer with an M1/M2 chip, you will need to use the following instead:

    ::

        pip install tensorflow-macos==2.9.1

    After you have installed TensorFlow, install the tensorflow-probability addon with:

    ::

        pip install tensorflow-probability==0.17

#. Finally, install osl-dynamics:

    ::

        pip install osl-dynamics

To remove osl-dynamics simply delete the conda environment:

::

    conda env remove -n osld
    conda clean --all


Training Speed
--------------

You can test if you've succesfully installed osl-dynamics by running the HMM and DyNeMo simulation example scripts:

- `HMM example <https://github.com/OHBA-analysis/osl-dynamics/blob/main/examples/simulation/hmm_hmm-mvn.py>`_.
- `DyNeMo example <https://github.com/OHBA-analysis/osl-dynamics/blob/main/examples/simulation/dynemo_hmm-mvn.py>`_.

A rough indication of the expected training speeds is given below. You could expect variations up to a factor of 2.

.. list-table:: Training speed: **ms/step**
   :widths: 25 25 25
   :header-rows: 1

   * - Computer
     - HMM
     - DyNeMo
   * - M1/M2 Macbook
     - 50
     - 60
   * - Linux with 1 GPU
     - 100
     - 20
   * - Linux CPU
     - 100
     - 100
