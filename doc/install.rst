Installation
============

Linux Instructions
------------------

OSL Dynamics can be installed in three steps. Open a Terminal and execute the following commands:

#. Create a virtual environment, we recommend using Anaconda: https://docs.anaconda.com/anaconda/install/index.html.

    Once you have installed Anaconda (or Miniconda) execute:

    ::

        conda create --name osld python=3.10.14
        conda activate osld

    Note, this environment must be activated every time you want to use osl-dynamics.

#. Install the deep learning library TensorFlow: https://www.tensorflow.org/overview (and the addon tensorflow-probability).

    To install TensorFlow use:

    ::

        pip install tensorflow==2.11.0

    If you have GPU resources you may need to install additional libraries (CUDA/cuDNN), see https://www.tensorflow.org/install/pip for detailed instructions.

    If you are using an Apple Mac, you will need to use the following instead:

    ::

        pip install tensorflow-macos==2.11.0

    You may also need to install the following package to get your GPUs working on a Mac:

    ::

        pip install tensorflow-metal==0.7.0

    See https://developer.apple.com/metal/tensorflow-plugin/ for further details.

    If pip can not find the package, then you can try installing TensorFlow with conda:

    ::

        conda install tensorflow=2.11.0

    After you have installed TensorFlow, install the tensorflow-probability addon with:

    ::

        pip install tensorflow-probability==0.19.0

#. Finally, install osl-dynamics:

    ::

        pip install osl-dynamics

To remove osl-dynamics simply delete the conda environment:

::

    conda env remove -n osld
    conda clean --all

Windows Instructions
--------------------

If you are using a Windows computer, we recommend first installing linux (Ubuntu) as a Windows Subsystem by following the instructions `here <https://ubuntu.com/wsl>`_. Then following the instructions above in the Ubuntu terminal.

TensorFlow Versions
-------------------

osl-dynamics has been tested with the following versions:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - tensorflow
     - tensorflow-probability
   * - 2.11
     - 0.19
   * - 2.12
     - 0.19
   * - 2.13
     - 0.20
   * - 2.14
     - 0.22
   * - 2.15
     - 0.22

Test your GPUs are working
--------------------------

You can use the following to check if TensorFlow is using any GPUs you have available:

::

    conda activate osld
    python
    >> import tensorflow as tf
    >> print(tf.test.is_gpu_available())

This should print :code:`True` if you have GPUs available (and :code:`False` otherwise).

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
