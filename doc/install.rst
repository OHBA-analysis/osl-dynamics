Installation
============

OSL Dynamics can be installed in three steps. Open a Terminal and execute the following commands:

#. Create a virtual environment, we recommend using Anaconda: https://docs.anaconda.com/anaconda/install/index.html.

    Once you have installed Anaconda (or Miniconda) execute:

    ::

        conda create --name osld python=3
        conda activate osld

    Note, this environment must be activated every time you want to use osl-dynamics.

#. Install the deep learning library TensorFlow: https://www.tensorflow.org/overview.

    ::

        pip install tensorflow

    If you are using an Apple computer with an M1/M2 chip the above command won't work, instead you can install TensorFlow with:

    ::

        pip install tensorflow-macos

    If you have GPU resources you need to install additional libraries (CUDA/cuDNN), see https://www.tensorflow.org/install/pip for detailed instructions. You maybe able you install a GPU-enabled version of TensorFlow using Anaconda:

    ::

        conda install -c conda-forge tensorflow-gpu

#. Install osl-dynamics:

    ::

        pip install osl-dynamics

To remove osl-dynamics simply delete the conda environment:

::

    conda env remove -n osld
    conda clean --all
