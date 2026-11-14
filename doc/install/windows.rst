Installing on Windows
=====================

conda-forge does not provide TensorFlow for Windows, so the full :code:`osl-dynamics` conda package is only available for Linux and macOS. There are two ways to use osl-dynamics on Windows.

Option 1: WSL (recommended)
---------------------------

The `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ (WSL) runs Linux inside Windows. With it you can install the full package, including GPU support for training models. To set up WSL, open PowerShell as an administrator and run:

.. code::

    wsl --install

then restart your computer. Open the Ubuntu app from the Start menu and follow the :ref:`conda installation <conda-install>` instructions in the Ubuntu terminal. If you have an NVIDIA GPU, install the latest NVIDIA driver for Windows; do not install a separate driver inside WSL. conda will install the GPU version of TensorFlow automatically.

Option 2: native Windows
------------------------

Install `Miniforge <https://conda-forge.org/download/>`_ using the Windows installer, then open the Miniforge Prompt from the Start menu and run:

.. code::

    mamba create -n osld -c conda-forge osl-dynamics-base python=3.12
    conda activate osld
    pip install tensorflow==2.19.1 tensorflow-probability==0.25.0 tf-keras==2.19.0

The :code:`osl-dynamics-base` conda package contains everything except TensorFlow, which is installed from PyPI with pip in the last step. Python is set to 3.12 because this is the newest version TensorFlow 2.19.1 supports on Windows. Note that TensorFlow does not support GPUs on native Windows, so models will be trained on the CPU. If you need a GPU, use WSL.
