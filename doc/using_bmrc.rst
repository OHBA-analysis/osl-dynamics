Using BMRC
==========

These notes are intended for users at OHBA. Here, we describe how to use osl-dynamics on the Biomedical Research Computing Cluster (BMRC) at the University of Oxford. Website: `https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/cluster-usage <https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/cluster-usage>`_.

Basic Usage
-----------

Login
*****

To login: ``ssh <username>@cluster1.bmrc.ox.ac.uk`` (or ``@cluster2.bmrc.ox.ac.uk``).

To login to an interactive GPU node: ``ssh compG017``.
compG017 has 2 GPUs. Can view usage with: ``nvidia-smi``.

To login with graphical output enabled use: ``ssh -X <username>@cluster1.bmrc.ox.ac.uk>``. An X11 forwarding client must be installed on your local computer, e.g. XQuartz.

Directories
***********

Your home directory is ``/users/woolrich/<username>``, however, this directory is limited in space. It is recommended you work from your data directory located at ``/well/woolrich/users/<username>``.

Copying files
*************

Execute ``rsync`` from your local machine.

Copy files from your local machine to the cluster:

.. code-block:: shell

    rsync -Phr <filename> <username>@cluster1.bmrc.ox.ac.uk:/path/on/bmrc

Copy files from the cluster to your local machine:

.. code-block:: shell

    rsync -Phr <username>@cluster1.bmrc.ox.ac.uk:/path/to/file /path/on/local/machine

Submitting Jobs
---------------

Request an interactive job on a normal node:

.. code-block:: shell

    screen
    srun -p short --pty bash

``Ctrl-A Crtl-D`` can be used to exit the screen session.

List screens with: ``screen -ls``. Reconnect to a session with: ``screen -r <id>``.

To request an interactive job on a node with a GPU:

.. code-block:: shell

    screen
    srun -p gpu_interactive --gres gpu:1 --pty bash

To submit a non-interactive GPU job, first create a ``job.sh`` file:

.. code-block:: shell

    #!/bin/bash
    #SBATCH -p gpu_short
    #SBATCH --gres gpu:1

    # Run scripts
    python dynemo_hmm_mvn.py

Note, the job will inherit the environment and working directory from when you submit the job.

Submit with: ``sbatch job.sh``.

Monitor jobs: ``watch squeue -u <username>``.

Look at a queue: ``squeue -p <queue>``, e.g. ``squeue -p gpu_short``.

Cancel a particular job: ``scancel <job-id>``.

Cancel a set of jobs: ``scancel {<start>..<end>}``.

Cancel all your jobs: ``scancel -u <username>``.

Further info:

- `https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/gpu-resources <https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/gpu-resources>`_.
- `https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/using-the-bmrc-cluster-with-slurm <https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/using-the-bmrc-cluster-with-slurm>`_

Modules
-------

The BMRC server uses software modules for popular packages. We install osl-dynamics within an Anaconda environment. We can load Anaconda with:

.. code-block:: shell

    module load Anaconda3/2022.05

We also need CUDA to use the GPUs on BMRC. We load CUDA with:

.. code-block:: shell

    module load cuDNN

Similarly we can load other useful software packages, e.g.

.. code-block:: shell

    module load git
    module load matlab/2019a

All of the above lines can be added to ``/users/woolrich/<username>/.bashrc`` which will load the modules automatically when you log in.

Using OSL Dynamics on BMRC
--------------------------

osl-dynamics can be used in a standalone python script, e.g. the scripts in ``/osl-dynamics/examples``. It is imported like any other python package:

.. code-block:: shell

    from osl_dynamics import array_ops
    from osl_dynamics.models.dynemo import Model

The script is executed via the command line on a GPU node with ``python <script>.py``.

Before you can run the script you need to activate the virtual environment in which osl-dynamics was installed with ``conda activate <env>``.

Editing OSL Dynamics on BMRC
----------------------------

A text editor is required for making changes to the source code. There are multiple options for this:

- Use an in terminal editor like vi, vim, or emacs. E.g. to use vim: ``vim <filename>``.
- Keep a copy of the source code on your local computer and copy it to the BMRC server. The files can be copied using ``rsync`` or you can setup a development environment on your local computer to sync the files automatically for you.

We recommend using VSCode locally and the ``Remote - SSH`` extension to edit remote files.

- Activate your Linux Shell Account: `https://help.it.ox.ac.uk/use-linux-service <https://help.it.ox.ac.uk/use-linux-service#collapse3091407>`_.
- Install VSCode: `https://code.visualstudio.com/ <https://code.visualstudio.com/>`_.
- Install the ``Remote - SSH`` extension: `https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh>`_.
- Click the green quick configuration button in the bottom left of VSCode.
- Click ``Open SSH Configation File...``.
- Select the first config file, e.g. for me: ``/Users/<username>/.ssh/config``.
- Paste the following into the text editor with your corresponding Oxford SSO and BMRC username:

.. code-block:: shell

    Host vscode-bmrc
        HostName cluster1.bmrc.ox.ac.uk
        ProxyJump <oxford-sso-username>@linux.ox.ac.uk
        User <bmrc-username>
        ForwardAgent yes

- Save with ``Ctrl-S``, after which the text editor can be closed.
- To connect to the server, click the green quick configuration bottom again and click ``Connect to Host...``. Then select ``vscode-bmrc``.
- You will be asked for your SSO password then BMRC password.
- If you are working on the university VPN, you can omit ``ProxyJump <oxford-sso-username>@linux.ox.ac.uk`` line.
- You can set up SSH keys for the university linux server if you want to avoid typing two passwords every time. `Guide <https://www.ssh.com/academy/ssh/copy-id>`_.

Formatting and Conventions
**************************

We use the python code formatter ``black`` to give a consistent code layout in our source files. To install:

.. code-block:: shell

    conda activate <env>
    pip install black

To format a source file:

.. code-block:: shell

    black <filename>.py

Please run ``black`` on any edited files before commiting changes.

Git Workflow
************

We use git for version control. There is one ``main`` branch. To add changes:

Create a feature branch for changes:

.. code-block:: shell

    git checkout main
    git pull
    git checkout -b <branch-name>

Make changes to file and commit it to the branch:

.. code-block:: shell

    git add <file>
    git commit -m "Short description of changes"

When writing commit messages please follow the conventions `here <https://www.conventionalcommits.org/en/v1.0.0-beta.2/#specification>`_.

Then either push the new branch to the remote repository:

.. code-block:: shell

    git push --set-upstream origin <branch-name>

and create a pull request (recommended), or merge branch into ``main`` and push:

.. code-block:: shell

    git checkout main
    git merge <branch-name>
    git push
