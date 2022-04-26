
# Contribution Guide

This is an introductory guide to using osl-dynamics on the Biomedical Research Computing (BMRC) cluster. This guide covers:
- [The BMRC Cluster](#the-bmrc-cluster)
- [Installing osl-dynamics on BMRC](#installing-osl-dynamics-on-bmrc)
- [Using osl-dynamics on BMRC](#using-osl-dynamics-on-bmrc)
- [Editing osl-dynamics Source Code](#editing-osl-dynamics-source-code)

## The BMRC Cluster

Website: https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/cluster-usage.

### Login
To login: `ssh <username>@cluster1.bmrc.ox.ac.uk` (or `@cluster2.bmrc.ox.ac.uk`).

To login to an interactive GPU node: `ssh compG017`.
compG017 has 2 GPUs. Can view usage with: `nvidia-smi`.

To login with graphical output enabled use: `ssh -X <username>@cluster1.bmrc.ox.ac.uk>`. An X11 forwarding client must be installed on your local computer, e.g. XQuartz.

### Directories
Your home directory is `/users/woolrich/<username>`, however, this directory is limited in space. It is recommended you work from your data directory located at `/well/woolrich/users/<username>`.

### Copying files
Execute `rsync` from your local machine.

Copy files from your local machine to the cluster:
```
rsync -Phr <filename> <username>@cluster1.bmrc.ox.ac.uk:/path/on/bmrc
```

Copy files from the cluster to your local machine:
```
rsync -Phr <username>@cluster1.bmrc.ox.ac.uk:/path/to/file /path/on/local/machine
```

### Submitting Jobs
Request an interactive job on a normal node:
```
screen
qlogin -q short.qc (or long.qc)
```
`Ctrl-A Crtl-D` can be used to exit the screen session.

List screens with: `screen -ls`. Reconnect to a session with: `screen -r <id>`.

To submit a non-interactive GPU job, first create a `submission.sh` file:
```
#!/bin/bash
#$ -q short.qg
#$ -l gpu=1
#$ -cwd
#$ -j y
#$ -o stdout.log

# Setup your environment
module load Anaconda3
module load cudnn
source activate osld-tf23

# Run scripts
python dynemo_hmm_mvn.py
```

Submit with: `qsub submission.sh`.

Monitor jobs: `watch qstat`.

Delete all jobs: `qdel -u <username>`.

Further info: https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/gpu-resources.

## Installing osl-dynamics on BMRC
It is recommended to install osl-dynamics within a virtual environment. Depending on the GPU node a different version of CUDA maybe available. This means different versions of TensorFlow maybe required on different GPU nodes (older nodes may not be able to run the latest version of TensorFlow). Below are installation instructions for different TensorFlow versions.

### TensorFlow 2.5 (Recommended for compG017)
```
module use /well/woolrich/projects/software/modulefiles
module load Anaconda3
module load cuda/11.2
conda create --name osld-tf25 python=3.8
conda activate osld-tf25
cd osl-dynamics
pip install -e .
```

### TensorFlow 2.4 
Update setup.cfg with dependencies:
```
tensorflow==2.4.1
tensorflow_probability==0.12.2
```

Install:
```
module load Anaconda3
module load cuDNN
conda create --name osld-tf24 python=3.8
conda activate osld-tf24
cd osl-dynamics
pip install -e .
```

### TensorFlow 2.3
Update setup.cfg with dependencies:
```
tensorflow==2.3.0
tensorflow_probability==0.11.1
```

Install:
```
module load Anaconda3
module load cudnn
conda create --name osld-tf23 python=3.8
conda activate osld-tf23
cd osl-dynamics
pip install -e .
```

## Using osl-dynamics on BMRC

osl-dynamics can be used in a standalone python script, e.g. the scripts in `/osl-dynamics/examples`. It is imported like any other python package:
```
from osl_dynamics import array_ops
from osl_dynamics.models.dynemo import Model
```

The script is executed via the command line on a GPU node with `python <script>.py`.

Before you can run the script you need to activate the virtual environment in which osl-dynamics was installed with `conda activate <env>`.

## Editing osl-dynamics Source Code

This section gives an overview of the source code and useful tips for editing.

### Overview of the osl-dynamics Package

The main source code is contained in `/osl-dynamics/src/osl_dynamics`. This directory contains 7 subpackages:
- `data`: Classes and functions used to load, save and manipulate data.
- `models` and `inference`: Classes for each model type and TensorFlow functions used for inference.
- `analysis`: Functions for analysing a fitted model.
- `simulation`: Classes for simulating training data.
- `utils` and `files`: Helpful utility functions and necessary files.

### Text Editors

A text editor is required for making changes to the source code. There are multiple options for this:
- Use an in terminal editor like vi, vim, or emacs. E.g. to use vim: `vim <filename>`.
- Use a graphical text editor such as gedit directly on the server. To do this you must login with X11 forwarding (see [The BMRC Cluster](#the-bmrc-cluster)). To launch the editor: `gedit <filename> &`.
- Keep a copy of the source code on your local computer and copy it to the BMRC server. The files can be copied using `rsync` (see [The BMRC Cluster](#the-bmrc-cluster)) or you can setup a development environment on your local computer to sync the files automatically for you.

We recommend using VSCode locally and the `Remote - SSH` extension to edit remote files.
- Activate your Linux Shell Account: [https://help.it.ox.ac.uk/use-linux-service](https://help.it.ox.ac.uk/use-linux-service#collapse3091407).
- Install VSCode: https://code.visualstudio.com/.
- Install the `Remote - SSH` extension: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh.
- Click the green quick configuration button in the bottom left of VSCode:
![configure ssh](https://microsoft.github.io/vscode-remote-release/images/remote-dev-status-bar.png)
- Click `Open SSH Configation File...`.
- Select the first config file, e.g. for me: `/Users/<username>/.ssh/config`.
- Paste the following into the text editor with your corresponding Oxford SSO and BMRC username:
```
Host vscode-bmrc
    HostName cluster1.bmrc.ox.ac.uk
    ProxyJump <oxford-sso-username>@linux.ox.ac.uk
    User <bmrc-username>
    ForwardAgent yes
```
- Save with `Ctrl-S`, after which the text editor can be closed.
- To connect to the server, click the green quick configuration bottom again and click `Connect to Host...`. Then select `vscode-bmrc`.
- You will be asked for your SSO password then BMRC password.
- If you are working on the university VPN, you can omit `ProxyJump <oxford-sso-username>@linux.ox.ac.uk` line.
- You can set up SSH keys for the university linux server if you want to avoid typing two passwords every time. [Guide](https://www.ssh.com/academy/ssh/copy-id).

### Formatting and Conventions

We use the python code formatter `black` to give a consistent code layout in our source files. To install:
```
conda activate <env>
pip install black
```
To format a source file:
```
black <filename>.py
```
Please run `black` on any edited files before commiting changes.

### Git Workflow

We use git for version control. There are two branches on the remote repo: `master` and `development`. The `development` branch is used for adding new features. We merge `development` into `master` when we have a stable version of the code. Please add new features to the `development` branch only. A standard git workflow is described below.

Create a branch for changes:
```
git checkout development
git pull
git checkout -b <branch-name>
```

Make changes to file and commit it to the branch:
```
git add <file>
git commit -m "Short description of changes"
```

When writing commit messages please follow the conventions [here](https://www.conventionalcommits.org/en/v1.0.0-beta.2/#specification).

Then either push the new branch to the remote repository:
```
git push --set-upstream origin <branch-name>
```
or merge branch into development and push:
```
git checkout development
git merge <branch-name>
git push
```
