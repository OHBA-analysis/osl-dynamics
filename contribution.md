# Contribution Guide

This is an introductory guide to using VRAD on the Biomedical Research Computing (BMRC) cluster. This guide covers:
- [The BMRC Cluster](#the-bmrc-cluster)
- [Installing VRAD on BMRC](#installing-vrad-on-bmrc)
- [Using VRAD on BMRC](#using-vrad-on-bmrc)
- [Editing VRAD Source Code](#editing-vrad-source-code)

## The BMRC Cluster

Website: https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/cluster-usage.

### Login
To login: `ssh <username>@rescomp1.well.ox.ac.uk` (or `@rescomp2.well.ox.ac.uk`).

To login to an interactive GPU node: `ssh <username>@compG017`.
compG017 has 2 GPUs. Can view usage with: `nvidia-smi`.

### Directories
Your home directory is `/users/woolrich/<username>`, however, this directory is limited in space. It is recommended you work from your data directory located at `/well/woolrich/users/<username>`.

### Copying files
Execute `rsync` from your local machine.

Copy files from your local machine to the cluster:
```
rsync -Phr <filename> <username>@rescomp1.well.ox.ac.uk:/path/on/bmrc
```

Copy files from the cluster to your local machine:
```
rsync -Phr <username>@rescomp1.well.ox.ac.uk:/path/to/file /path/on/local/machine
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
#$ -l gpu=2
#$ -cwd
#$ -j y
#$ -o stdout.log

# Setup your environment
module load Anaconda3
module load cudnn
source activate vrad-tf23

# Run scripts
python simulation_hmm_mvn.py
```

Submit with: `qsub submission.sh`.

Monitor jobs: `watch qstat`.

Delete all jobs: `qdel -u <username>`.

Further info: https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/gpu-resources.

## Installing VRAD on BMRC
It is recommended to install VRAD within a virtual environment. Depending on the GPU node a different version of CUDA maybe available. This means different versions of TensorFlow maybe required on different GPU nodes (older nodes may not be able to run the latest version of TensorFlow). Below are installation instructions for different TensorFlow versions.

### TensorFlow 2.5 (Recommended for compG017)
```
module use /well/woolrich/projects/software/modulefiles
module load Anaconda3
module load cuda/11.2
conda create --name vrad-tf25 python=3.8
conda activate vrad-tf25
cd VRAD
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
module load cudNN
conda create --name vrad-tf24
conda activate vrad-tf24
cd VRAD
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
conda create --name vrad-tf23
conda activate vrad-tf23
cd VRAD
pip install -e .
```

## Using VRAD on BMRC

There are two options for running VRAD:
- In a standalone script, e.g. the scripts in `/VRAD/examples`.
- In a Jupyter Notebook setup.

### Standalone Scripts

VRAD can be imported like any other python package in a script:
```
from vrad import array_ops
from vrad.models import Model
```

The script is executed via the command line on a GPU node with `python <script>.py`.

Before you can run the script you need to activate the virtual environment in which VRAD was installed with `conda activate <env>`.

### Jupyter Notebook

This is a work in progress.

## Editing VRAD Source Code

This section gives an overview of the source code and useful tips for editing.

### Overview of the VRAD Package

The main source code is contained in `/VRAD/src/vrad`. This directory contains 7 subpackages:
- `data`: Classes and functions used to read/load and manipulate data.
- `models` and `inference`: Classes for each model type and TensorFlow functions used for inference.
- `analysis`: Functions for analysing a fitted model.
- `simulation`: Classes for simulating training data.
- `utils` and `files`: Helpful utility functions and necessary files.

### Text Editors

A text editor is required for making changes to the source code. The source code must be located on the computer your running VRAD on, which in our case is the BMRC cluster. There are two options for editing the source code:
- Using an in terminal editor like vi, vim, or emacs.
- Keeping a copy of the source code on your local computer and copying it to the BMRC server. The files can be copied using the `rsync` command described above. In this case, we can use editors with a graphical interface, e.g. PyCharm.

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

Either push the new branch to remote repo:
```
git push --set-upstream origin <branch-name>
```
or merge branch into development and push:
```
git checkout development
git merge <branch-name>
git push
```
