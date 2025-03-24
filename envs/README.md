# Conda Environment Files

The latest conda envs are:

- `osld.yml` - install the the latest osl-dynamics.
- `osld-tf.yml` - install the latest osl-dynamics and TensorFlow 2.19.
- `osld-tf-cuda.yml` - install the latest osl-dynamics and TensorFlow 2.19 with CUDA.

Oxford-specific computers:

- `bmrc.yml`
- `hbaws.yml`

Note, these do not include osl-dynamics. You need to install osl-dynamics with
```
git clone https://github.com/OHBA-analysis/osl-dynamics.git
cd osl-dynamics
conda activate osld
pip install -e .
````

Old installations:

- `osld-2.0.yml`
- `osld-2.0-bmrc.yml`
- `osld-2.0-hbaws.yml`

Note, these do not include osl-dynamics. You need to install osl-dynamics with
```
conda activate osld-2.0
pip install osl-dynamics==2.0.X
````
Any of the versions (2.0.0, 2.0.1, 2.0.2) can be installed. For v2.1+ use the newer environments.

If osl-dynamics 2.0.X is installed on the BMRC cluster, then use the following CUDA/cuDNN module:
```
module load cuDNN/8.4.1.50-CUDA-11.7.0
```
