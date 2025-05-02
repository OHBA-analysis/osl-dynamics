# Conda Environment Files

## Latest version

The latest conda environment files are:

- `osld.yml` - install the the latest osl-dynamics.
- `osld-tf.yml` - install the latest osl-dynamics and TensorFlow 2.19.
- `osld-tf-cuda.yml` - install the latest osl-dynamics and TensorFlow 2.19 with CUDA.

Older Macs with an Intel chip:

- `osld-tf-macos.yml` - install osl-dynamics and tensorflow-macos 2.16.1.

Oxford-specific computers:

- `bmrc.yml`
- `hbaws.yml`

These will install osl-dynamics v2.1+.

## Old versions

- `osld-2.0.yml`
- `osld-2.0-bmrc.yml`
- `osld-2.0-hbaws.yml`

Note, these do not include osl-dynamics. You need to install osl-dynamics with:
```
conda activate osld-2.0
pip install osl-dynamics==2.0.X
````
Any of the versions (2.0.0, 2.0.1, 2.0.2) can be installed.

If osl-dynamics 2.0.X is installed on the BMRC cluster, then use the following CUDA/cuDNN module:
```
module load cuDNN/8.4.1.50-CUDA-11.7.0
```
