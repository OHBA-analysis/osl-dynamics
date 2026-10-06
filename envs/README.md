# Conda Environment Files

osl-dynamics is on conda-forge, so most people do not need an environment file:

```
conda create -n osld -c conda-forge osl-dynamics
```

See the [installation instructions](https://osl-dynamics.readthedocs.io/en/latest/install.html).

The files here are for machines that need a specific set of pinned packages:

- `bmrc.yml` - the Biomedical Research Computing (BMRC) cluster at Oxford.
- `hbaws.yml` - the OHBA workstation (hbaws) at Oxford.

`fsl.yml` is used for installing osl-dynamics as an additional optional component of [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/index.html).
