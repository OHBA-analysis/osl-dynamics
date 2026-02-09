:orphan:

Glasser52 Parcellation
======================

In osl-dynamics, this parcellation file is named :code:`atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz`, however, this parcellation file was previously named :code:`Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz` (both names will work).

Reference
---------

If you use the this parcellation, please cite:

    Kohl, O., Woolrich, M., Nobre, A. C., & Quinn, A. (2023). Glasser52: A parcellation for MEG-Analysis [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.10401793

See the link for further info regarding the parcel name/locations.

Parcels
-------

.. image:: glasser52.png

Example Code
------------

Example code for plotting with this parcellation:

.. code::

    from osl_dynamics.analysis import power

    power.save(
        ...,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz",
        filename="map_.png",
    )
