:orphan:

Glasser50 Parcellation
======================

In osl-dynamics, this parcellation file is named :code:`atlas-Glasser_nparc-50_space-MNI_res-8x8x8.nii.gz`, however, this parcellation file was previously named :code:`Glasser50_binary_space-MNI152NLin6_res-8x8x8.nii.gz` (both names will work).

This is a reduced version of the :doc:`Glasser52 parcellation <glasser52>`.

Parcels
-------

.. image:: glasser50.png

Example Code
------------

Example code for plotting with this parcellation:

.. code::

    from osl_dynamics.analysis import power

    power.save(
        ...,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="atlas-Glasser_nparc-50_space-MNI_res-8x8x8.nii.gz",
        filename="map_.png",
    )
