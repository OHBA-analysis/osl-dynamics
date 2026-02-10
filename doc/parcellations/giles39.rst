:orphan:

Giles39 Parcellation
====================

In osl-dynamics, this parcellation file is named :code:`atlas-Giles_nparc-39_space-MNI_res-8x8x8.nii.gz`, however, this parcellation file was previously named :code:`fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz` (both names will work).

This is a modified version of the :doc:`original Giles parcellation <giles38>` to include the PCC.

This parcellation was used in `Quinn et al. (2018) <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00603/full>`_.

Parcels
-------

.. image:: giles39.png

Example Code
------------

Example code for plotting with this parcellation:

.. code::

    from osl_dynamics.analysis import power

    power.save(
        ...,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="atlas-Giles_nparc-39_space-MNI_res-8x8x8.nii.gz",
        filename="map_.png",
    )
