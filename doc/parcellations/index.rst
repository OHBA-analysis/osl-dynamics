:orphan:

Parcellations
=============

Parcellations are stored on the GitHub repository `here <https://github.com/OHBA-analysis/osl-dynamics/tree/main/osl_dynamics/files/parcellation>`_.

Available Parcellations
-----------------------

Desikan-Killiany parcellations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------------------------------------------+
| :doc:`atlas-DK_nparc-54_space-MNI_res-8x8x8.nii.gz <dk54>` |
+------------------------------------------------------------+
| :doc:`atlas-DK_nparc-68_space-MNI_res-8x8x8.nii.gz <dk68>` |
+------------------------------------------------------------+

Glasser parcellations
^^^^^^^^^^^^^^^^^^^^^

+----------------------------------------------------------------------+
| :doc:`atlas-Glasser_nparc-50_space-MNI_res-8x8x8.nii.gz <glasser50>` |
+----------------------------------------------------------------------+
| :doc:`atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz <glasser52>` |
+----------------------------------------------------------------------+

Giles parcellations
^^^^^^^^^^^^^^^^^^^

+------------------------------------------------------------------+
| :doc:`atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz <giles38>` |
+------------------------------------------------------------------+
| :doc:`atlas-Giles_nparc-39_space-MNI_res-8x8x8.nii.gz <giles39>` |
+------------------------------------------------------------------+
| :doc:`atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz <giles42>` |
+------------------------------------------------------------------+

AAL parcellation
^^^^^^^^^^^^^^^^^

+--------------------------------------------------------------+
| :doc:`atlas-AAL_nparc-78_space-MNI_res-8x8x8.nii.gz <aal78>` |
+--------------------------------------------------------------+

Old Naming
----------

Note, the parcellation files in osl-dynamics have been renamed:

+---------------------------------------------------+--------------------------------------------------------------------------+
| New name                                          | Old name                                                                 |
+===================================================+==========================================================================+
| atlas-Glasser_nparc-50_space-MNI_res-8x8x8.nii.gz | Glasser50_space-MNI152NLin6_res-8x8x8.nii.gz                             |
+---------------------------------------------------+--------------------------------------------------------------------------+
| atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz | Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz                      |
+---------------------------------------------------+--------------------------------------------------------------------------+
| atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz   | fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz           |
+---------------------------------------------------+--------------------------------------------------------------------------+
| atlas-Giles_nparc-39_space-MNI_res-8x8x8.nii.gz   | fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz               |
+---------------------------------------------------+--------------------------------------------------------------------------+
| atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz   | fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz  |
+---------------------------------------------------+--------------------------------------------------------------------------+
| atlas-DK_nparc-68_space-MNI_res-8x8x8.nii.gz      | dk_cortical.nii.gz                                                       |
+---------------------------------------------------+--------------------------------------------------------------------------+
| atlas-AAL_nparc-78_space-MNI_res-8x8x8.nii.gz     | aal_cortical_merged_8mm_stacked.nii.gz                                   |
+---------------------------------------------------+--------------------------------------------------------------------------+

MNI Coordinates
---------------

The parcellations provided are in MNI space.

Example code for obtaining the MNI coordinates of each parcel center:

.. code::

    from osl_dynamics.utils.parcellation import Parcellation

    filename 'atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz'
    parc = Parcellation(filename)
    mni_coords = parc.roi_centers()
