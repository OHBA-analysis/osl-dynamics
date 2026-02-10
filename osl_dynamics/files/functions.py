"""Functions used to manage files kept within osl-dynamics."""

from os import path


def check_exists(filename, directory):
    """Looks for a file in the current working directory and in osl-dynamics.

    Parameters
    ----------
    filename : str
        Name of file to look for or a path to a file.
    directory : str
        Path to directory to look in.

    Returns
    -------
    filename : str
        Full path to the file if found.

    Raises
    ------
    FileNotFoundError
        If the file could not be found.
    """
    if not path.exists(filename):

        # Mapping from old parcellation filenames to new ones
        old_filenames = {
            "Glasser50_space-MNI152NLin6_res-8x8x8.nii.gz": "atlas-Glasser_nparc-50_space-MNI_res-8x8x8.nii.gz",
            "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz": "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz",
            "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz": "atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
            "fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz": "atlas-Giles_nparc-39_space-MNI_res-8x8x8.nii.gz",
            "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz": "atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz",
            "dk_cortical.nii.gz": "atlas-DK_nparc-68_space-MNI_res-8x8x8.nii.gz",
            "aal_cortical_merged_8mm_stacked.nii.gz": "atlas-AAL_nparc-78_space-MNI_res-8x8x8.nii.gz",
        }

        if filename in old_filenames:
            filename = old_filenames[filename]

        if path.exists(f"{directory}/{filename}"):
            filename = f"{directory}/{filename}"
        else:
            raise FileNotFoundError(filename)

    return filename
