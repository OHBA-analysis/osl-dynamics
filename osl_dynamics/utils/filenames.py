"""Filename containers for M/EEG processing pipelines."""

import os
from typing import Optional


class SurfaceFilenames:
    """Container for surface extraction file paths.

    Parameters
    ----------
    root : str
        Root directory for surface files.
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.fsl_dir = os.environ["FSLDIR"]

        # Nifti files
        self.mri_file = f"{root}/smri.nii.gz"
        self.std_brain = f"{self.fsl_dir}/data/standard/MNI152_T1_1mm_brain.nii.gz"
        self.std_brain_bigfov = (
            f"{self.fsl_dir}/data/standard/MNI152_T1_1mm_BigFoV_facemask.nii.gz"
        )

        # Transformations
        self.mni2mri_flirt_xform_file = f"{root}/mni2mri_flirt_xform.txt"
        self.mni_mri_t_file = f"{root}/mni_mri-trans.fif"

        # BET mesh / surfaces
        self.bet_outskin_mesh_vtk_file = f"{root}/outskin_mesh.vtk"
        self.bet_inskull_mesh_vtk_file = f"{root}/inskull_mesh.vtk"
        self.bet_outskull_mesh_vtk_file = f"{root}/outskull_mesh.vtk"
        self.bet_outskin_mesh_file = f"{root}/outskin_mesh.nii.gz"
        self.bet_outskin_plus_nose_mesh_file = f"{root}/outskin_plus_nose_mesh.nii.gz"
        self.bet_inskull_mesh_file = f"{root}/inskull_mesh.nii.gz"
        self.bet_outskull_mesh_file = f"{root}/outskull_mesh.nii.gz"


class CoregFilenames:
    """Container for coregistration file paths.

    Parameters
    ----------
    root : str
        Root directory for coregistration files.
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

        # Nifti files
        self.mri_file = f"{root}/scaled_mri.nii.gz"

        # Fif files
        self.info_fif_file = f"{root}/info-raw.fif"
        self.head_scaledmri_t_file = f"{root}/head_scaledmri-trans.fif"
        self.head_mri_t_file = f"{root}/head_mri-trans.fif"
        self.ctf_head_mri_t_file = f"{root}/ctf_head_mri-trans.fif"
        self.mrivoxel_scaledmri_t_file = f"{root}/mrivoxel_scaledmri_t_file-trans.fif"

        # Fiducials / headshape points
        self.mri_nasion_file = f"{root}/mri_nasion.txt"
        self.mri_rpa_file = f"{root}/mri_rpa.txt"
        self.mri_lpa_file = f"{root}/mri_lpa.txt"
        self.head_nasion_file = f"{root}/head_nasion.txt"
        self.head_rpa_file = f"{root}/head_rpa.txt"
        self.head_lpa_file = f"{root}/head_lpa.txt"
        self.head_headshape_file = f"{root}/head_headshape.txt"

        # Freesurfer mesh in native space
        self.bet_outskin_surf_file = f"{root}/scaled_outskin.surf"
        self.bet_outskin_plus_nose_surf_file = f"{root}/scaled_outskin_plus_nose.surf"
        self.bet_inskull_surf_file = f"{root}/scaled_inskull.surf"
        self.bet_outskull_surf_file = f"{root}/scaled_outskull.surf"

        # BET mesh / surfaces in native space
        self.bet_outskin_mesh_vtk_file = f"{root}/scaled_outskin_mesh.vtk"
        self.bet_inskull_mesh_vtk_file = f"{root}/scaled_inskull_mesh.vtk"
        self.bet_outskull_mesh_vtk_file = f"{root}/scaled_outskull_mesh.vtk"
        self.bet_outskin_mesh_file = f"{root}/scaled_outskin_mesh.nii.gz"
        self.bet_outskin_plus_nose_mesh_file = (
            f"{root}/scaled_outskin_plus_nose_mesh.nii.gz"
        )
        self.bet_inskull_mesh_file = f"{root}/scaled_inskull_mesh.nii.gz"
        self.bet_outskull_mesh_file = f"{root}/scaled_outskull_mesh.nii.gz"


class OSLFilenames:
    """Container for all pipeline file paths for processing a single M/EEG session.

    Parameters
    ----------
    outdir : str
        Base output directory.
    id : str
        Session identifier.
    preproc_file : str
        Path to the preprocessed data file.
    surfaces_dir : str
        Path to the surfaces directory.
    pos_file : str, optional
        Path to a .pos file (only needed for CTF data).
    """

    def __init__(
        self,
        outdir: str,
        id: str,
        preproc_file: str,
        surfaces_dir: str,
        pos_file: Optional[str] = None,
    ):
        self.outdir = outdir
        self.id = id

        self.preproc_file = preproc_file

        self.surfaces_dir = surfaces_dir
        self.surfaces = SurfaceFilenames(surfaces_dir)

        self.bem_dir = f"{outdir}/{id}/bem"
        os.makedirs(self.bem_dir, exist_ok=True)

        self.coreg_dir = f"{outdir}/{id}/coreg"
        self.coreg = CoregFilenames(self.coreg_dir)
        self.fwd_model = f"{self.coreg_dir}/model-fwd.fif"
        self.pos_file = pos_file  # only needed for CTF data

        self.src_dir = f"{outdir}/{id}/src"
        os.makedirs(self.src_dir, exist_ok=True)
        self.filters = f"{self.src_dir}/filters-lcmv.h5"

    def __str__(self) -> str:
        lines = [
            f"OSLFilenames for {self.id}:",
            f"  Output directory:  {self.outdir}",
            f"  Preprocessed file: {self.preproc_file}",
            f"  Surfaces directory: {self.surfaces_dir}",
            f"  BEM directory:     {self.bem_dir}",
            f"  Coreg directory:   {self.coreg_dir}",
            f"    \u2514\u2500 Forward model: {self.fwd_model}",
            f"  Source directory:  {self.src_dir}",
            f"    \u2514\u2500 lcmv filters:  {self.filters}",
        ]
        if self.pos_file is not None:
            lines += [
                f"  pos file:  {self.pos_file}",
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<OSLFilenames id='{self.id}' outdir='{self.outdir}'>"
