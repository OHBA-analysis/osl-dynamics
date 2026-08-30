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
    elc_file : str, optional
        Path to an .elc file (alternative format for head shape points
        from CTF data).
    head_model_id : str, optional
        Identifier owning the head model: the coregistration, BEM and forward
        model. Defaults to :code:`id`. Pass a subject when the head model is
        shared by several sessions, e.g. a template montage that gives every
        session of a subject the same head shape points, so that the
        coregistration and forward model are computed once and every session
        reads them.
    """

    def __init__(
        self,
        outdir: str,
        id: str,
        preproc_file: str,
        surfaces_dir: str,
        pos_file: Optional[str] = None,
        elc_file: Optional[str] = None,
        head_model_id: Optional[str] = None,
    ):
        self.outdir = outdir
        self.id = id
        self.head_model_id = head_model_id if head_model_id is not None else id

        self.preproc_file = preproc_file

        self.surfaces_dir = surfaces_dir
        self.surfaces = SurfaceFilenames(surfaces_dir)

        self._bem_dir = f"{outdir}/{self.head_model_id}/bem"
        self._coreg_dir = f"{outdir}/{self.head_model_id}/coreg"
        self._coreg = CoregFilenames(self._coreg_dir)
        self._src_dir = f"{outdir}/{id}/src"

        self.pos_file = pos_file
        self.elc_file = elc_file

    @property
    def bem_dir(self) -> str:
        """BEM directory, created on first use."""
        os.makedirs(self._bem_dir, exist_ok=True)
        return self._bem_dir

    @property
    def coreg_dir(self) -> str:
        """Coregistration directory, created on first use."""
        os.makedirs(self._coreg_dir, exist_ok=True)
        return self._coreg_dir

    @property
    def coreg(self) -> CoregFilenames:
        """Coregistration file paths. Creates the coregistration directory."""
        os.makedirs(self._coreg_dir, exist_ok=True)
        return self._coreg

    @property
    def fwd_model(self) -> str:
        """Forward model file. Creates the coregistration directory."""
        return f"{self.coreg_dir}/model-fwd.fif"

    @property
    def src_dir(self) -> str:
        """Source reconstruction directory, created on first use."""
        os.makedirs(self._src_dir, exist_ok=True)
        return self._src_dir

    @property
    def filters(self) -> str:
        """LCMV filters file. Creates the source directory."""
        return f"{self.src_dir}/filters-lcmv.h5"

    def __str__(self) -> str:
        lines = [
            f"OSLFilenames for {self.id}:",
            f"  Output directory:  {self.outdir}",
            f"  Preprocessed file: {self.preproc_file}",
            f"  Surfaces directory: {self.surfaces_dir}",
            f"  BEM directory:     {self._bem_dir}",
            f"  Coreg directory:   {self._coreg_dir}",
            f"    \u2514\u2500 Forward model: {self._coreg_dir}/model-fwd.fif",
            f"  Source directory:  {self._src_dir}",
            f"    \u2514\u2500 lcmv filters:  {self._src_dir}/filters-lcmv.h5",
        ]
        if self.pos_file is not None:
            lines += [
                f"  pos file:  {self.pos_file}",
            ]
        if self.elc_file is not None:
            lines += [
                f"  elc file:  {self.elc_file}",
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<OSLFilenames id='{self.id}' outdir='{self.outdir}'>"
