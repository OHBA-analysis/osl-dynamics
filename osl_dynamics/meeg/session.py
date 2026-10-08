"""The files of a session in the source reconstruction pipeline.

Each step of the pipeline (coregistration, forward model, LCMV beamformer,
parcellation) reads the files written by the steps before it and writes its
own at the paths held by a :class:`Session`, so a step can be rerun on its own,
sessions can be processed in parallel and every intermediate file can be
inspected (e.g. with fsleyes).
"""

from __future__ import annotations

import json
import os


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
        self.std_brain_file = f"{self.fsl_dir}/data/standard/MNI152_T1_1mm_brain.nii.gz"
        self.std_brain_bigfov_file = (
            f"{self.fsl_dir}/data/standard/MNI152_T1_1mm_BigFoV_facemask.nii.gz"
        )

        # Transformations
        self.mni2mri_flirt_xform_file = f"{root}/mni2mri_flirt_xform.txt"
        self.mni_mri_t_file = f"{root}/mni_mri-trans.fif"

        # Registration to MNI space (affine, and nonlinear if requested)
        self.std_head_2mm_file = f"{self.fsl_dir}/data/standard/MNI152_T1_2mm.nii.gz"
        self.std_brain_mask_2mm_file = (
            f"{self.fsl_dir}/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"
        )
        self.mri_mni_affine_file = f"{root}/smri_mni_affine.nii.gz"
        self.mri2mni_warp_file = f"{root}/mri2mni_warpcoef.nii.gz"
        self.mri_mni_nonlinear_file = f"{root}/smri_mni_nonlinear.nii.gz"
        self.mni_registration_plot_file = f"{root}/mni_registration.png"
        self.mni_registration_quality_file = f"{root}/mni_registration.json"

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


class Session:
    """The files of one M/EEG session in the source reconstruction pipeline.

    The pipeline functions (in :mod:`~osl_dynamics.meeg.rhino`,
    :mod:`~osl_dynamics.meeg.source_recon` and
    :mod:`~osl_dynamics.meeg.parcellation`) take a Session and read and write
    the files held here. Nothing is created on disk until a step writes its
    output. The output layout is::

        {outdir}/{head_model_id}/bem/      surfaces in the format MNE expects
        {outdir}/{head_model_id}/coreg/    coregistration, forward model, MNI grid
        {outdir}/{id}/src/                 LCMV filters

    Parameters
    ----------
    outdir : str
        Base output directory.
    id : str
        Session identifier.
    surfaces_dir : str
        Directory with the surfaces extracted from the structural MRI by
        :func:`rhino.extract_surfaces`. Pass
        :code:`files.mni152_surfaces.directory` to use the MNI152 standard
        brain instead of a subject's MRI.
    preproc_file : str, optional
        Path to the preprocessed data file. Needed by the steps that use the
        sensor data or its info (extracting fiducials, coregistration, the
        beamformer and parcellation) unless the data is passed to them.
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
        surfaces_dir: str,
        preproc_file: str | None = None,
        pos_file: str | None = None,
        elc_file: str | None = None,
        head_model_id: str | None = None,
    ):
        self.outdir = str(outdir)
        self.id = id
        self.head_model_id = head_model_id if head_model_id is not None else id
        self._preproc_file = None if preproc_file is None else str(preproc_file)
        self.pos_file = None if pos_file is None else str(pos_file)
        self.elc_file = None if elc_file is None else str(elc_file)

        self.surfaces_dir = str(surfaces_dir)
        self.surfaces = SurfaceFilenames(self.surfaces_dir)

        self.bem_dir = f"{self.outdir}/{self.head_model_id}/bem"
        self.coreg_dir = f"{self.outdir}/{self.head_model_id}/coreg"
        self.coreg = CoregFilenames(self.coreg_dir)
        self.fwd_model_file = f"{self.coreg_dir}/model-fwd.fif"
        self.fwd_model_params_file = f"{self.coreg_dir}/model-fwd.json"
        self.bem_solution_file = f"{self.coreg_dir}/model-bem-sol.fif"
        self.mni_grid_file = f"{self.coreg_dir}/model-mni-grid.nii.gz"

        self.src_dir = f"{self.outdir}/{id}/src"
        # Assign a different path to keep several sets of filters for the
        # same session
        self.filters_file = f"{self.src_dir}/filters-lcmv.h5"

    @property
    def preproc_file(self) -> str:
        """Path to the preprocessed data file."""
        if self._preproc_file is None:
            raise ValueError(
                f"Session '{self.id}' was created without preproc_file, which "
                "this step needs. Pass preproc_file to Session, or pass the "
                "data to this function."
            )
        return self._preproc_file

    @preproc_file.setter
    def preproc_file(self, preproc_file: str | None) -> None:
        self._preproc_file = None if preproc_file is None else str(preproc_file)

    def __str__(self) -> str:
        lines = [
            f"Session {self.id}:",
            f"  Output directory:   {self.outdir}",
            f"  Preprocessed file:  {self._preproc_file}",
            f"  Surfaces directory: {self.surfaces_dir}",
            f"  BEM directory:      {self.bem_dir}",
            f"  Coreg directory:    {self.coreg_dir}",
            f"    ├─ Forward model: {self.fwd_model_file}",
            f"    ├─ BEM solution:  {self.bem_solution_file}",
            f"    └─ MNI grid:      {self.mni_grid_file}",
            f"  Source directory:   {self.src_dir}",
            f"    └─ LCMV filters:  {self.filters_file}",
        ]
        if self.pos_file is not None:
            lines.append(f"  pos file: {self.pos_file}")
        if self.elc_file is not None:
            lines.append(f"  elc file: {self.elc_file}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Session id='{self.id}' outdir='{self.outdir}'>"


# The steps of the pipeline in order: the name of the step, the file that
# step writes last, and the function that runs it
_STEPS = [
    ("surfaces", lambda s: s.surfaces.mni_mri_t_file, "rhino.extract_surfaces"),
    (
        "coregistration",
        lambda s: s.coreg.head_scaledmri_t_file,
        "rhino.coregister_head_and_mri",
    ),
    ("forward model", lambda s: s.fwd_model_file, "rhino.forward_model"),
    ("LCMV filters", lambda s: s.filters_file, "source_recon.lcmv_beamformer"),
]


def check_up_to_date(session: Session, step: str) -> None:
    """Warn if a step's output is older than the output of the step before it.

    Rerunning one step without the steps after it is an easy mistake (e.g.
    coregistering again but keeping the old forward model), so the steps that
    read a file check its date against the file it was computed from.

    Parameters
    ----------
    session : Session
        Files of the session.
    step : str
        The last step whose output is needed: 'coregistration', 'forward
        model' or 'LCMV filters'. Every step up to it is checked.
    """
    names = [name for name, _, _ in _STEPS]
    if step not in names:
        raise ValueError(f"step must be one of {names}")
    for (_, upstream, _), (name, downstream, function) in zip(
        _STEPS, _STEPS[1 : names.index(step) + 1]
    ):
        upstream_file, downstream_file = upstream(session), downstream(session)
        if not os.path.exists(upstream_file) or not os.path.exists(downstream_file):
            continue
        if os.path.getmtime(upstream_file) > os.path.getmtime(downstream_file):
            print(
                f"WARNING: the {name} ({downstream_file}) is older than the "
                f"files it was computed from ({upstream_file}). Rerun "
                f"{function} (and the steps after it)."
            )


def save_params(filename: str, params: dict) -> None:
    """Save the parameters a step was run with next to its output.

    Parameters
    ----------
    filename : str
        JSON file to write.
    params : dict
        Parameters. Values must be JSON serialisable.
    """
    with open(filename, "w") as file:
        json.dump(params, file, indent=4)


def load_params(filename: str) -> dict | None:
    """Load the parameters a step was run with.

    Parameters
    ----------
    filename : str
        JSON file written by :func:`save_params`.

    Returns
    -------
    params : dict
        Parameters, or None if the file does not exist.
    """
    if not os.path.exists(filename):
        return None
    with open(filename) as file:
        return json.load(file)
