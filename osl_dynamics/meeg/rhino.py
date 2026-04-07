"""RHINO functions."""

import os
import copy
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import nibabel as nib
import nilearn as nil
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage import measure
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr, CPointer
from scipy import ndimage, LowLevelCallable
from scipy.spatial import KDTree

from fsl import wrappers as fsl_wrappers

import mne
from mne.transforms import (
    Transform,
    read_trans,
    write_trans,
    invert_transform,
    combine_transforms,
    apply_trans,
    rotation,
    _get_trans,
)
from mne.io.constants import FIFF
from mne.viz.backends.renderer import _get_renderer

from osl_dynamics.utils.filenames import OSLFilenames, SurfaceFilenames
from osl_dynamics.utils.misc import system_call


def scale_surfaces_to_headshape(
    preproc_file: str,
    surfaces_dir: str,
    outdir: str,
    n_init: int = 10,
) -> str:
    """Scale surfaces to match headshape points.

    Computes an affine transform that scales an existing set of surfaces
    (from ``extract_surfaces`` or the bundled MNI152 surfaces) to match
    the headshape digitisation points in a FIF file. The scaled MRI and
    surfaces are written to ``outdir``.

    This is useful when no individual structural MRI is available (e.g.,
    OPM recordings with EinScan headshape).

    The method:

    1. Aligns the headshape to the template scalp surface using fiducials
       and ICP.
    2. Computes an affine transform (with anisotropic scaling) from
       nearest-neighbour correspondences between the aligned surfaces.
    3. Applies the transform to the MRI and surfaces.

    Parameters
    ----------
    preproc_file : str
        Path to preprocessed FIF file containing headshape points in
        ``info['dig']``.
    surfaces_dir : str
        Path to a directory containing extracted surfaces (from
        ``extract_surfaces``). For the default MNI152 brain, use
        ``osl_dynamics.files.mni152_surfaces.directory``.
    outdir : str
        Output directory for the scaled MRI and surfaces. Pass this
        as ``surfaces_dir`` when creating ``OSLFilenames``.
    n_init : int, optional
        Number of random initialisations for ICP alignment.

    Returns
    -------
    outdir : str
        Path to the output directory containing the scaled MRI and
        surfaces.
    """
    outdir = str(Path(outdir).resolve())
    os.makedirs(outdir, exist_ok=True)

    print()
    print("Scaling surfaces to headshape")
    print("-----------------------------")

    # -------------
    # Load surfaces
    # -------------
    sfns = SurfaceFilenames(surfaces_dir)

    outskin_sform = _get_sform(sfns.bet_outskin_mesh_file)["trans"]
    scalp_voxels = _niimask2indexpointcloud(sfns.bet_outskin_mesh_file)
    mni2mri_mm = read_trans(sfns.mni_mri_t_file)["trans"]

    # Subsample scalp surface for ICP efficiency
    n_scalp = scalp_voxels.shape[1]
    if n_scalp > 5000:
        indices = np.random.choice(n_scalp, 5000, replace=False)
        scalp_voxels = scalp_voxels[:, indices]
    scalp_mm = _xform_points(outskin_sform, scalp_voxels)

    # -------------------------------
    # Extract headshape from FIF file
    # -------------------------------
    print(f"Loading headshape from {preproc_file}")
    headshape_mm, nasion_mm, rpa_mm, lpa_mm = _extract_headshape(preproc_file)
    print(f"Found {headshape_mm.shape[1]} headshape points")

    # ---------------------------------
    # Initial alignment using fiducials
    # ---------------------------------
    print("Computing initial alignment via fiducials")

    mni_nasion = np.array([1.0, 85.0, -41.0])
    mni_rpa = np.array([83.0, -20.0, -65.0])
    mni_lpa = np.array([-83.0, -20.0, -65.0])

    mri_nasion = _xform_points(mni2mri_mm, mni_nasion)
    mri_rpa = _xform_points(mni2mri_mm, mni_rpa)
    mri_lpa = _xform_points(mni2mri_mm, mni_lpa)

    polhemus_fid = np.column_stack(
        [
            nasion_mm.reshape(-1, 1),
            rpa_mm.reshape(-1, 1),
            lpa_mm.reshape(-1, 1),
        ]
    )
    mri_fid = np.column_stack(
        [
            mri_nasion.reshape(-1, 1),
            mri_rpa.reshape(-1, 1),
            mri_lpa.reshape(-1, 1),
        ]
    )
    head2mri_xform, _ = _rigid_transform_3D(mri_fid, polhemus_fid)

    # --------------
    # ICP refinement
    # --------------
    print(f"Running ICP with {n_init} initialisations")

    headshape_mri_mm = _xform_points(head2mri_xform, headshape_mm)
    xform_icp, _, _ = _rhino_icp(scalp_mm, headshape_mri_mm, n_init=n_init)
    head2mri_refined = xform_icp @ head2mri_xform

    # -----------------------------------------
    # Compute affine from point correspondences
    # -----------------------------------------
    print("Computing affine transform from surface correspondences")
    headshape_aligned = _xform_points(head2mri_refined, headshape_mm)

    scalp_tree = KDTree(scalp_mm.T)
    _, nn_indices = scalp_tree.query(headshape_aligned.T)
    template_correspondences = scalp_mm[:, nn_indices]

    n_pts = headshape_aligned.shape[1]
    src = np.vstack([template_correspondences, np.ones(n_pts)])
    dst = headshape_aligned

    affine_warp, _, _, _ = np.linalg.lstsq(src.T, dst.T, rcond=None)
    affine_warp = affine_warp.T
    warp_xform = np.eye(4)
    warp_xform[:3, :] = affine_warp

    print(
        f"  Scaling: x={warp_xform[0,0]:.3f} "
        f"y={warp_xform[1,1]:.3f} z={warp_xform[2,2]:.3f}"
    )

    # ---------------
    # Save scaled MRI
    # ---------------
    print("Saving scaled MRI and surfaces")

    mri_img = nib.load(sfns.mri_file)
    mri_sform = mri_img.header.get_sform()
    sform_code = int(mri_img.header["sform_code"])
    scaled_sform = warp_xform @ mri_sform

    smri_out = os.path.join(outdir, "smri.nii.gz")
    smri_img = nib.Nifti1Image(mri_img.get_fdata(), scaled_sform)
    smri_img.header.set_sform(scaled_sform, code=sform_code)
    nib.save(smri_img, smri_out)

    # --------------------
    # Save scaled surfaces
    # --------------------
    for mesh_name in ["outskin_mesh", "inskull_mesh", "outskull_mesh"]:
        # Scale NIfTI mesh sform
        src_nii = os.path.join(surfaces_dir, f"{mesh_name}.nii.gz")
        dst_nii = os.path.join(outdir, f"{mesh_name}.nii.gz")
        mesh_img = nib.load(src_nii)
        mesh_sform = warp_xform @ mesh_img.header.get_sform()
        new_img = nib.Nifti1Image(mesh_img.get_fdata(), mesh_sform)
        new_img.header.set_sform(mesh_sform, code=sform_code)
        nib.save(new_img, dst_nii)

        # Scale VTK mesh vertex coordinates
        src_vtk = os.path.join(surfaces_dir, f"{mesh_name}.vtk")
        dst_vtk = os.path.join(outdir, f"{mesh_name}.vtk")
        if os.path.exists(src_vtk):
            _transform_vtk_mesh(src_vtk, src_nii, dst_vtk, dst_nii, warp_xform)

    # Save scaled MNI -> MRI transform
    mni_mri_t = warp_xform @ mni2mri_mm
    mni_mri_t_out = Transform("mni_tal", "mri", mni_mri_t)
    write_trans(
        os.path.join(outdir, "mni_mri-trans.fif"),
        mni_mri_t_out,
        overwrite=True,
    )

    print(f"Surfaces saved: {outdir}")
    return outdir


def extract_surfaces(
    mri_file: str,
    outdir: str,
    include_nose: bool = True,
    do_mri2mniaxes_xform: bool = True,
    bet_fval: float = None,
    show: bool = False,
) -> None:
    """Extract surfaces.

    Extracts inner skull, outer skin (scalp) and brain surfaces from passed
    in mri_file, which is assumed to be a T1, using FSL. Assumes that the
    MRI file has a valid sform.

    In more detail:
    1) Transform MRI to be aligned with the MNI axes so that BET works well
    2) Use bet to skull strip MRI so that flirt works well
    3) Use flirt to register skull stripped MRI to MNI space
    4) Use BET/BETSURF to get:
       a) The scalp surface (excluding nose), this gives the MRI-derived
          headshape points in native MRI space, which can be used in the
          headshape points registration later.
       b) The scalp surface (outer skin), inner skull and brain surface, these
          can be used for forward modelling later. Note that  due to the unusual
          naming conventions used by BET:
          - bet_inskull_mesh_file is actually the brain surface
          - bet_outskull_mesh_file is actually the inner skull surface
          - bet_outskin_mesh_file is the outer skin/scalp surface
    5) Add nose to scalp surface (optional)
    6) Output the transform from MRI space to MNI
    7) Output surfaces in MRI space

    Parameters
    ----------
    mri_file : str
        Full path to structural MRI in NIfTI format (with .nii.gz, .nii,
        .hdr or .mgz extension).
        The sform code must be 1 (Scanner Anat) or 4 (MNI), and the sform
        matrix should transform from voxel indices to coordinates in mm. The
        sform defines the native/MRI coordinate system used throughout RHINO.
        The qform is ignored. You can check the sform code with
        ``fslorient -getsformcode <mri_file>`` and set it with
        ``fslorient -setsformcode 1 <mri_file>``.
    outdir : str
        Output directory.
    include_nose : bool, optional
        Specifies whether to add the nose to the outer skin (scalp) surface.
        This can help RHINO's coregistration to work better, assuming that there
        are headshape points that also include the nose.
        Requires the structural MRI to have a FOV that includes the nose!
    do_mri2mniaxes_xform : bool, optional
        Specifies whether to do step (1) above, i.e. transform MRI to be
        aligned with the MNI axes. Sometimes needed when the MRI goes out
        of the MNI FOV after step (1).
    bet_fval : float, optional
        Fractional intensity threshold for FSL's BET. Default is None, which
        uses BET's default (0.5). Higher values (e.g. 0.6-0.7) give more
        aggressive skull stripping, which can help when the inner skull
        surface includes non-brain tissue.
    show : bool, optional
        Whether to display the surface plots interactively. Default is
        False (suitable for batch processing).
    """

    # Note the jargon used varies for xforms and coord spaces:
    # - MEG (device): dev_head_t --> HEAD (polhemus)
    # - HEAD (polhemus): head_mri_t (polhemus2native) --> MRI (native)
    # - MRI (native): mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    # - MRI (native): sform (mri2mniaxes) --> MNI axes

    # RHINO does everything in mm

    print()
    print("Extracting surfaces")
    print("-------------------")

    fns = SurfaceFilenames(outdir)

    # Check mri_file
    mri_ext = "".join(Path(mri_file).suffixes)
    if mri_ext not in [".nii", ".nii.gz", ".hdr", ".mgz"]:
        raise ValueError(
            "mri_file needs to have a .nii, .nii.gz, .hdr or .mgz extension"
        )

    # Copy MRI to new file for modification
    img = nib.load(mri_file)
    nib.save(img, fns.mri_file)

    # We will always use the sform, and so we will set the qform to be same
    # to stop the original qform from being used by mistake (e.g. by flirt)
    #
    # Command: fslorient -copysform2qform <mri_file>
    fsl_wrappers.misc.fslorient(fns.mri_file, copysform2qform=True)

    # Check orientation of the MRI
    mri_orient = _get_orient(fns.mri_file)

    if mri_orient != "RADIOLOGICAL" and mri_orient != "NEUROLOGICAL":
        raise ValueError(
            "Cannot determine orientation of brain, please check output of:\n "
            f"fslorient -getorient {fns.mri_file}"
        )

    # If orientation is not RADIOLOGICAL then force it to be RADIOLOGICAL
    if mri_orient != "RADIOLOGICAL":
        print("Reorienting brain to be RADIOLOGICAL")

        # Command: fslorient -forceradiological <mri_file>
        fsl_wrappers.misc.fslorient(fns.mri_file, forceradiological=True)

    print(
        "You can use the following command line call to check the MRI is "
        "appropriate, including checking that the L-R, S-I, A-P labels are "
        "sensible:"
    )
    print(f"fsleyes {fns.mri_file} {fns.std_brain}")

    # ------------------------------------------------------------------------
    # 1) Transform MRI to be aligned with the MNI axes so that BET works well
    # ------------------------------------------------------------------------

    img = nib.load(fns.mri_file)
    img_density = np.sum(img.get_fdata()) / np.prod(img.get_fdata().shape)

    # We will start by transforming MRI so that its voxel indices axes are
    # aligned to MNI's. This helps BET work.

    # Calculate mri2mniaxes
    if do_mri2mniaxes_xform:
        flirt_mri2mniaxes_xform = _get_flirt_xform_between_axes(
            fns.mri_file, fns.std_brain
        )
    else:
        flirt_mri2mniaxes_xform = np.eye(4)

    # Write xform to disk so flirt can use it
    flirt_mri2mniaxes_xform_file = f"{fns.root}/flirt_mri2mniaxes_xform.txt"
    np.savetxt(flirt_mri2mniaxes_xform_file, flirt_mri2mniaxes_xform)

    # Apply mri2mniaxes xform to mri to get mri_mniaxes, which means MRIs
    # voxel indices axes are aligned to be the same as MNI's
    # Command: flirt -in <mri_file> -ref <std_brain> -applyxfm \
    #          -init <mri2mniaxes_xform_file> -out <mri_mni_axes_file>
    flirt_mri_mniaxes_file = f"{fns.root}/flirt_mri_mniaxes.nii.gz"
    fsl_wrappers.flirt(
        fns.mri_file,
        fns.std_brain,
        applyxfm=True,
        init=flirt_mri2mniaxes_xform_file,
        out=flirt_mri_mniaxes_file,
    )

    img = nib.load(flirt_mri_mniaxes_file)
    img_latest_density = np.sum(img.get_fdata()) / np.prod(img.get_fdata().shape)

    if 5 * img_latest_density < img_density:
        raise Exception(
            "Something is wrong with the passed in structural MRI: "
            f"{fns.mri_file}\n"
            "Either it is empty or the sformcode is incorrectly set.\n\n"
            "Try running the following from a command line:\n"
            f"fsleyes {fns.std_brain} {fns.mri_file}\n\n"
            "And see if the standard space brain is shown in the same postcode "
            "as the structural.\n"
            "If it is not, then the sformcode in the structural image needs "
            "setting correctly.\n"
            "Or try passing do_mri2mniaxes_xform=True."
        )

    # ------------------------------------------------------
    # 2) Use BET to skull strip MRI so that FLIRT works well
    # ------------------------------------------------------

    # Check MRI doesn't contain nans
    # (this can cause segmentation faults with FSL's bet)
    if _check_nii_for_nan(fns.mri_file):
        print("WARNING: nan found in MRI file.")

    print("Running BET pre-FLIRT...")

    # Command: bet <flirt_mri_mniaxes_file> <flirt_mri_mniaxes_bet_file>
    flirt_mri_mniaxes_bet_file = f"{fns.root}/flirt_mri_mniaxes_bet"
    bet_kwargs = {}
    if bet_fval is not None:
        bet_kwargs["fracintensity"] = bet_fval
    fsl_wrappers.bet(flirt_mri_mniaxes_file, flirt_mri_mniaxes_bet_file, **bet_kwargs)

    # ---------------------------------------------------------
    # 3) Use FLIRT to register skull stripped MRI to MNI space
    # ---------------------------------------------------------

    print("Running FLIRT...")

    # Flirt is run on the skull stripped brains to register the mri_mniaxes
    # to the MNI standard brain
    #
    # Command: flirt -in <flirt_mri_mniaxes_bet_file> -ref <std_brain> \
    #          -omat <flirt_mniaxes2mni_file> -o <flirt_mri_mni_bet_file>
    flirt_mniaxes2mni_file = f"{fns.root}/flirt_mniaxes2mni.txt"
    flirt_mri_mni_bet_file = f"{fns.root}/flirt_mri_mni_bet.nii.gz"
    fsl_wrappers.flirt(
        flirt_mri_mniaxes_bet_file,
        fns.std_brain,
        omat=flirt_mniaxes2mni_file,
        o=flirt_mri_mni_bet_file,
    )

    # Calculate overall flirt transform, from MRI to MNI
    #
    # Command: convert_xfm -omat <flirt_mri2mni_xform_file> \
    #          -concat <flirt_mniaxes2mni_file> <flirt_mri2mniaxes_xform_file>
    flirt_mri2mni_xform_file = f"{fns.root}/flirt_mri2mni_xform.txt"
    fsl_wrappers.concatxfm(
        flirt_mri2mniaxes_xform_file,
        flirt_mniaxes2mni_file,
        flirt_mri2mni_xform_file,
    )  # Note, the wrapper reverses the order of arguments

    # and also calculate its inverse, from MNI to MRI
    #
    # Command: convert_xfm -omat <mni2mri_flirt_xform_file> \
    #          -inverse <flirt_mri2mni_xform_file>
    mni2mri_flirt_xform_file = fns.mni2mri_flirt_xform_file
    fsl_wrappers.invxfm(
        flirt_mri2mni_xform_file, mni2mri_flirt_xform_file
    )  # Note, the wrapper reverses the order of arguments

    # Move full MRI into MNI space to do full bet and betsurf
    #
    # Command: flirt -in <mri_file> -ref <std_brain> -applyxfm \
    #          -init <flirt_mri2mni_xform_file> -out <flirt_mri_mni_file>
    flirt_mri_mni_file = f"{fns.root}/flirt_mri_mni.nii.gz"
    fsl_wrappers.flirt(
        fns.mri_file,
        fns.std_brain,
        applyxfm=True,
        init=flirt_mri2mni_xform_file,
        out=flirt_mri_mni_file,
    )

    # ------------------------------------------------------------------------
    # 4) Use BET/BETSURF to get:
    # a) The scalp surface (excluding nose), this gives the MRI-derived
    #    headshape points in native MRI space, which can be used in the
    #    headshape points registration later.
    # b) The scalp surface (outer skin), inner skull and brain surface, these
    #    can be used for forward modelling later. Note that due to the unusual
    #    naming conventions used by BET:
    #    - bet_inskull_mesh_file is actually the brain surface
    #    - bet_outskull_mesh_file is actually the inner skull surface
    #    - bet_outskin_mesh_file is the outer skin/scalp surface
    # ------------------------------------------------------------------------

    print("Running BET and BETSURF...")

    # Run BET and BETSURF on mri to get the surface mesh (in MNI space)
    #
    # Command: bet <flirt_mri_mni_file> <flirt_mri_mni_bet_file> -A
    flirt_mri_mni_bet_file = f"{fns.root}/flirt"
    fsl_wrappers.bet(flirt_mri_mni_file, flirt_mri_mni_bet_file, A=True, **bet_kwargs)

    # ---------------------------------------
    # 5) Add nose to scalp surface (optional)
    # ---------------------------------------

    if include_nose:
        print("Refining scalp surface...")

        # We do this in MNI big FOV space, to allow the full nose to be included

        # Calculate flirt_mni2mnibigfov_xform
        mni2mnibigfov_xform = _get_flirt_xform_between_axes(
            from_nii=flirt_mri_mni_file, target_nii=fns.std_brain_bigfov
        )
        flirt_mni2mnibigfov_xform_file = f"{fns.root}/flirt_mni2mnibigfov_xform.txt"
        np.savetxt(flirt_mni2mnibigfov_xform_file, mni2mnibigfov_xform)

        # Calculate overall transform, from mri to MNI big fov
        #
        # Command: convert_xfm -omat <flirt_mri2mnibigfov_xform_file> \
        #          -concat <flirt_mni2mnibigfov_xform_file> <flirt_mri2mni_xform_file>"
        flirt_mri2mnibigfov_xform_file = f"{fns.root}/flirt_mri2mnibigfov_xform.txt"
        fsl_wrappers.concatxfm(
            flirt_mri2mni_xform_file,
            flirt_mni2mnibigfov_xform_file,
            flirt_mri2mnibigfov_xform_file,
        )  # Note, the wrapper reverses the order of arguments

        # Move MRI to MNI big FOV space and load in
        #
        # Command: flirt -in <mri_file> -ref <std_brain_bigfov> -applyxfm \
        #          -init <flirt_mri2mnibigfov_xform_file> \
        #          -out <flirt_mri_mni_bigfov_file>
        flirt_mri_mni_bigfov_file = f"{fns.root}/flirt_mri_mni_bigfov"
        fsl_wrappers.flirt(
            fns.mri_file,
            fns.std_brain_bigfov,
            applyxfm=True,
            init=flirt_mri2mnibigfov_xform_file,
            out=flirt_mri_mni_bigfov_file,
        )

        # Move scalp to MNI big FOV space and load in
        #
        # Command: flirt -in <flirt_outskin_file> -ref <std_brain_bigfov> \
        #          -applyxfm -init <flirt_mni2mnibigfov_xform_file> \
        #          -out <flirt_outskin_bigfov_file>
        flirt_outskin_file = f"{fns.root}/flirt_outskin_mesh"
        flirt_outskin_bigfov_file = f"{fns.root}/flirt_outskin_mesh_bigfov"
        fsl_wrappers.flirt(
            flirt_outskin_file,
            fns.std_brain_bigfov,
            applyxfm=True,
            init=flirt_mni2mnibigfov_xform_file,
            out=flirt_outskin_bigfov_file,
        )
        scalp = nib.load(f"{flirt_outskin_bigfov_file}.nii.gz")

        # Create mask by filling outline

        # Add a border of ones to the mask, in case the complete head is
        # not in the FOV, without this binary_fill_holes will not work
        mask = np.ones(np.add(scalp.shape, 2))

        # Note that z=100 is where the standard MNI FOV starts in the big FOV
        mask[1:-1, 1:-1, 102:-1] = scalp.get_fdata()[:, :, 101:]
        mask[:, :, :101] = 0

        # We assume that the top of the head is not cutoff by the FOV,
        # we need to assume this so that binary_fill_holes works:
        mask[:, :, -1] = 0
        mask = ndimage.morphology.binary_fill_holes(mask)

        # Remove added border
        mask[:, :, :102] = 0
        mask = mask[1:-1, 1:-1, 1:-1]

        print("Adding nose to scalp surface...")

        # Reclassify bright voxels outside of mask (to put nose inside
        # the mask since bet will have excluded it)
        vol = nib.load(f"{flirt_mri_mni_bigfov_file}.nii.gz")
        vol_data = vol.get_fdata()

        # Normalise vol data
        vol_data = vol_data / np.max(vol_data.flatten())

        # Estimate observation model params of 2 class GMM with diagonal
        # cov matrix where the two classes correspond to inside and outside
        # the bet mask
        means = np.zeros([2, 1])
        means[0] = np.mean(vol_data[np.where(mask == 0)])
        means[1] = np.mean(vol_data[np.where(mask == 1)])
        precisions = np.zeros([2, 1])
        precisions[0] = 1 / np.var(vol_data[np.where(mask == 0)])
        precisions[1] = 1 / np.var(vol_data[np.where(mask == 1)])
        weights = np.zeros([2])
        weights[0] = np.sum((mask == 0))
        weights[1] = np.sum((mask == 1))

        # Create GMM with those observation models
        gm = GaussianMixture(n_components=2, random_state=0, covariance_type="diag")
        gm.means_ = means
        gm.precisions_ = precisions
        gm.precisions_cholesky_ = np.sqrt(precisions)
        gm.weights_ = weights

        # Classify voxels outside BET mask with GMM
        labels = gm.predict(vol_data[np.where(mask == 0)].reshape(-1, 1))

        # Insert new labels for voxels outside BET mask into mask
        mask[np.where(mask == 0)] = labels

        # Ignore anything that is well below the nose and above top of head
        mask[:, :, 0:50] = 0
        mask[:, :, 300:] = 0

        # Clean up mask
        mask[:, :, 50:300] = ndimage.morphology.binary_fill_holes(mask[:, :, 50:300])
        mask[:, :, 50:300] = _binary_majority3d(mask[:, :, 50:300])
        mask[:, :, 50:300] = ndimage.morphology.binary_fill_holes(mask[:, :, 50:300])

        for i in range(mask.shape[0]):
            mask[i, :, 50:300] = ndimage.morphology.binary_fill_holes(
                mask[i, :, 50:300]
            )
        for i in range(mask.shape[1]):
            mask[:, i, 50:300] = ndimage.morphology.binary_fill_holes(
                mask[:, i, 50:300]
            )
        for i in range(50, 300, 1):
            mask[:, :, i] = ndimage.morphology.binary_fill_holes(mask[:, :, i])

        # Extract outline
        outline = np.zeros(mask.shape)
        mask = mask.astype(np.uint8)

        # Use morph gradient to find the outline of the solid mask
        structure = np.ones((3, 3), dtype=bool)

        for i in range(outline.shape[0]):
            slice_bool = mask[i, :, :].astype(bool)
            grad = ndimage.binary_dilation(
                slice_bool, structure=structure
            ) ^ ndimage.binary_erosion(slice_bool, structure=structure)
            outline[i, :, :] += grad.astype(np.uint8)

        for i in range(outline.shape[1]):
            slice_bool = mask[:, i, :].astype(bool)
            grad = ndimage.binary_dilation(
                slice_bool, structure=structure
            ) ^ ndimage.binary_erosion(slice_bool, structure=structure)
            outline[:, i, :] += grad.astype(np.uint8)

        for i in range(50, 300, 1):
            slice_bool = mask[:, :, i].astype(bool)
            grad = ndimage.binary_dilation(
                slice_bool, structure=structure
            ) ^ ndimage.binary_erosion(slice_bool, structure=structure)
            outline[:, :, i] += grad.astype(np.uint8)

        outline /= 3

        outline[np.where(outline > 0.6)] = 1
        outline[np.where(outline <= 0.6)] = 0
        outline = outline.astype(np.uint8)

        # Save as NIFTI
        outline_nii = nib.Nifti1Image(outline, scalp.affine)
        nib.save(outline_nii, f"{flirt_outskin_bigfov_file}_plus_nose.nii.gz")

        # Command: fslcpgeom <src> <dest>
        fsl_wrappers.fslcpgeom(
            f"{flirt_outskin_bigfov_file}.nii.gz",
            f"{flirt_outskin_bigfov_file}_plus_nose.nii.gz",
        )

        # Transform outskin plus nose nii mesh from MNI big FOV to MRI space

        # First we need to invert the flirt_mri2mnibigfov_xform_file xform:
        #
        # Command: convert_xfm -omat <flirt_mnibigfov2mri_xform_file> \
        #          -inverse <flirt_mri2mnibigfov_xform_file>
        flirt_mnibigfov2mri_xform_file = f"{fns.root}/flirt_mnibigfov2mri_xform.txt"
        fsl_wrappers.invxfm(
            flirt_mri2mnibigfov_xform_file,
            flirt_mnibigfov2mri_xform_file,
        )  # Note, the wrapper reverses the order of arguments

        # Command: flirt -in <dest> -ref <smri_file> -applyxfm \
        #          -init <flirt_mnibigfov2mri_xform_file> \
        #          -out <bet_outskin_plus_nose_mesh_file>
        fsl_wrappers.flirt(
            f"{flirt_outskin_bigfov_file}_plus_nose.nii.gz",
            fns.mri_file,
            applyxfm=True,
            init=flirt_mnibigfov2mri_xform_file,
            out=fns.bet_outskin_plus_nose_mesh_file,
        )

    # ----------------------------------------------
    # 6) Output the transform from MRI space to MNI
    # ----------------------------------------------

    flirt_mni2mri = np.loadtxt(mni2mri_flirt_xform_file)
    xform_mni2mri = _get_mne_xform_from_flirt_xform(
        flirt_mni2mri, fns.std_brain, fns.mri_file
    )
    mni_mri_t = Transform("mni_tal", "mri", xform_mni2mri)
    write_trans(fns.mni_mri_t_file, mni_mri_t, overwrite=True)

    # ----------------------------------------
    # 7) Output surfaces in MRI (native) space
    # ----------------------------------------

    # Transform betsurf output mask/mesh output from MNI to MRI space
    for mesh_name in ["outskin_mesh", "inskull_mesh", "outskull_mesh"]:
        # xform mask
        #
        # Command: flirt -in <flirt_mesh_file> -ref <mri_file> \
        #          -interp nearestneighbour -applyxfm \
        #          -init <mni2mri_flirt_xform_file> -out <out_file>
        fsl_wrappers.flirt(
            f"{fns.root}/flirt_{mesh_name}.nii.gz",
            fns.mri_file,
            interp="nearestneighbour",
            applyxfm=True,
            init=mni2mri_flirt_xform_file,
            out=f"{fns.root}/{mesh_name}",
        )

        # xform vtk mesh
        _transform_vtk_mesh(
            f"{fns.root}/flirt_{mesh_name}.vtk",
            f"{fns.root}/flirt_{mesh_name}.nii.gz",
            f"{fns.root}/{mesh_name}.vtk",
            f"{fns.root}/{mesh_name}.nii.gz",
            fns.mni_mri_t_file,
        )

    print("Cleaning up FLIRT files")
    system_call(f"rm -f {fns.root}/flirt*", verbose=False)

    # Plot the surfaces
    plot_surfaces(outdir, id, include_nose=include_nose)
    if not show:
        plt.close("all")

    print("Surface extraction complete.")


def plot_surfaces(
    outdir: str,
    id: str,
    include_nose: bool = True,
) -> None:
    """Plot a structural MRI and extracted surfaces.

    Parameters
    ----------
    outdir : str
        Output directory.
    id : str
        Identifier for the subject/session subdirectory in the output directory.
    include_nose : bool, optional
        Should we also plot the outskin surface including the nose?
    """
    fns = SurfaceFilenames(outdir)

    # Surfaces to plot
    surfaces = ["inskull", "outskull", "outskin"]
    if include_nose:
        surfaces.append("outskin_plus_nose")
    output_files = [f"{fns.root}/{surface}.png" for surface in surfaces]

    # Check surfaces exist
    for surface in surfaces:
        file = Path(getattr(fns, f"bet_{surface}_mesh_file"))
        if not file.exists():
            raise ValueError(f"{file} does not exist")

    # Plot the structural MRI
    from nilearn import plotting

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        display = plotting.plot_anat(fns.mri_file)

    # Plot each surface
    for surface, output_file in zip(surfaces, output_files):
        display_copy = copy.deepcopy(display)
        nii_file = getattr(fns, f"bet_{surface}_mesh_file")
        img = nil.image.load_img(nii_file)
        data = nil.image.get_data(img)
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        display_copy.add_overlay(img, vmin=vmin, vmax=vmax)

        print(f"Saving {output_file}")
        display_copy.savefig(output_file)


def extract_fiducials_and_headshape_from_fif(
    fns: OSLFilenames,
    include_eeg_as_headshape: bool = False,
    include_hpi_as_headshape: bool = True,
) -> None:
    """Extract headshape points and fiducials from FIF info.

    Extract (polhemus) fiducials and headshape points from MNE raw.info and
    write them out in the required file format for RHINO (in HEAD space in
    mm).

    Should only be used with MNE-derived .fif files that have the expected
    digitised points held in info['dig'] of fif_file.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    include_eeg_as_headshape : bool, optional
        Should we include EEG locations as headshape points?
    include_hpi_as_headshape : bool, optional
        Should we include HPI locations as headshape points?
    """
    print()
    print("Extracting fiducials/headshape points from fif info")
    print("---------------------------------------------------")

    # Read info from fif file
    info = mne.io.read_info(fns.preproc_file)

    # Lists to hold polhemus data
    headshape = []
    rpa = []
    lpa = []
    nasion = []

    # Get fiducials/headshape points
    for dig in info["dig"]:

        # Check dig is in HEAD/Polhemus space
        if dig["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
            raise ValueError(f"{dig['ident']} is not in HEAD space")

        if dig["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            if dig["ident"] == FIFF.FIFFV_POINT_LPA:
                lpa = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_RPA:
                rpa = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_NASION:
                nasion = dig["r"]
            else:
                raise ValueError(f"Unknown fiducial: {dig['ident']}")
        elif dig["kind"] == FIFF.FIFFV_POINT_EXTRA:
            headshape.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_EEG and include_eeg_as_headshape:
            headshape.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_HPI and include_hpi_as_headshape:
            headshape.append(dig["r"])

    headshape = np.array(headshape)
    rpa = np.array(rpa)
    lpa = np.array(lpa)
    nasion = np.array(nasion)

    # Check if info is from a CTF scanner
    if info["dev_ctf_t"] is not None:
        print("Detected CTF data")

        nasion = np.copy(nasion)
        lpa = np.copy(lpa)
        rpa = np.copy(rpa)

        nasion[0], nasion[1], nasion[2] = nasion[1], -nasion[0], nasion[2]
        lpa[0], lpa[1], lpa[2] = lpa[1], -lpa[0], lpa[2]
        rpa[0], rpa[1], rpa[2] = rpa[1], -rpa[0], rpa[2]

        # CTF data won't contain headshape points, use a dummy point to avoid errors
        headshape = [0, 0, 0]

    # Save
    print(f"Saved: {fns.coreg.head_nasion_file}")
    np.savetxt(fns.coreg.head_nasion_file, nasion * 1000)
    print(f"Saved: {fns.coreg.head_rpa_file}")
    np.savetxt(fns.coreg.head_rpa_file, rpa * 1000)
    print(f"Saved: {fns.coreg.head_lpa_file}")
    np.savetxt(fns.coreg.head_lpa_file, lpa * 1000)
    print(f"Saved: {fns.coreg.head_headshape_file}")
    np.savetxt(fns.coreg.head_headshape_file, np.array(headshape).T * 1000)

    if info["dev_ctf_t"] is not None:
        print(
            "Dummy headshape points saved, overwrite "
            f"{fns.coreg.head_headshape_file} "
            "or set use_headshape=False in coregisteration."
        )

    # Warning if 'trans' in filename we assume -trans was applied using MaxFiltering
    # This may make the coregistration appear incorrect, but this is not an issue.
    if "_trans" in fns.preproc_file:
        print(
            "fif filename contains '_trans' which suggests -trans was passed "
            "during MaxFiltering. This means the location of the head in the "
            "coregistration plot may not be correct. Either use the _tsss.fif "
            "file or ignore the centroid of the head in coregistration plot."
        )


def extract_fiducials_and_headshape_from_pos(fns: OSLFilenames) -> None:
    """Saves fiducials/headshape from a pos file.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    """
    if fns.pos_file is None:
        raise ValueError("pos_file must have been passed to OSLFilenames")

    print(f"Saving fiducials/headshape points from {fns.pos_file}")

    # These values are in cm in HEAD space
    num_headshape_pnts = int(pd.read_csv(fns.pos_file, header=None).to_numpy()[0])
    data = pd.read_csv(fns.pos_file, header=None, skiprows=[0], sep=r"\s+")

    # RHINO is going to work with distances in mm
    data.iloc[:, 1:4] = data.iloc[:, 1:4] * 10

    # Polhemus fiducial points in HEAD space
    nasion = (
        data[data.iloc[:, 0].str.match("nasion")]
        .iloc[0, 1:4]
        .to_numpy()
        .astype("float64")
        .T
    )
    rpa = (
        data[data.iloc[:, 0].str.match("right")]
        .iloc[0, 1:4]
        .to_numpy()
        .astype("float64")
        .T
    )
    lpa = (
        data[data.iloc[:, 0].str.match("left")]
        .iloc[0, 1:4]
        .to_numpy()
        .astype("float64")
        .T
    )

    # Polhemus headshape points in HEAD space in mm
    headshape = data[0:num_headshape_pnts].iloc[:, 1:4].to_numpy().astype("float64").T

    # Save
    print(f"Saved: {fns.coreg.head_nasion_file}")
    np.savetxt(fns.coreg.head_nasion_file, nasion)
    print(f"Saved: {fns.coreg.head_rpa_file}")
    np.savetxt(fns.coreg.head_rpa_file, rpa)
    print(f"Saved: {fns.coreg.head_lpa_file}")
    np.savetxt(fns.coreg.head_lpa_file, lpa)
    print(f"Saved: {fns.coreg.head_headshape_file}")
    np.savetxt(fns.coreg.head_headshape_file, headshape)


def extract_fiducials_and_headshape_from_elc(
    fns: OSLFilenames,
    remove_nose: bool = True,
) -> None:
    """Extract fiducials and headshape points from an ELC file.

    Reads fiducial locations (nasion, LPA, RPA) and headshape points from
    an ELC file (ANT Neuro) and saves them in the RHINO file format.

    The ELC file is expected to contain a ``Positions`` section with three
    fiducial rows (nasion, LPA, RPA) and a ``HeadShapePoints`` section
    with 3D coordinates. All values are assumed to be in mm.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames. ``fns.elc_file`` must be set.
    remove_nose : bool, optional
        Remove headshape points on the nose? A point is considered to be
        on the nose if it is anterior to both LPA and RPA and inferior to
        the nasion.
    """
    if fns.elc_file is None:
        raise ValueError("elc_file must have been passed to OSLFilenames")

    print(f"Saving fiducials/headshape points from {fns.elc_file}")

    with open(fns.elc_file, "r") as f:
        lines = f.readlines()

    # Extract fiducials from the Positions section
    for i in range(len(lines)):
        if lines[i] == "Positions\n":
            nasion = np.array(lines[i + 1].split()[-3:]).astype(np.float64)
            lpa = np.array(lines[i + 2].split()[-3:]).astype(np.float64)
            rpa = np.array(lines[i + 3].split()[-3:]).astype(np.float64)
            break

    # Extract headshape points from the HeadShapePoints section
    for i in range(len(lines)):
        if lines[i] == "HeadShapePoints\n":
            headshape = (
                np.array([line.split() for line in lines[i + 1 :]]).astype(np.float64).T
            )
            break

    # Optionally remove headshape points on the nose
    if remove_nose:
        on_nose = np.logical_and(
            headshape[0] > max(lpa[0], rpa[0]),
            headshape[2] < nasion[2],
        )
        headshape = headshape[:, ~on_nose]

    # Save
    print(f"Saved: {fns.coreg.head_nasion_file}")
    np.savetxt(fns.coreg.head_nasion_file, nasion)
    print(f"Saved: {fns.coreg.head_rpa_file}")
    np.savetxt(fns.coreg.head_rpa_file, rpa)
    print(f"Saved: {fns.coreg.head_lpa_file}")
    np.savetxt(fns.coreg.head_lpa_file, lpa)
    print(f"Saved: {fns.coreg.head_headshape_file}")
    np.savetxt(fns.coreg.head_headshape_file, headshape)


def remove_stray_headshape_points(
    fns: OSLFilenames,
    nose: bool = True,
) -> None:
    """Remove stray headshape points.

    Removes headshape points near the nose, on the neck or far away from the head.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    nose : bool, optional
        Should we remove headshape points near the nose? Useful to remove these
        if we have defaced structurals or aren't extracting the nose from the
        structural.
    """
    fns = fns.coreg

    # Load saved headshape and nasion files
    hs = np.loadtxt(fns.head_headshape_file)
    nas = np.loadtxt(fns.head_nasion_file)
    lpa = np.loadtxt(fns.head_lpa_file)
    rpa = np.loadtxt(fns.head_rpa_file)

    if nose:
        # Remove headshape points on the nose
        remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
        hs = hs[:, ~remove]

    # Remove headshape points on the neck
    remove = hs[2] < min(lpa[2], rpa[2]) - 4
    hs = hs[:, ~remove]

    # Remove headshape points far from the head in any direction
    remove = np.logical_or(
        hs[0] < lpa[0] - 5, np.logical_or(hs[0] > rpa[0] + 5, hs[1] > nas[1] + 5)
    )
    hs = hs[:, ~remove]

    # Overwrite headshape file
    print(f"Overwriting: {fns.head_headshape_file}")
    np.savetxt(fns.head_headshape_file, hs)


def save_coregistration_files(fns: OSLFilenames) -> None:
    """Data is already coregistered, just save the files needed for RHINO.

    Assumes that the sensor locations and fiducials/headshape points (if
    there are any) are already in MRI space. This means that dev_head_t
    is identity and that dev_mri_t is identity.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    """

    print()
    print("Running dummy coregistration")
    print("----------------------------")

    # Paths to files
    cfns = fns.coreg
    sfns = fns.surfaces

    # ------------------------------------------
    # Copy fif_file to new file for modification
    # ------------------------------------------

    # Get info from fif file
    info = mne.io.read_info(fns.preproc_file)

    raw = mne.io.RawArray(np.zeros([len(info["ch_names"]), 1]), info)
    raw.save(cfns.info_fif_file, overwrite=True)

    # Write native (mri) voxel index to native (mri) transform
    xform_nativeindex2scalednative = _get_sform(sfns.bet_outskin_mesh_file)["trans"]
    mrivoxel_scaledmri_t = Transform(
        "mri_voxel", "mri", np.copy(xform_nativeindex2scalednative)
    )
    write_trans(cfns.mrivoxel_scaledmri_t_file, mrivoxel_scaledmri_t, overwrite=True)

    # head_mri-trans.fif for scaled MRI
    head_mri_t = Transform("head", "mri", np.identity(4))
    write_trans(cfns.head_mri_t_file, head_mri_t, overwrite=True)
    write_trans(cfns.head_scaledmri_t_file, head_mri_t, overwrite=True)

    # Copy meshes to coreg dir from surfaces dir
    for filename in [
        "mri_file",
        "bet_outskin_mesh_file",
        "bet_outskin_plus_nose_mesh_file",
        "bet_inskull_mesh_file",
        "bet_outskull_mesh_file",
        "bet_outskin_mesh_vtk_file",
        "bet_inskull_mesh_vtk_file",
        "bet_outskull_mesh_vtk_file",
    ]:
        src = getattr(sfns, filename)
        dst = getattr(cfns, filename)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    # ------------------------------------------------------------------------
    # Create sMRI-derived freesurfer meshes in native/mri space in mm, for use
    # by forward modelling
    # ------------------------------------------------------------------------

    nativeindex_scalednative_t = np.copy(xform_nativeindex2scalednative)
    mrivoxel_scaledmri_t = Transform("mri_voxel", "mri", nativeindex_scalednative_t)
    _create_freesurfer_meshes_from_bet_surfaces(cfns, mrivoxel_scaledmri_t["trans"])

    # -----------------------
    # Plot the coregistration
    # -----------------------

    plot_coregistration(
        fns,
        display_sensors=False,
        display_fiducials=False,
        display_headshape_pnts=False,
        include_nose=False,
    )

    print("Coregistration complete.")


def coregister_head_and_mri(
    fns: OSLFilenames,
    use_headshape: bool = True,
    use_nose: bool = True,
    allow_mri_scaling: bool = False,
    mni_fiducials: Optional[Dict[str, np.ndarray]] = None,
    n_init: int = 1,
    plot_type: str = "png",
    show: bool = False,
) -> None:
    """Coregister HEAD (polhemus) and MRI space.

    Calculates a linear, affine transform from HEAD space to MRI space
    using headshape points (if use_headshape=True).

    Requires ``rhino.extract_surfaces`` to have been run.

    RHINO firsts registers the polhemus-derived fiducials (nasion, rpa, lpa) in
    HEAD space to the MRI-derived fiducials in native MRI space.

    RHINO then refines this by making use of polhemus-derived headshape points
    that trace out the surface of the head (scalp).

    Finally, these polhemus-derived headshape points in HEAD space are
    registered to the MRI-derived scalp surface in native MRI space.

    In more detail:
    1)  Map location of fiducials in MNI standard space brain to native MRI
        space. These are then used as the location of the MRI-derived fiducials
        in native MRI space.

    2a) We have polhemus-derived fiducials in HEAD space and MRI-derived fiducials
        in native MRI space. Use these to estimate the affine xform from native
        MRI space to HEAD space.

    2b) We can also optionally learn the best scaling to add to this affine
        xform, such that the MRI-derived fiducials are scaled in size to better
        match the polhemus-derived fiducials. This assumes that we trust the size
        (e.g. in mm) of the polhemus-derived fiducials, but not the size of
        MRI-derived fiducials. E.g. this might be the case if we do not trust
        the size (e.g. in mm) of the MRI, or if we are using a template MRI
        that would has not come from this subject.

    3)  If a scaling is learnt in step 2, we apply it to MRI, and to anything
        derived from MRI.

    4)  Transform MRI-derived headshape points into HEAD space.

    5)  We have the polhemus-derived headshape points in HEAD space and the
        MRI-derived headshape (scalp surface) in native MRI space.  Use these
        to estimate the affine xform from native MRI space using the ICP
        algorithm initilaised using the xform estimate in step 2.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    use_headshape : bool, optional
        Determines whether polhemus derived headshape points are used.
    use_nose : bool, optional
        Determines whether nose is used to aid coregistration,
        only relevant if use_headshape=True.
    allow_mri_scaling : bool, optional
        Indicates if we are to allow scaling of the MRI, such that the
        MRI-derived fiducials are scaled in size to better match the
        polhemus-derived fiducials. This assumes that we trust the size
        (e.g. in mm) of the polhemus-derived fiducials, but not the size
        of the MRI-derived fiducials.
        E.g. this might be the case if we do not trust the size (e.g. in mm)
        of the MRI, or if we are using a template MRI that has not come from
        this subject.
    mni_fiducials : list, optional
        Fiducials for the MRI in MNI space. Must be [nasion, rpa, lpa],
        where nasion, rpa, lpa are 3D coordinates.
        Defaults to [[1, 85, -41], [83, -20, -65], [-83, -20, -65]].
    n_init : int, optional
        Number of initialisations for the ICP algorithm that performs coregistration.
    plot_type : str, optional
        Type of coregistration plot to save. Options are "png", "html" or None.
        "png" saves static PNG images (requires a display/render window).
        "html" saves an interactive HTML file (works in headless environments).
        None skips plotting.
    show : bool, optional
        Whether to display the coregistration plot interactively. Default is
        False (suitable for batch processing).
    """

    # Note the jargon used varies for xforms and coord spaces:
    # - MEG (device): dev_head_t --> HEAD (polhemus)
    # - HEAD (polhemus): head_mri_t (polhemus2native) --> MRI (native)
    # - MRI (native): mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices

    # RHINO does everything in mm

    print()
    print("Running coregistration (HEAD (polhemus) -> MRI)")
    print("-----------------------------------------------")

    # Paths to files
    cfns = fns.coreg
    sfns = fns.surfaces

    if not use_headshape:
        use_nose = False

    if use_nose:
        print("The MRI-derived nose is going to be used to aid coregistration.")
        print(
            "Please ensure that rhino.extract_surfaces was run with include_nose=True."
        )
        print("Please ensure that the headshape points include the nose.")
    else:
        print("The MRI-derived nose is not going to be used to aid coregistration.")
        print("Please ensure that the headshape points do not include the nose")

    # Copy fif_file to new file for modification
    # and change dev_head_t to equal dev_ctf_t in fif file info
    info = mne.io.read_info(fns.preproc_file)

    dev_ctf_t = info["dev_ctf_t"]
    if dev_ctf_t is not None:
        print("Detected CTF data")
        print("Setting dev_head_t equal to dev_ctf_t in fif file info.")
        dev_head_t, _ = _get_trans(info["dev_head_t"], "meg", "head")
        dev_head_t["trans"] = dev_ctf_t["trans"]

    raw = mne.io.RawArray(np.zeros([len(info["ch_names"]), 1]), info)
    raw.save(cfns.info_fif_file, overwrite=True)

    # Load in the "polhemus-derived fiducials" in HEAD space
    print(f"Loading: {cfns.head_headshape_file}")
    polhemus_headshape_head = np.loadtxt(cfns.head_headshape_file)

    print(f"Loading: {cfns.head_nasion_file}")
    polhemus_nasion_head = np.loadtxt(cfns.head_nasion_file)

    print(f"Loading: {cfns.head_rpa_file}")
    polhemus_rpa_head = np.loadtxt(cfns.head_rpa_file)

    print(f"Loading: {cfns.head_lpa_file}")
    polhemus_lpa_head = np.loadtxt(cfns.head_lpa_file)

    # ----------------------------------------------------------------------
    # 1) Map location of fiducials in MNI standard space brain to native
    #    MRI space. These are then used as the location of the MRI-derived
    #    fiducials in native MRI space.
    # ----------------------------------------------------------------------

    if mni_fiducials is None:
        # Known locations of MNI derived fiducials in MNI coords
        print("Using known MNI-derived fiducials")
        mni_fiducials = [[1, 85, -41], [83, -20, -65], [-83, -20, -65]]

    mni_nasion_mni = np.asarray(mni_fiducials[0])
    mni_rpa_mni = np.asarray(mni_fiducials[1])
    mni_lpa_mni = np.asarray(mni_fiducials[2])

    mni_mri_t = read_trans(sfns.mni_mri_t_file)

    # Apply this xform to the MNI fiducials to get what we call the
    # "MRI-derived fiducials" in native space
    mri_nasion_native = _xform_points(mni_mri_t["trans"], mni_nasion_mni)
    mri_lpa_native = _xform_points(mni_mri_t["trans"], mni_lpa_mni)
    mri_rpa_native = _xform_points(mni_mri_t["trans"], mni_rpa_mni)

    # ----------------------------------------------------------------------
    # 2a) We have polhemus-derived fiducials in polhemus space and MRI-derived
    #     fiducials in native MRI space. Use these to estimate the affine xform
    #     from native MRI space to polhemus (head) space.
    #
    # 2b) We can also optionally learn the best scaling to add to this
    #     affine xform, such that the MRI-derived fiducials are scaled in size
    #     to better match the polhemus-derived fiducials. This assumes that we
    #     trust the size (e.g. in mm) of the polhemus-derived fiducials, but not
    #     the size of the MRI-derived fiducials. E.g. this might be the case if
    #     we do not trust the size (e.g. in mm) of the MRI, or if we are
    #     using a template MRI that has not come from this subject.
    # ----------------------------------------------------------------------

    # Note that mri_fid_native are the MRI-derived fiducials in native space
    polhemus_fid_head = np.concatenate(
        (
            np.reshape(polhemus_nasion_head, [-1, 1]),
            np.reshape(polhemus_rpa_head, [-1, 1]),
            np.reshape(polhemus_lpa_head, [-1, 1]),
        ),
        axis=1,
    )
    mri_fid_native = np.concatenate(
        (
            np.reshape(mri_nasion_native, [-1, 1]),
            np.reshape(mri_rpa_native, [-1, 1]),
            np.reshape(mri_lpa_native, [-1, 1]),
        ),
        axis=1,
    )

    # Estimate the affine xform from native MRI space to polhemus (head)
    # space. Optionally includes a scaling of the MRI, captured by
    # xform_native2scalednative
    xform_scalednative2head, xform_native2scalednative = _rigid_transform_3D(
        polhemus_fid_head,
        mri_fid_native,
        compute_scaling=allow_mri_scaling,
    )

    # ----------------------------------------------------------------------
    # 3) Apply scaling from xform_native2scalednative to MRI, and to stuff
    #    derived from MRI, including:
    #    - MRI
    #    - MRI-derived surfaces
    #    - MRI-derived fiducials
    # ----------------------------------------------------------------------

    # Scale MRI and MRI-derived mesh files by changing their sform
    xform_nativeindex2native = _get_sform(sfns.mri_file)["trans"]
    xform_nativeindex2scalednative = (
        xform_native2scalednative @ xform_nativeindex2native
    )
    filenames = [
        "mri_file",
        "bet_outskin_mesh_file",
        "bet_inskull_mesh_file",
        "bet_outskull_mesh_file",
    ]
    if use_nose:
        filenames.append("bet_outskin_plus_nose_mesh_file")
    for filename in filenames:
        shutil.copyfile(getattr(sfns, filename), getattr(cfns, filename))
        # Command: fslorient -setsform <sform> <mri_file>
        sform = xform_nativeindex2scalednative.flatten()
        fsl_wrappers.misc.fslorient(getattr(cfns, filename), setsform=tuple(sform))

    # Scale vtk meshes
    for mesh_fname, vtk_fname in zip(
        [
            "bet_outskin_mesh_file",
            "bet_inskull_mesh_file",
            "bet_outskull_mesh_file",
        ],
        [
            "bet_outskin_mesh_vtk_file",
            "bet_inskull_mesh_vtk_file",
            "bet_outskull_mesh_vtk_file",
        ],
    ):
        _transform_vtk_mesh(
            getattr(sfns, vtk_fname),
            getattr(sfns, mesh_fname),
            getattr(cfns, vtk_fname),
            getattr(cfns, mesh_fname),
            xform_native2scalednative,
        )

    # Put MRI-derived fiducials into scaled MRI space
    xform = xform_native2scalednative @ mni_mri_t["trans"]
    mri_nasion_scalednative = _xform_points(xform, mni_nasion_mni)
    mri_lpa_scalednative = _xform_points(xform, mni_lpa_mni)
    mri_rpa_scalednative = _xform_points(xform, mni_rpa_mni)

    # --------------------------------------------------------------------
    # 4) Now we can transform MRI-derived headshape points into HEAD space
    # --------------------------------------------------------------------

    # File containing the "MRI-derived headshape points"
    if use_nose:
        outskin_mesh_file = cfns.bet_outskin_plus_nose_mesh_file
    else:
        outskin_mesh_file = cfns.bet_outskin_mesh_file

    # Get native (mri) voxel index to scaled native (mri) transform
    xform_nativeindex2scalednative = _get_sform(outskin_mesh_file)["trans"]

    # Put MRI-derived headshape points into native space (in mm)
    mri_headshape_nativeindex = _niimask2indexpointcloud(outskin_mesh_file)
    mri_headshape_scalednative = _xform_points(
        xform_nativeindex2scalednative, mri_headshape_nativeindex
    )

    # Put MRI-derived headshape points into HEAD space
    mri_headshape_head = _xform_points(
        xform_scalednative2head, mri_headshape_scalednative
    )

    # ----------------------------------------------------------------------
    # 5) We have the polhemus-derived headshape points in HEAD space and
    #    the MRI-derived headshape (scalp surface) in native MRI space. We
    #    use these to estimate the affine xform from native MRI space using
    #    the ICP algorithm initilaised using the xform estimate in step 2.
    # ----------------------------------------------------------------------

    if use_headshape:
        print("Running ICP...")

        # Run ICP with multiple initialisations to refine registration of
        # MRI-derived headshape points to polhemus derived headshape points,
        # with both in HEAD space

        # Combined polhemus-derived headshape points and polhemus-derived
        # fiducials, with them both in HEAD space. These are the "source"
        # points that will be moved around
        polhemus_headshape_head_4icp = np.concatenate(
            (polhemus_headshape_head, polhemus_fid_head),
            axis=1,
        )

        xform_icp, _, e = _rhino_icp(
            mri_headshape_head,
            polhemus_headshape_head_4icp,
            n_init=n_init,
        )

    else:
        # No refinement by ICP:
        xform_icp = np.eye(4)

    # Create refined xforms using result from ICP
    xform_scalednative2head_refined = np.linalg.inv(xform_icp) @ xform_scalednative2head

    # Put MRI-derived fiducials into refined HEAD space
    mri_nasion_head = _xform_points(
        xform_scalednative2head_refined, mri_nasion_scalednative
    )
    mri_rpa_head = _xform_points(xform_scalednative2head_refined, mri_rpa_scalednative)
    mri_lpa_head = _xform_points(xform_scalednative2head_refined, mri_lpa_scalednative)

    # ---------------
    # Save coreg info
    # ---------------

    # Save xforms in MNE format in mm

    # Save xform from head to mri for the scaled mri
    head_scaledmri_t = Transform(
        "head", "mri", np.linalg.inv(xform_scalednative2head_refined)
    )
    write_trans(cfns.head_scaledmri_t_file, head_scaledmri_t, overwrite=True)

    # Save xform from head to mri for the unscaled mri, this is needed if
    # we later want to map back into MNI space from head space following
    # source recon, i.e. by combining this xform with sfns.mni_mri_t_file
    xform_native2head_refined = (
        np.linalg.inv(xform_icp) @ xform_scalednative2head @ xform_native2scalednative
    )
    xform_native2head_refined_copy = np.copy(xform_native2head_refined)
    head_mri_t = Transform("head", "mri", np.linalg.inv(xform_native2head_refined_copy))
    write_trans(cfns.head_mri_t_file, head_mri_t, overwrite=True)

    # Save xform from mrivoxel to mri
    nativeindex_scalednative_t = np.copy(xform_nativeindex2scalednative)
    mrivoxel_scaledmri_t = Transform("mri_voxel", "mri", nativeindex_scalednative_t)
    write_trans(cfns.mrivoxel_scaledmri_t_file, mrivoxel_scaledmri_t, overwrite=True)

    # Save MRI derived fiducials in mm in HEAD space
    np.savetxt(cfns.mri_nasion_file, mri_nasion_head)
    np.savetxt(cfns.mri_rpa_file, mri_rpa_head)
    np.savetxt(cfns.mri_lpa_file, mri_lpa_head)

    # ------------------------------------------------------------------------
    # Create MRI-derived freesurfer meshes in native/mri space in mm, for use
    # by forward modelling
    # ------------------------------------------------------------------------

    nativeindex_scalednative_t = np.copy(xform_nativeindex2scalednative)
    mrivoxel_scaledmri_t = Transform("mri_voxel", "mri", nativeindex_scalednative_t)
    _create_freesurfer_meshes_from_bet_surfaces(cfns, mrivoxel_scaledmri_t["trans"])

    # -----------------------
    # Plot the coregistration
    # -----------------------
    if plot_type is not None:
        filename = f"{fns.coreg_dir}/coreg.{plot_type}"
        plot_coregistration(fns, include_nose=use_nose, filename=filename, show=show)

    print("Coregistration complete.")


def plot_coregistration(
    fns: OSLFilenames,
    display_outskin: bool = True,
    display_sensors: bool = True,
    display_sensor_oris: bool = True,
    display_fiducials: bool = True,
    display_headshape_pnts: bool = True,
    include_nose: bool = True,
    filename: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot coregistration.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    display_outskin : bool, optional
        Whether to show scalp surface in the display.
    display_sensors : bool, optional
        Whether to include sensors in the display.
    display_sensor_oris : bool, optional
        Whether to include sensor orientations in the display.
    display_fiducials : bool, optional
        Whether to include fiducials in the display.
    display_headshape_pnts : bool, optional
        Whether to include headshape points in the display.
    include_nose : bool, option
        Should we use the outskin surface with the nose?
    filename : str, optional
        Filename to save display to (as an interactive html).
        Must have extension .html.
    show : bool, optional
        Should we show the plots? Only used if filename has
        extension '.png'.
    """

    # Note the jargon used varies for xforms and coord spaces:
    # MEG (device): dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus): head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native): mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices

    # RHINO does everything in mm

    print("Plotting coregistration")

    if filename is None:
        filename = f"{fns.coreg_dir}/coreg.png"

    fns = fns.coreg

    bet_outskin_mesh_file = fns.bet_outskin_mesh_file
    bet_outskin_mesh_vtk_file = fns.bet_outskin_mesh_vtk_file
    bet_outskin_surf_file = fns.bet_outskin_surf_file

    bet_outskin_plus_nose_mesh_file = fns.bet_outskin_plus_nose_mesh_file
    bet_outskin_plus_nose_surf_file = fns.bet_outskin_plus_nose_surf_file

    head_scaledmri_t_file = fns.head_scaledmri_t_file
    mrivoxel_scaledmri_t_file = fns.mrivoxel_scaledmri_t_file
    mri_nasion_file = fns.mri_nasion_file
    mri_rpa_file = fns.mri_rpa_file
    mri_lpa_file = fns.mri_lpa_file
    head_nasion_file = fns.head_nasion_file
    head_rpa_file = fns.head_rpa_file
    head_lpa_file = fns.head_lpa_file
    head_headshape_file = fns.head_headshape_file
    info_fif_file = fns.info_fif_file

    if include_nose:
        outskin_mesh_file = bet_outskin_plus_nose_mesh_file
        outskin_mesh_4surf_file = bet_outskin_plus_nose_mesh_file
        outskin_surf_file = bet_outskin_plus_nose_surf_file
    else:
        outskin_mesh_file = bet_outskin_mesh_file
        outskin_mesh_4surf_file = bet_outskin_mesh_vtk_file
        outskin_surf_file = bet_outskin_surf_file

    # ------------
    # Setup xforms
    # ------------
    info = mne.io.read_info(info_fif_file)

    mrivoxel_scaledmri_t = read_trans(mrivoxel_scaledmri_t_file)
    head_scaledmri_t = read_trans(head_scaledmri_t_file)
    dev_head_t, _ = _get_trans(info["dev_head_t"], "meg", "head")

    # Change xform from metres to mm.
    # Note that MNE xform in fif.info assume metres, whereas we want it in mm.
    # To change units for an xform, just need to change the translation part
    # and leave the rotation alone.
    dev_head_t["trans"][0:3, -1] = dev_head_t["trans"][0:3, -1] * 1000

    # We are going to display everything in MEG (device) coord frame in mm
    head_trans = invert_transform(dev_head_t)
    meg_trans = Transform("meg", "meg")
    mri_trans = invert_transform(
        combine_transforms(dev_head_t, head_scaledmri_t, "meg", "mri")
    )

    # ------------------------------------
    # Setup fiducials and headshape points
    # ------------------------------------
    if display_fiducials:
        # Load polhemus-derived fiducials, these are in mm in HEAD space
        polhemus_nasion_meg = None
        if os.path.isfile(head_nasion_file):
            polhemus_nasion_head = np.loadtxt(head_nasion_file)
            polhemus_nasion_meg = _xform_points(
                head_trans["trans"], polhemus_nasion_head
            )
        polhemus_rpa_meg = None
        if os.path.isfile(head_rpa_file):
            polhemus_rpa_head = np.loadtxt(head_rpa_file)
            polhemus_rpa_meg = _xform_points(head_trans["trans"], polhemus_rpa_head)
        polhemus_lpa_meg = None
        if os.path.isfile(head_lpa_file):
            polhemus_lpa_head = np.loadtxt(head_lpa_file)
            polhemus_lpa_meg = _xform_points(head_trans["trans"], polhemus_lpa_head)

        # Load MRI derived fiducials, these are in mm in HEAD space
        mri_nasion_meg = None
        if os.path.isfile(mri_nasion_file):
            mri_nasion_head = np.loadtxt(mri_nasion_file)
            mri_nasion_meg = _xform_points(head_trans["trans"], mri_nasion_head)
        mri_rpa_meg = None
        if os.path.isfile(mri_rpa_file):
            mri_rpa_head = np.loadtxt(mri_rpa_file)
            mri_rpa_meg = _xform_points(head_trans["trans"], mri_rpa_head)
        mri_lpa_meg = None
        if os.path.isfile(mri_lpa_file):
            mri_lpa_head = np.loadtxt(mri_lpa_file)
            mri_lpa_meg = _xform_points(head_trans["trans"], mri_lpa_head)
    else:
        polhemus_nasion_meg = polhemus_rpa_meg = polhemus_lpa_meg = None
        mri_nasion_meg = mri_rpa_meg = mri_lpa_meg = None

    if display_headshape_pnts:
        polhemus_headshape_meg = None
        if os.path.isfile(head_headshape_file):
            polhemus_headshape_head = np.loadtxt(head_headshape_file)
            polhemus_headshape_meg = _xform_points(
                head_trans["trans"], polhemus_headshape_head
            )
    else:
        polhemus_headshape_meg = None

    # -----------------
    # Setup MEG sensors
    # -----------------
    meg_rrs, meg_tris, meg_sensor_locs, meg_sensor_oris = [], [], [], []
    try:
        if display_sensors or display_sensor_oris:
            meg_picks = mne.pick_types(info, meg=True, ref_meg=False, exclude=())
            coil_transs = [
                mne._fiff.tag._loc_to_coil_trans(info["chs"][pick]["loc"])
                for pick in meg_picks
            ]
            coils = mne.forward._create_meg_coils(
                [info["chs"][pick] for pick in meg_picks], acc="normal"
            )

            degenerate_sensor_indices = []
            offset = 0
            sensor_idx = 0

            for coil, coil_trans in zip(coils, coil_transs):
                try:
                    rrs, tris = mne.viz._3d._sensor_shape(coil)
                except Exception as exc:
                    is_qhull = (
                        isinstance(exc, RuntimeError)
                        or "Qhull" in repr(exc)
                        or "Initial simplex is flat" in str(exc)
                    )
                    if is_qhull:
                        # Create a tiny 3-point triangle in metres (so later transforms work)
                        tiny = 0.001  # 1 mm
                        rrs = np.array(
                            [[0.0, 0.0, 0.0], [tiny, 0.0, 0.0], [-tiny, 0.0, 0.0]]
                        )
                        tris = np.array([[0, 1, 2]])
                        degenerate_sensor_indices.append(sensor_idx)
                    else:
                        # Unexpected exception - re-raise
                        raise

                # apply coil transform to get coil shape in device coords (metres)
                rrs = apply_trans(coil_trans, rrs)
                meg_rrs.append(rrs)
                meg_tris.append(tris + offset)

                # sensor location: origin transformed by coil_trans
                sens_locs = np.array([[0.0, 0.0, 0.0]])
                sens_locs = apply_trans(coil_trans, sens_locs)

                # orientation: unit z vector (small scale)
                sens_oris = np.array([[0.0, 0.0, 1.0]]) * 0.01
                sens_oris = apply_trans(coil_trans, sens_oris)
                sens_oris = sens_oris - sens_locs

                meg_sensor_locs.append(sens_locs)
                meg_sensor_oris.append(sens_oris)

                offset += len(rrs)
                sensor_idx += 1

            if len(meg_rrs) == 0:
                print("MEG sensors not found. Cannot plot MEG locations.")
            else:
                meg_rrs = apply_trans(meg_trans, np.concatenate(meg_rrs, axis=0))
                meg_sensor_locs = apply_trans(
                    meg_trans, np.concatenate(meg_sensor_locs, axis=0)
                )
                meg_sensor_oris = apply_trans(
                    meg_trans, np.concatenate(meg_sensor_oris, axis=0)
                )
                meg_tris = np.concatenate(meg_tris, axis=0)

            # convert to mm
            meg_rrs = meg_rrs * 1000
            meg_sensor_locs = meg_sensor_locs * 1000
            meg_sensor_oris = meg_sensor_oris * 1000

    except Exception as e:
        # If anything goes catastrophically wrong in sensor setup, report and continue
        print(f"Warning: problem setting up MEG sensors: {e}")
        meg_rrs = np.array([]).reshape((0, 3))
        meg_tris = np.array([]).reshape((0, 3))
        meg_sensor_locs = np.array([]).reshape((0, 3))
        meg_sensor_oris = np.array([]).reshape((0, 3))

    # --------
    # Do plots
    # --------
    import pyvista

    pyvista.OFF_SCREEN = True
    mne.viz.set_3d_backend("notebook")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Initialize figure
        renderer = _get_renderer(None, bgcolor=(0.5, 0.5, 0.5), size=(500, 500))

        # Headshape points
        if display_headshape_pnts:
            if polhemus_headshape_meg is not None and len(polhemus_headshape_meg.T) > 0:
                polhemus_headshape_megt = polhemus_headshape_meg.T
                if len(polhemus_headshape_megt) < 200:
                    scale = 0.007
                elif (
                    len(polhemus_headshape_megt) >= 200
                    and len(polhemus_headshape_megt) < 400
                ):
                    scale = 0.005
                elif len(polhemus_headshape_megt) >= 400:
                    scale = 0.003
                color, alpha = "red", 1
                renderer.sphere(
                    center=polhemus_headshape_megt,
                    color=color,
                    scale=scale * 1000,
                    opacity=alpha,
                    backface_culling=True,
                )
            else:
                print("There are no headshape points to display")

        # Fiducials
        if display_fiducials:
            # MRI-derived nasion, rpa, lpa
            if mri_nasion_meg is not None and len(mri_nasion_meg.T) > 0:
                color, scale, alpha = "yellow", 0.09, 1
                for data in [mri_nasion_meg.T, mri_rpa_meg.T, mri_lpa_meg.T]:
                    transform = np.eye(4)
                    transform[:3, :3] = mri_trans["trans"][:3, :3] * scale * 1000
                    transform = transform @ rotation(0, 0, np.pi / 4)
                    renderer.quiver3d(
                        x=data[:, 0],
                        y=data[:, 1],
                        z=data[:, 2],
                        u=1.0,
                        v=0.0,
                        w=0.0,
                        color=color,
                        mode="oct",
                        scale=scale,
                        opacity=alpha,
                        backface_culling=True,
                        solid_transform=transform,
                    )
            else:
                print("There are no MRI derived fiducials to display")

            # Polhemus-derived nasion, rpa, lpa
            if polhemus_nasion_meg is not None and len(polhemus_nasion_meg.T) > 0:
                color, scale, alpha = "pink", 0.012, 1
                for data in [
                    polhemus_nasion_meg.T,
                    polhemus_rpa_meg.T,
                    polhemus_lpa_meg.T,
                ]:
                    renderer.sphere(
                        center=data,
                        color=color,
                        scale=scale * 1000,
                        opacity=alpha,
                        backface_culling=True,
                    )
            else:
                print("There are no polhemus derived fiducials to display")

        # Sensors
        if display_sensors:
            if len(meg_rrs) > 0:
                color, alpha = (0.0, 0.25, 0.5), 0.2
                surf = dict(rr=meg_rrs, tris=meg_tris)
                renderer.surface(
                    surface=surf,
                    color=color,
                    opacity=alpha,
                    backface_culling=True,
                )
            else:
                print("No sensor surfaces available to display")

        # Sensor orientations (arrows)
        if display_sensor_oris:
            if len(meg_sensor_locs) > 0:
                color, scale = (0.0, 0.25, 0.5), 15
                renderer.quiver3d(
                    x=meg_sensor_locs[:, 0],
                    y=meg_sensor_locs[:, 1],
                    z=meg_sensor_locs[:, 2],
                    u=meg_sensor_oris[:, 0],
                    v=meg_sensor_oris[:, 1],
                    w=meg_sensor_oris[:, 2],
                    color=color,
                    mode="arrow",
                    scale=scale,
                    backface_culling=False,
                )

        # Outskin surface
        if display_outskin:
            _create_freesurfer_mesh_from_bet_surface(
                infile=outskin_mesh_4surf_file,
                surf_outfile=outskin_surf_file,
                nii_mesh_file=outskin_mesh_file,
                xform_mri_voxel2mri=mrivoxel_scaledmri_t["trans"],
            )
            coords_native, faces = nib.freesurfer.read_geometry(outskin_surf_file)

            coords_meg = _xform_points(mri_trans["trans"], coords_native.T).T
            surf_mri = dict(rr=coords_meg, tris=faces)

            renderer.surface(
                surface=surf_mri,
                color=(0, 0.7, 0.7),
                opacity=0.4,
                backface_culling=False,
            )

        renderer.set_camera(
            azimuth=90,
            elevation=90,
            distance=600,
            focalpoint=(0.0, 0.0, 0.0),
        )

        # Save
        ext = Path(filename).suffix.lower()

        if ext == ".html":
            print(f"Saving {filename}")
            renderer.figure.plotter.export_html(filename)

        elif ext == ".png":
            # Capture three views and composite into a single PNG
            views = [
                (
                    "Frontal",
                    dict(
                        azimuth=90,
                        elevation=90,
                        distance=600,
                        focalpoint=(0.0, 0.0, 0.0),
                    ),
                ),
                (
                    "Right",
                    dict(
                        azimuth=0,
                        elevation=90,
                        distance=600,
                        focalpoint=(0.0, 0.0, 0.0),
                    ),
                ),
                (
                    "Top",
                    dict(
                        azimuth=90,
                        elevation=0,
                        distance=600,
                        focalpoint=(0.0, 0.0, 0.0),
                    ),
                ),
            ]

            plotter = renderer.figure.plotter
            screenshots = []

            for name, cam in views:
                renderer.set_camera(
                    azimuth=cam["azimuth"],
                    elevation=cam["elevation"],
                    distance=cam["distance"],
                    focalpoint=cam["focalpoint"],
                )
                screenshots.append(plotter.screenshot())

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, img, (name, _) in zip(axes, screenshots, views):
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(name, fontsize=22)
            fig.tight_layout()
            print(f"Saving {filename}")
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

        else:
            raise ValueError("Extension must be png or html.")


def forward_model(
    fns: OSLFilenames,
    model: str = "Single Layer",
    gridstep: int = 8,
    mindist: float = 4.0,
    exclude: float = 0.0,
    eeg: bool = False,
    meg: bool = True,
    verbose: bool = False,
) -> None:
    """Compute forward model.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    model : string, optional
        Options are:
        - 'Single Layer' to use single layer (brain/cortex).
          Recommended for MEG.
        - 'Triple Layer' to three layers (scalp, inner skull, brain/cortex).
          Recommended for EEG.
    gridstep : int, optional
        A grid will be constructed with the spacing given by ``gridstep`` in mm
        generating a volume source space.
    mindist : float
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude : float, optional
        Exclude points closer than this distance (mm) from the center of mass of
        the bounding surface.
    eeg : bool, optional
        Whether to compute forward model for EEG sensors.
    meg : bool, optional
        Whether to compute forward model for MEG sensors.
    """
    print()
    print("Calculating forward model")
    print("-------------------------")

    # Compute MNE bem solution
    if model == "Single Layer":
        conductivity = (0.3,)  # for single layer
    elif model == "Triple Layer":
        conductivity = (0.3, 0.006, 0.3)  # for three layers
    else:
        raise ValueError(f"{model} is an invalid model choice")

    vol_src = _setup_volume_source_space(
        fns,
        gridstep=gridstep,
        mindist=mindist,
        exclude=exclude,
    )

    # The BEM solution requires a BEM model which describes the geometry of the
    # head the conductivities of the different tissues.
    # See: https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py
    #
    # Note that the BEM does not involve any use of transforms between spaces.
    # The BEM only depends on the head geometry and conductivities.
    # It is therefore independent from the MEG data and the head position.
    #
    # This will get the surfaces from: subjects_dir/subject/bem/inner_skull.surf,
    # which is where rhino.setup_volume_source_space will have put it.

    model = mne.make_bem_model(
        subjects_dir=fns.outdir,
        subject=fns.id,
        ico=None,
        conductivity=conductivity,
        verbose=verbose,
    )
    bem = mne.make_bem_solution(model)
    fwd = _make_fwd_solution(
        fns,
        src=vol_src,
        ignore_ref=True,
        bem=bem,
        eeg=eeg,
        meg=meg,
        verbose=verbose,
    )
    mne.write_forward_solution(fns.fwd_model, fwd, overwrite=True)

    print("Forward model complete.")


def _setup_volume_source_space(fns, gridstep=5, mindist=5.0, exclude=0.0):
    """Set up a volume source space grid inside the inner skull surface.

    This is a RHINO specific version of mne.setup_volume_source_space.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    gridstep : int, optional
        A grid will be constructed with the spacing given by ``gridstep`` in mm
        generating a volume source space.
    mindist : float, optional
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude : float, optional
        Exclude points closer than this distance (mm) from the center of mass of
        the bounding surface.

    Returns
    -------
    src : mne.SourceSpaces
        A single source space object.

    Notes
    -----
    This is a RHINO-specific version of mne.setup_volume_source_space,
    which can handle mri's that are niftii files.

    This specifically uses the inner skull surface in
    CoregFilenames.bet_inskull_surf_file to define the source space grid.

    This will also copy the CoregFilenames.bet_inskull_surf_file file to:
    `subjects_dir/subject/bem/inner_skull.surf` since this is where mne expects
    to find it when mne.make_bem_model is called.

    The coords of points to reconstruct to can be found in the output here:

    >>> src[0]['rr'][src[0]['vertno']]

    where they are in native MRI space in metres.
    """
    # Note that due to the unusual naming conventions used by BET and MNE:
    # - bet_inskull_*_file is actually the brain surface
    # - bet_outskull_*_file is actually the inner skull surface
    # - bet_outskin_*_file is the outer skin/scalp surface
    #
    # These correspond in mne to (in order):
    # - inner_skull
    # - outer_skull
    # - outer_skin
    #
    # This means that for single shell model, i.e. with conductivities set to
    # length one, the surface used by MNE will always be the inner_skull,
    # i.e. it actually corresponds to the brain/cortex surface!! Not sure that
    # is correct/optimal.
    #
    # Note that this is done in Fieldtrip too!, see the "Realistic single-shell
    # model, using brain surface from segmented mri" section at:
    # https://www.fieldtriptoolbox.org/example/make_leadfields_using_different_headmodels/#realistic-single-shell-model-using-brain-surface-from-segmented-mri
    #
    # However, others are clear that it should really be the actual inner
    # surface of the skull, see the "single-shell Boundary Element Model (BEM)"
    # bit at: https://imaging.mrc-cbu.cam.ac.uk/meg/SpmForwardModels

    # -------------------------------------------------------------------
    # Move the surfaces to where MNE expects to find them for the forward
    # modelling, see make_bem_model in mne/bem.py
    # -------------------------------------------------------------------

    # Note that the coreg surf files are in scaled MRI space
    verts, tris = mne.surface.read_surface(fns.coreg.bet_inskull_surf_file)
    tris = tris.astype(int)
    mne.surface.write_surface(
        f"{fns.bem_dir}/inner_skull.surf",
        verts,
        tris,
        file_format="freesurfer",
        overwrite=True,
    )
    print("Using bet_inskull_surf_file for single shell surface")

    verts, tris = mne.surface.read_surface(fns.coreg.bet_outskull_surf_file)
    tris = tris.astype(int)
    mne.surface.write_surface(
        f"{fns.bem_dir}/outer_skull.surf",
        verts,
        tris,
        file_format="freesurfer",
        overwrite=True,
    )

    verts, tris = mne.surface.read_surface(fns.coreg.bet_outskin_surf_file)
    tris = tris.astype(int)
    mne.surface.write_surface(
        f"{fns.bem_dir}/outer_skin.surf",
        verts,
        tris,
        file_format="freesurfer",
        overwrite=True,
    )

    # ------------------------------------------------
    # Setup main MNE call to _make_volume_source_space
    # ------------------------------------------------

    pos = float(int(gridstep))
    pos /= 1000.0  # convert pos to m from mm for MNE

    vol_info = _get_vol_info_from_nii(fns.coreg.mri_file)

    surface = f"{fns.bem_dir}/inner_skull.surf"
    surf = mne.surface.read_surface(surface, return_dict=True)[-1]
    surf = copy.deepcopy(surf)
    surf["rr"] *= 1e-3  # must be in metres for MNE call

    # -------------
    # Main MNE call
    # -------------

    sp = mne.source_space._source_space._make_volume_source_space(
        surf,
        pos,
        exclude,
        mindist,
        fns.coreg.mri_file,
        None,
        vol_info=vol_info,
        single_volume=False,
    )
    sp[0]["type"] = "vol"

    # ----------------------
    # Save and return result
    # ----------------------

    sp = mne.source_space._source_space._complete_vol_src(sp, fns.id)

    # Add dummy mri_ras_t and vox_mri_t transforms as these are needed
    # for the forward model to be saved (for some reason)
    sp[0]["mri_ras_t"] = Transform("mri", "ras")
    sp[0]["vox_mri_t"] = Transform("mri_voxel", "mri")

    if sp[0]["coord_frame"] != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError("source space is not in MRI coordinates")

    return sp


def _make_fwd_solution(
    fns,
    src,
    bem,
    meg=True,
    eeg=True,
    mindist=0.0,
    ignore_ref=False,
    verbose=None,
):
    """Calculate a forward solution for a subject.

    This is a wrapper for mne.make_forward_solution.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    src : instance of SourceSpaces
        Volumetric source space.
    bem : instance of ConductorModel
        BEM model.
    meg : bool, optional
        Include MEG computations?
    eeg : bool, optional
        Include EEG computations?
    mnidist : float, optional
        Minimum distance of sources from inner skull surface (in mm).
    ignore_ref : bool, optional
        If True, do not include reference channels in compensation.
        This option should be True for KIT files, since forward computation
        with reference channels is not currently supported.
    verbose : bool, optional
        Should we print info to the screen?

    Returns
    -------
    fwd : instance of Forward
        The forward solution.

    Notes
    -----
    Forward modelling is done in head space.

    The coords of points to reconstruct to can be found in the output here:

    >>> fwd['src'][0]['rr'][fwd['src'][0]['vertno']]

    where they are in head space in metres.

    The same coords of points to reconstruct to can be found in the input here:

    >>> src[0]['rr'][src[0]['vertno']]

    where they are in native MRI space in metres.
    """
    fns = fns.coreg

    # src should be in MRI space. Let's just check that is the case
    if src[0]["coord_frame"] != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError("src is not in MRI coordinates")

    # --------------------------------------------
    # Setup main MNE call to make_forward_solution
    # --------------------------------------------

    # The forward model is done in head space
    # We need the transformation from MRI to HEAD coordinates (or vice versa)
    head_scaledmri_trans_file = fns.head_scaledmri_t_file
    if isinstance(head_scaledmri_trans_file, str):
        head_mri_t = read_trans(head_scaledmri_trans_file)
    else:
        head_mri_t = head_scaledmri_trans_file

    # RHINO does everything in mm, so need to convert it to metres which is
    # what MNE expects. To change units on an xform, just need to change the
    # translation part and leave the rotation alone
    head_mri_t["trans"][0:3, -1] = head_mri_t["trans"][0:3, -1] / 1000

    # Get bem solution
    if isinstance(bem, str):
        bem = mne.read_bem_solution(bem)
    else:
        if not isinstance(bem, mne.bem.ConductorModel):
            raise TypeError("bem must be a string or ConductorModel")
        bem = bem.copy()

    for i in range(len(bem["surfs"])):
        bem["surfs"][i]["tris"] = bem["surfs"][i]["tris"].astype(int)

    # Load fif info
    info_fif_file = fns.info_fif_file
    info = mne.io.read_info(info_fif_file)

    # -------------
    # Main MNE call
    # -------------

    fwd = mne.make_forward_solution(
        info,
        trans=head_mri_t,
        src=src,
        bem=bem,
        eeg=eeg,
        meg=meg,
        mindist=mindist,
        ignore_ref=ignore_ref,
        verbose=verbose,
    )

    # fwd should be in Head space. Let's just check that is the case:
    if fwd["src"][0]["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError("fwd['src'][0] is not in HEAD coordinates")

    return fwd


def _get_orient(nii_file):
    """Get orientation of nii file."""
    cmd = f"fslorient -getorient {nii_file}"
    return os.popen(cmd).read().strip()


def _get_sform(nii_file):
    """Get sform of nii file."""
    sformcode = int(nib.load(nii_file).header["sform_code"])
    if sformcode == 1 or sformcode == 4:
        sform = nib.load(nii_file).header.get_sform()
    else:
        raise ValueError(
            f"sformcode for {nii_file} is {sformcode}, needs to be 1 or 4.\n\n"
            "The sform code indicates how the sform matrix should be "
            "interpreted:\n"
            "  1 = Scanner Anat (native scanner coordinates)\n"
            "  4 = MNI (MNI-152 standard space)\n\n"
            "How to fix this:\n\n"
            "1. If the qform is valid (check with "
            f"'fslorient -getqformcode {nii_file}'),\n"
            "   copy it to the sform:\n"
            f"     fslorient -copyqform2sform {nii_file}\n\n"
            "2. If the sform matrix is correct but only the code is wrong "
            "(check with\n"
            f"   'fslorient -getsform {nii_file}'), set the code directly:\n"
            f"     fslorient -setsformcode 1 {nii_file}\n\n"
            "3. If the orientation is non-standard, reorient to standard "
            "axes:\n"
            f"     fslreorient2std {nii_file} {nii_file}\n\n"
            "4. If both sform and qform are invalid, re-convert the "
            "original DICOMs\n"
            "   with dcm2niix, which correctly populates both.\n\n"
        )
    sform = mne.Transform("mri_voxel", "mri", sform)
    return sform


def _check_nii_for_nan(filename):
    """Check nii file for nans."""
    img = nib.load(filename)
    data = img.get_fdata()
    return np.isnan(data).any()


def _get_flirt_xform_between_axes(from_nii, target_nii):
    """
    Computes flirt xform that moves from_nii to have voxel indices on the same
    axis as  the voxel indices for target_nii.

    Note that this is NOT the same as registration, i.e. the images are not aligned.
    In fact the actual coordinates (in mm) are unchanged.

    It is instead about putting from_nii onto the same axes so that the voxel INDICES
    are comparable. This is achieved by using a transform that sets the sform of
    from_nii to be the same as target_nii without changing the actual coordinates
    (in mm). Transform needed to do this is:

      from2targetaxes = inv(targetvox2target) * fromvox2from

    In more detail, we need the sform for the transformed from_nii to be the same as
    the sform for the target_nii, without changing the actual coordinates (in mm).

    In other words, we need:

        fromvox2from * from_nii_vox = targetvox2target * from_nii_target_vox

    where
    - fromvox2from is sform for from_nii (i.e. converts from voxel indices to
      voxel coords in mm)
    - targetvox2target is sform for target_nii
    - from_nii_vox are the voxel indices for from_nii
    - from_nii_target_vox are the voxel indices for from_nii when transformed onto
      the target axis.

    => from_nii_target_vox = from2targetaxes * from_nii_vox

    where
    - from2targetaxes = inv(targetvox2target) * fromvox2from
    """

    to2tovox = np.linalg.inv(_get_sform(target_nii)["trans"])
    fromvox2from = _get_sform(from_nii)["trans"]
    from2to = to2tovox @ fromvox2from
    return from2to


def _get_mne_xform_from_flirt_xform(flirt_xform, nii_mesh_file_in, nii_mesh_file_out):
    """
    Returns a mm coordinates to mm coordinates MNE xform that corresponds to
    the passed in flirt xform.

    Note that we need to do this as flirt xforms include an extra xform based
    on the voxel dimensions (see get_flirtcoords2native_xform).
    """
    flirtcoords2native_xform_in = _get_flirtcoords2native_xform(nii_mesh_file_in)
    flirtcoords2native_xform_out = _get_flirtcoords2native_xform(nii_mesh_file_out)
    return (
        flirtcoords2native_xform_out
        @ flirt_xform
        @ np.linalg.inv(flirtcoords2native_xform_in)
    )


def _get_flirtcoords2native_xform(nii_mesh_file):
    """
    Returns xform_flirtcoords2native transform that transforms from flirtcoords
    space in mm into native space in mm, where the passed in nii_mesh_file specifies
    the native space

    Note that for some reason flirt outputs transforms of the form:
    flirt_mni2mri = mri2flirtcoords x mni2mri x flirtcoords2mni

    and bet_surf outputs the .vtk file vertex values in the same flirtcoords mm
    coordinate system.

    See the bet_surf manual:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#betsurf

    If the image has radiological ordering (see fslorient) then the mm coordinates
    are the voxel coordinates scaled by the mm voxel sizes.

    i.e. (x_mm = x_dim * x) where x_mm are the flirtcoords coords in mm, x is the
    voxel coordinate and x_dim is the voxel size in mm.
    """
    mri_orient = _get_orient(nii_mesh_file)
    if mri_orient != "RADIOLOGICAL":
        raise ValueError(
            "Orientation of file must be RADIOLOGICAL, please check output of: "
            f"fslorient -getorient {nii_mesh_file}"
        )
    xform_nativevox2native = _get_sform(nii_mesh_file)["trans"]
    dims = np.append(nib.load(nii_mesh_file).header.get_zooms(), 1)
    xform_flirtcoords2nativevox = np.diag(1.0 / dims)
    return xform_nativevox2native @ xform_flirtcoords2nativevox


def _transform_vtk_mesh(
    vtk_mesh_file_in,
    nii_mesh_file_in,
    out_vtk_file,
    nii_mesh_file_out,
    xform_file,
):
    """
    Outputs mesh to out_vtk_file, which is the result of applying the
    transform xform to vtk_mesh_file_in

    nii_mesh_file_in needs to be the corresponding niftii file from bet
    that corresponds to the same mesh as in vtk_mesh_file_in

    nii_mesh_file_out needs to be the corresponding niftii file from bet
    that corresponds to the same mesh as in out_vtk_file
    """
    rrs_in, tris_in = _get_vtk_mesh_native(vtk_mesh_file_in, nii_mesh_file_in)
    xform_flirtcoords2native_out = _get_flirtcoords2native_xform(nii_mesh_file_out)
    if isinstance(xform_file, str):
        xform = read_trans(xform_file)["trans"]
    else:
        xform = xform_file
    overall_xform = np.linalg.inv(xform_flirtcoords2native_out) @ xform
    rrs_out = _xform_points(overall_xform, rrs_in.T).T
    data = pd.read_csv(vtk_mesh_file_in, sep=r"\s+")
    num_rrs = int(data.iloc[3, 1])
    for col_idx in range(3):
        col = data.columns[col_idx]
        data[col] = data[col].astype(object)
        data.iloc[4 : num_rrs + 4, col_idx] = rrs_out[:, col_idx]
    data.to_csv(out_vtk_file, sep=" ", index=False)


def _get_vtk_mesh_native(vtk_mesh_file, nii_mesh_file):
    data = pd.read_csv(vtk_mesh_file, sep=r"\s+")
    num_rrs = int(data.iloc[3, 1])
    rrs_flirtcoords = data.iloc[4 : num_rrs + 4, 0:3].to_numpy().astype(np.float64)
    xform_flirtcoords2nii = _get_flirtcoords2native_xform(nii_mesh_file)
    rrs_nii = _xform_points(xform_flirtcoords2nii, rrs_flirtcoords.T).T
    num_tris = int(data.iloc[num_rrs + 4, 1])
    tris_nii = (
        data.iloc[num_rrs + 5 : num_rrs + 5 + num_tris, 1:4].to_numpy().astype(int)
    )
    return rrs_nii, tris_nii


def _xform_points(xform, pnts):
    """Applies homogeneous linear transformation to an array of 3D coordinates.

    Parameters
    ----------
    xform : numpy.ndarray
        4x4 matrix containing the affine transform.
    pnts : numpy.ndarray
        points to transform, should be 3 x num_points.

    Returns
    -------
    newpnts : numpy.ndarray
        pnts following the xform, will be 3 x num_points.
    """
    if len(pnts.shape) == 1:
        pnts = np.reshape(pnts, [-1, 1])
    num_rows, num_cols = pnts.shape
    if num_rows != 3:
        raise Exception(f"pnts is not 3xN, it is {num_rows}x{num_cols}")
    pnts = np.concatenate((pnts, np.ones([1, pnts.shape[1]])), axis=0)
    newpnts = xform @ pnts
    return newpnts[:3]


def _rigid_transform_3D(B, A, compute_scaling=False):
    """Calculate affine transform from points in A to point in B.

    Parameters
    ----------
    A : numpy.ndarray
        3 x num_points. Set of points to register from.
    B : numpy.ndarray
        3 x num_points. Set of points to register to.

    compute_scaling : bool
        Do we compute a scaling on top of rotation and translation?

    Returns
    -------
    xform : numpy.ndarray
        Calculated affine transform, does not include scaling.
    scaling_xform : numpy.ndarray
        Calculated scaling transform (a diagonal 4x4 matrix),
        does not include rotation or translation.

    see http://nghiaho.com/?page_id=671
    """
    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    scaling_xform = np.eye(4)
    if compute_scaling:
        RAm = R @ Am
        U2, S2, V2t = np.linalg.svd(Bm @ np.linalg.pinv(RAm))
        S2 = np.identity(3) * np.mean(S2[S2 > 1e-9])
        scaling_xform[0:3, 0:3] = S2
    t = -R @ centroid_A + centroid_B
    xform = np.eye(4)
    xform[0:3, 0:3] = R
    xform[0:3, -1] = np.reshape(t, (1, -1))
    return xform, scaling_xform


def _niimask2indexpointcloud(nii_fname, volindex=None):
    """Takes in a nii.gz mask file name (which equals zero for background
    and != zero for the mask) and returns the mask as a 3 x npoints point cloud.

    Parameters
    ----------
    nii_fname : string
        A nii.gz mask file name (with zero for background, and !=0 for the mask).
    volindex : int
        Volume index, used if nii_mask is a 4D file.

    Returns
    -------
    pc : numpy.ndarray
        3 x npoints point cloud as voxel indices.
    """
    vol = nib.load(nii_fname).get_fdata()
    if len(vol.shape) == 4 and volindex is not None:
        vol = vol[:, :, :, volindex]
    if not len(vol.shape) == 3:
        Exception(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume "
            "with volindex specifying a volume index"
        )
    return np.asarray(np.where(vol != 0))


def _rhino_icp(mri_headshape_head, polhemus_headshape_head, n_init=10):
    """Runs Iterative Closest Point (ICP) with multiple initialisations.

    Parameters
    ----------
    smri_headshape_polhemus : numpy.ndarray
        [3 x N] locations of the headshape points from MRI in HEAD space
    polhemus_headshape_polhemus : numpy.ndarray
        [3 x N] locations of the headshape points from polhemus in HEAD space.
    n_init : int
        Number of random initialisations to perform.

    Returns
    -------
    xform : numpy.ndarray
        [4 x 4] rigid transformation matrix mapping data2 to data.

    Notes
    -----
    Based on Matlab version from Adam Baker 2014.
    """
    data1 = mri_headshape_head
    data2 = polhemus_headshape_head
    err_old = np.inf
    err = np.zeros(n_init)
    Mr = np.eye(4)
    incremental = False
    if incremental:
        Mr_total = np.eye(4)
    data2r = data2
    for init in range(n_init):
        Mi, distances, i = _icp(data2r.T, data1.T)
        e = np.sqrt(np.mean(np.square(distances)))
        err[init] = e
        if err[init] < err_old:
            print(f"ICP found better xform, error={e}")
            err_old = e
            if incremental:
                Mr_total = Mr @ Mr_total
                xform = Mi @ Mr_total
            else:
                xform = Mi @ Mr
        a = (np.random.uniform() - 0.5) * np.pi / 6
        b = (np.random.uniform() - 0.5) * np.pi / 6
        c = (np.random.uniform() - 0.5) * np.pi / 6
        Rx = np.array(
            [(1, 0, 0), (0, np.cos(a), -np.sin(a)), (0, np.sin(a), np.cos(a))]
        )
        Ry = np.array(
            [(np.cos(b), 0, np.sin(b)), (0, 1, 0), (-np.sin(b), 0, np.cos(b))]
        )
        Rz = np.array(
            [(np.cos(c), -np.sin(c), 0), (np.sin(c), np.cos(c), 0), (0, 0, 1)]
        )
        T = 10 * np.array(
            [
                np.random.uniform() - 0.5,
                np.random.uniform() - 0.5,
                np.random.uniform() - 0.5,
            ]
        )
        Mr = np.eye(4)
        Mr[0:3, 0:3] = Rx @ Ry @ Rz
        Mr[0:3, -1] = np.reshape(T, (1, -1))
        if incremental:
            data2r = Mr @ Mr_total @ np.vstack((data2, np.ones((1, data2.shape[1]))))
        else:
            data2r = Mr @ np.vstack((data2, np.ones((1, data2.shape[1]))))
        data2r = data2r[0:3, :]
    return xform, err, err_old


def _icp(A, B, init_pose=None, max_iterations=50, tolerance=0.0001):
    """The Iterative Closest Point method:
    finds best-fit transform that maps points A on to points B.

    Parameters
    ----------
    A : numpy.ndarray
        Nxm numpy array of source mD points.
    B : numpy.ndarray
        Nxm numpy array of destination mD point.
    init_pose : numpy.ndarray
        (m+1)x(m+1) homogeneous transformation.
    max_iterations : int
        Exit algorithm after max_iterations.
    tolerance : float
        Convergence criteria.

    Returns
    -------
    T : numpy.ndarray
        (4 x 4) Final homogeneous transformation that maps A on to B.
    distances : numpy.ndarray
        Euclidean distances (errors) of the nearest neighbor.
    i : float
        Number of iterations to converge.

    Notes
    -----
    From: https://github.com/ClayFlannigan/icp/blob/master/icp.py
    """
    m = A.shape[1]
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0
    kdtree = KDTree(dst[:m, :].T)
    for i in range(max_iterations):
        distances, indices = kdtree.query(src[:m, :].T)
        T = _best_fit_transform(src[:m, :].T, dst[:m, indices].T)
        src = np.dot(T, src)
        mean_error = np.sqrt(np.mean(np.square(distances)))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    T = _best_fit_transform(A, src[:m, :].T)
    return T, distances, i


def _best_fit_transform(A, B):
    """Calculates the least-squares best-fit transform that maps corresponding
    points A to B in m spatial dimensions.

    Parameters
    ----------
    A : numpy.ndarray
        Nxm numpy array of corresponding points.
    B : numpy.ndarray
        Nxm numpy array of corresponding points.

    Outputs
    -------
    T : numpy.ndarray
        (m+1)x(m+1) homogeneous transformation matrix that maps A on to B.
    """
    assert A.shape == B.shape
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T


def _create_freesurfer_meshes_from_bet_surfaces(fns, xform_mri_voxel2mri):
    """
    Create sMRI-derived freesurfer surfaces in native/mri space in mm,
    for use by forward modelling
    """
    _create_freesurfer_mesh_from_bet_surface(
        infile=fns.bet_inskull_mesh_vtk_file,
        surf_outfile=fns.bet_inskull_surf_file,
        nii_mesh_file=fns.bet_inskull_mesh_file,
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )
    _create_freesurfer_mesh_from_bet_surface(
        infile=fns.bet_outskull_mesh_vtk_file,
        surf_outfile=fns.bet_outskull_surf_file,
        nii_mesh_file=fns.bet_outskull_mesh_file,
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )
    _create_freesurfer_mesh_from_bet_surface(
        infile=fns.bet_outskin_mesh_vtk_file,
        surf_outfile=fns.bet_outskin_surf_file,
        nii_mesh_file=fns.bet_outskin_mesh_file,
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )


def _create_freesurfer_mesh_from_bet_surface(
    infile,
    surf_outfile,
    xform_mri_voxel2mri,
    nii_mesh_file=None,
):
    """Creates surface mesh in .surf format and in native mri space in mm from infile.

    Parameters
    ----------
    infile : str
        Either:
        1) .nii.gz file containing zero's for background and one's for surface
        2) .vtk file generated by bet_surf (in which case the path to the
        structural MRI, smri_file, must be included as an input)
    surf_outfile : str
        Path to the .surf file generated, containing the surface mesh in mm
    xform_mri_voxel2mri : np.ndarray
        4x4 array. Transform from voxel indices to native/mri mm.
    nii_mesh_file : str, optional
        Path to the niftii mesh file that is the niftii equivalent of vtk file
        passed in as infile (only needed if infile is a vtk file).
    """
    pth, name = os.path.split(infile)
    name, ext = os.path.splitext(name)

    if ext == ".gz":
        print("Creating surface mesh for {} .....".format(infile))

        name, ext = os.path.splitext(name)
        if ext != ".nii":
            raise ValueError("Invalid infile. Needs to be a .nii.gz or .vtk file")

        # Load NIfTI and binarize
        nii = nib.load(infile)
        vol = nii.get_fdata()
        # Ensure binary mask (0 background, >0 surface)
        vol = (vol > 0).astype(np.uint8)

        # Run marching cubes
        # level=0.5 extracts the surface between 0 and 1
        # spacing left as (1,1,1) because we will apply a full 4x4 transform below.
        try:
            verts_vox, faces, normals, values = measure.marching_cubes(
                vol, level=0.5, spacing=(1.0, 1.0, 1.0)
            )
        except Exception as e:
            raise RuntimeError(
                "marching_cubes failed. Check that the NIfTI file is a "
                "proper binary mask."
            ) from e

        if verts_vox.size == 0 or faces.size == 0:
            raise RuntimeError(
                "marching_cubes produced no vertices/faces. Check input volume/mask."
            )

        # verts_vox is (M,3) in voxel coordinates (voxel index space)
        # Convert to homogeneous coordinates and apply the provided 4x4 transform
        ones = np.ones((verts_vox.shape[0], 1), dtype=verts_vox.dtype)
        verts_vox_h = np.hstack([verts_vox, ones])  # shape (M,4)

        # Ensure xform is numpy array and has shape (4,4)
        xform = np.asarray(xform_mri_voxel2mri)
        if xform.shape != (4, 4):
            raise ValueError("xform_mri_voxel2mri must be a 4x4 array")

        verts_mm_h = (xform @ verts_vox_h.T).T  # (M,4)
        verts_mm = verts_mm_h[:, :3]

        # faces already has shape (F,3) and is integer
        faces = faces.astype(int)

        # Write FreeSurfer surface
        mne.surface.write_surface(
            surf_outfile, verts_mm, faces, file_format="freesurfer", overwrite=True
        )

    elif ext == ".vtk":
        if nii_mesh_file is None:
            raise ValueError(
                "You must specify a nii_mesh_file (niftii format) "
                "if infile format is vtk"
            )

        rrs_native, tris_native = _get_vtk_mesh_native(infile, nii_mesh_file)

        mne.surface.write_surface(
            surf_outfile,
            rrs_native,
            tris_native,
            file_format="freesurfer",
            overwrite=True,
        )

    else:
        raise ValueError("Invalid infile. Needs to be a .nii.gz or .vtk file")


def _get_vol_info_from_nii(mri):
    """Read volume info from an MRI file.

    Parameters
    ----------
    mri : str
        Path to MRI file.

    Returns
    -------
    out : dict
        Dictionary with keys 'mri_width', 'mri_height', 'mri_depth'
        and 'mri_volume_name'.
    """
    dims = nib.load(mri).get_fdata().shape
    return dict(
        mri_width=dims[0],
        mri_height=dims[1],
        mri_depth=dims[2],
        mri_volume_name=mri,
    )


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def _majority(values_ptr, len_values, result, data):
    """
    def _majority(buffer, required_majority):
       return buffer.sum() >= required_majority

    See: https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/

    Numba cfunc that takes in:
    a double pointer pointing to the values within the footprint,
    a pointer-sized integer that specifies the number of values in the footprint,
    a double pointer for the result, and
    a void pointer, which could point to additional parameters
    """
    values = carray(values_ptr, (len_values,), dtype=float64)
    required_majority = 14  # in 3D we have 27 voxels in total
    result[0] = values.sum() >= required_majority
    return 1


def _binary_majority3d(img):
    """
    Set a pixel to 1 if a required majority (default=14) or more pixels
    in its 3x3x3 neighborhood are 1, otherwise, set the pixel to 0. img
    is a 3D binary image
    """
    if img.dtype != "bool":
        raise ValueError("binary_majority3d(img) requires img to be binary")
    if len(img.shape) != 3:
        raise ValueError("binary_majority3d(img) requires img to be 3D")
    return ndimage.generic_filter(
        img, LowLevelCallable(_majority.ctypes), size=3
    ).astype(int)


def _extract_headshape(preproc_file):
    """Extract headshape points and fiducials from a FIF file.

    Parameters
    ----------
    preproc_file : str
        Path to FIF file.

    Returns
    -------
    headshape_mm : numpy.ndarray
        3 x N array of headshape points in mm (HEAD space).
    nasion_mm : numpy.ndarray
        Nasion position in mm (HEAD space).
    rpa_mm : numpy.ndarray
        RPA position in mm (HEAD space).
    lpa_mm : numpy.ndarray
        LPA position in mm (HEAD space).
    """
    info = mne.io.read_info(preproc_file)

    headshape_m = []
    nasion_m = None
    rpa_m = None
    lpa_m = None

    for dig in info["dig"]:
        if dig["kind"] == FIFF.FIFFV_POINT_EXTRA:
            headshape_m.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            if dig["ident"] == FIFF.FIFFV_POINT_NASION:
                nasion_m = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_LPA:
                lpa_m = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_RPA:
                rpa_m = dig["r"]

    if not headshape_m:
        raise ValueError("No headshape points found in FIF file")
    if nasion_m is None or rpa_m is None or lpa_m is None:
        raise ValueError("Fiducials (nasion, LPA, RPA) not found in FIF file")

    headshape_mm = np.array(headshape_m).T * 1000
    nasion_mm = np.array(nasion_m) * 1000
    rpa_mm = np.array(rpa_m) * 1000
    lpa_mm = np.array(lpa_m) * 1000

    # Remove points below the ears (neck/body)
    z_min = min(lpa_mm[2], rpa_mm[2])
    keep = headshape_mm[2] >= z_min
    headshape_mm = headshape_mm[:, keep]

    return headshape_mm, nasion_mm, rpa_mm, lpa_mm
