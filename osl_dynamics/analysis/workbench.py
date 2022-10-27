"""Functions to use Connectome Workbench.

"""

import os
import pathlib
import re
import subprocess
import warnings

import nibabel as nib
from tqdm import trange

from osl_dynamics import files

surfs = {
    0: [files.mask.surf_left, files.mask.surf_right],
    1: [files.mask.surf_left_inf, files.mask.surf_right_inf],
    2: [files.mask.surf_left_vinf, files.mask.surf_right_vinf],
}


def setup(path):
    """Sets up workbench.

    Adds workbench to the PATH environmental variable.

    Parameters
    ----------
    path : str
        Path to workbench installation.
    """
    # Check if workbench is already in PATH and if it's not add it
    if path not in os.environ["PATH"]:
        os.environ["PATH"] = f"{path}:{os.environ['PATH']}"


def render(
    nii,
    save_dir=None,
    interptype="trilinear",
    gui=True,
    inflation=0,
    image_name=None,
):
    """Render map in workbench.

    Parameters
    ----------
    nii : str
        Path to nii image file.
    save_dir : str
        Path to save rendered surface plots.
    interptype : str
        Interpolation type. Default is 'trilinear'.
    gui : bool
        Should we display the rendered plots in workbench? Default is True.
    image_name : str
        Filename of image to save.
    """
    nii = pathlib.Path(nii)

    if ".nii" not in nii.suffixes:
        raise ValueError(f"nii should be a nii or nii.gz file, got {nii}.")

    if not nii.exists():
        raise FileNotFoundError(nii)

    if save_dir is None:
        save_dir = pathlib.Path.cwd()
    else:
        save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_file = save_dir / nii.stem
    surf_left, surf_right = surfs.get(inflation, surfs[0])

    stem_right = out_file.with_name(out_file.stem + "_right")
    stem_left = out_file.with_name(out_file.stem + "_left")

    output_right = stem_right.with_suffix(".func.gii")
    output_left = stem_left.with_suffix(".func.gii")

    volume_to_surface(nii, surf=surf_right, output=output_right, interptype=interptype)

    volume_to_surface(nii, surf=surf_left, output=output_left, interptype=interptype)

    cifti_right = stem_right.with_suffix(".dtseries.nii")
    cifti_left = stem_left.with_suffix(".dtseries.nii")

    dense_timeseries(cifti=cifti_right, output=output_right, left_or_right="right")
    dense_timeseries(cifti=cifti_left, output=output_left, left_or_right="left")

    temp_scene = str(save_dir) + "/temp_scene.scene"

    if image_name:
        image(
            cifti_left=cifti_left,
            cifti_right=cifti_right,
            file_name=image_name,
            inflation=inflation,
            temp_scene=temp_scene,
        )

    if gui:
        visualise(
            cifti_left=cifti_left,
            cifti_right=cifti_right,
            inflation=inflation,
            temp_scene=temp_scene,
        )


def create_scene(cifti_left, cifti_right, inflation, temp_scene):
    scene_file = files.scene.mode_scene
    temp_scene = pathlib.Path(temp_scene)

    surf_left, surf_right = surfs.get(inflation, surfs[0])

    scene = scene_file.read_text()
    scene = re.sub("{left_series}", str(cifti_left.name), scene)
    scene = re.sub("{right_series}", str(cifti_right.name), scene)
    scene = re.sub("{parcellation_file_left}", surf_left, scene)
    scene = re.sub("{parcellation_file_right}", surf_right, scene)
    temp_scene.write_text(scene)


def visualise(cifti_left, cifti_right, inflation=0, temp_scene=None):
    surface = surfs.get(inflation, None)
    if surface is None:
        warnings.warn(
            f"Inflation of {inflation} is not a valid selection. Using '0' instead.",
            RuntimeWarning,
        )

    if temp_scene is None:
        temp_scene = "temp_scene.scene"
    create_scene(cifti_left, cifti_right, inflation, temp_scene)

    subprocess.run(
        [
            "wb_view",
            "-scene-load",
            temp_scene,
            "ready",
            *surface,
            cifti_left,
            cifti_right,
        ]
    )

    pathlib.Path(temp_scene).unlink()


def image(cifti_left, cifti_right, file_name, inflation=0, temp_scene=None):
    file_path = pathlib.Path(file_name)
    suffix = file_path.suffix or ".png"
    file_path = file_path.with_suffix("")

    if temp_scene is None:
        temp_scene = "temp_scene.scene"
    create_scene(cifti_left, cifti_right, inflation, temp_scene)

    n_modes = nib.load(cifti_left).shape[0]
    max_int_length = len(str(n_modes))

    pathlib.Path(file_path).parent.mkdir(exist_ok=True, parents=True)
    file_pattern = f"{file_path}{{:0{max_int_length}d}}{suffix}"

    for i in trange(n_modes, desc="Saving images", ncols=98):
        subprocess.run(
            [
                "wb_command",
                "-show-scene",
                temp_scene,
                "ready",
                file_pattern.format(i),
                "0",
                "0",
                "-use-window-size",
                "-set-map-yoke",
                "I",
                f"{i + 1}",
            ],
            capture_output=True,
        )

    pathlib.Path(temp_scene).unlink()


def volume_to_surface(nii, surf, output, interptype="trilinear"):
    subprocess.run(
        [
            "wb_command",
            "-volume-to-surface-mapping",
            str(nii),
            str(surf),
            str(output),
            f"-{interptype}",
        ]
    )


def dense_timeseries(cifti, output, left_or_right):
    subprocess.run(
        [
            "wb_command",
            "-cifti-create-dense-timeseries",
            cifti,
            f"-{left_or_right}-metric",
            output,
        ]
    )
