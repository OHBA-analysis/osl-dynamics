import logging
import pathlib
import re
import subprocess

import nibabel as nib
from tqdm import trange
from vrad.analysis import std_masks
from vrad.analysis.scenes import state_scene

_logger = logging.getLogger("VRAD")

surfs = {
    0: [std_masks.surf_left, std_masks.surf_right],
    1: [std_masks.surf_left_inf, std_masks.surf_right_inf],
    2: [std_masks.surf_left_vinf, std_masks.surf_right_vinf],
}


def render(
    nii: str,
    save_dir: str = None,
    interptype: str = "trilinear",
    gui: bool = True,
    inflation: int = 0,
    image_name: str = None,
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
    """
    nii = pathlib.Path(nii)

    if not nii.exists() or ".nii" not in nii.suffixes:
        raise ValueError(f"nii should be a nii or nii.gz file." f"found {nii}.")

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

    if image_name:
        image(
            cifti_left=cifti_left,
            cifti_right=cifti_right,
            file_name=image_name,
            inflation=inflation,
        )

    if gui:
        visualise(cifti_left=cifti_left, cifti_right=cifti_right, inflation=inflation)


def visualise(cifti_left, cifti_right, inflation=0):
    surface = surfs.get(inflation, None)
    if surface is None:
        _logger.warning(
            f"Inflation of {inflation} is not a valid selection. Using '0' instead."
        )

    subprocess.run(["wb_view", *surface, cifti_left, cifti_right])


def image(cifti_left, cifti_right, file_name: str, inflation=0):
    file_path = pathlib.Path(file_name)
    suffix = file_path.suffix or ".png"
    file_path = file_path.with_suffix("")

    scene_file = state_scene
    temp_scene = pathlib.Path("temp_scene.scene")

    surf_left, surf_right = surfs.get(inflation, surfs[0])

    scene = scene_file.read_text()
    scene = re.sub("{left_series}", str(cifti_left), scene)
    scene = re.sub("{right_series}", str(cifti_right), scene)
    scene = re.sub("{parcellation_file_left}", surf_left, scene)
    scene = re.sub("{parcellation_file_right}", surf_right, scene)
    temp_scene.write_text(scene)

    n_states = nib.load(cifti_left).shape[0]
    max_int_length = len(str(n_states))

    pathlib.Path(file_path).parent.mkdir(exist_ok=True, parents=True)
    file_pattern = f"{file_path}{{:0{max_int_length}d}}{suffix}"

    for i in trange(n_states, desc="processing state"):
        subprocess.run(
            [
                "wb_command",
                "-show-scene",
                "temp_scene.scene",
                "ready",
                file_pattern.format(i),
                "0",
                "0",
                "-use-window-size",
                "-set-map-yoke",
                "I",
                f"{i + 1}",
            ],
        )

    temp_scene.unlink()


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
