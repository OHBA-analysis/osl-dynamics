"""Functions to calculate/plot connectivity.

"""

from pathlib import Path

import numpy as np
from nilearn import plotting
from tqdm import trange
from vrad.utils.parcellation import Parcellation


def save(
    connectivity_map: np.ndarray,
    threshold: float,
    filename: str,
    parcellation_file: str,
    **plot_kwargs,
):
    """Save connectivity maps.

    Parameters
    ----------
    connectivity_map : np.ndarray
        Matrices containing connectivity strengths to plot.
        Shape must be (n_states, n_channels, n_channels).
    threshold : float
        Threshold to determine which connectivity to show.
        Should be between 0 and 1.
    filename : str
        Output filename.
    parcellation_file : str
        Name of parcellation file used.
    """
    if threshold > 1 or threshold < 0:
        raise ValueError("threshold must be between 0 and 1.")

    parcellation = Parcellation(parcellation_file)
    n_states = len(connectivity_map)
    for i in trange(n_states, desc="Saving images", ncols=98):
        c = connectivity_map[i].copy()
        np.fill_diagonal(c, 0)
        output_file = "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
            fn=Path(filename), i=i, w=len(str(n_states))
        )
        plotting.plot_connectome(
            c,
            parcellation.roi_centers(),
            colorbar=True,
            edge_threshold=f"{threshold * 100}%",
            output_file=output_file,
            **plot_kwargs,
        )
