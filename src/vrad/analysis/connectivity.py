import numpy as np
from visbrain.objects import BrainObj, ConnectObj, SceneObj
from vrad.analysis import std_masks
from vrad.utils.parcellation import Parcellation


def exclude_by_sigma(edges, sigma=1):
    edges = edges.copy()
    np.fill_diagonal(edges, np.nan)

    mean = edges[~np.isnan(edges)].mean()
    std = edges[~np.isnan(edges)].std()

    np.fill_diagonal(edges, mean)
    selection = (edges >= (mean + sigma * std)) & (edges <= (mean - sigma * std))
    return selection


def plot_connectivity(edges, parcellation, *, inflation=0, selection=None):
    if isinstance(parcellation, str):
        parcellation = Parcellation(parcellation)
    nodes = parcellation.roi_centers()

    scene = SceneObj()

    c = ConnectObj(
        "Connect", nodes, edges, select=selection, cmap="inferno", antialias=True
    )

    points, triangles = std_masks.get_surf(inflation)
    b = BrainObj("Custom", vertices=points, faces=triangles)

    scene.add_to_subplot(c)
    scene.add_to_subplot(b)

    scene.preview()
