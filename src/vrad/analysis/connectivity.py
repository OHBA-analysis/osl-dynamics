import numpy as np
from nilearn.plotting import plot_connectome
from vrad.files import mask
from vrad.utils.parcellation import Parcellation


def exclude_by_sigma(edges, sigma=1):
    edges = edges.copy()
    np.fill_diagonal(edges, np.nan)

    mean = edges[~np.isnan(edges)].mean()
    std = edges[~np.isnan(edges)].std()

    np.fill_diagonal(edges, mean)
    selection = (edges >= (mean + sigma * std)) & (edges <= (mean - sigma * std))
    return selection


def std_filter(arr, sigma=0.95):
    copy = arr.copy()

    mean = copy.mean()
    std = copy.std()

    low_pass = mean - sigma * std
    high_pass = mean + sigma * std

    copy[(copy > low_pass) & (copy < high_pass)] = 0
    return copy


def make_symmetric(arr, upper=True):
    copy = arr.copy()
    index = np.tril_indices(copy.shape[-1], -1)
    copy[..., index[0], index[1]] = np.inf
    return np.minimum(copy, np.swapaxes(copy, -2, -1))


def plot_connectivity(edges, parcellation, sigma=0, **kwargs):
    filtered = std_filter(edges, sigma)
    if not isinstance(parcellation, Parcellation):
        parcellation = Parcellation(parcellation)
    plot_connectome(filtered, parcellation.roi_centers(), **kwargs)


def plot_connectivity_many(states, parcellation, sigma=0, zero_center=True, **kwargs):
    states = np.array(states)
    vmin = states.min()
    vmax = states.max()

    true_max = np.abs(states).max()
    if zero_center:
        vmin = -true_max
        vmax = true_max

    for state in states:
        plot_connectivity(
            state,
            parcellation,
            sigma,
            edge_vmin=vmin,
            edge_vmax=vmax,
            **kwargs,
        )
