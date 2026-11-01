"""Cluster finding and threshold-free cluster enhancement (TFCE).

Clusters are connected sets of targets whose statistic exceeds a threshold.
Target dimensions are of two kinds:

- **Cluster dimensions** are connected: neighbouring indices along a
  lattice (e.g. consecutive time points or frequencies), or a user-supplied
  adjacency (e.g. neighbouring parcels).
- All other dimensions are unconnected: a cluster never spans two indices
  of such a dimension (e.g. two states), but the maximum cluster statistic
  is still taken over them.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components

Adjacency = Optional[Dict[int, sparse.spmatrix]]


def lattice_adjacency(n: int) -> sparse.csr_matrix:
    """Adjacency of a 1D lattice: index i is adjacent to i-1 and i+1.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix
        Shape is (n, n).
    """
    i = np.arange(n - 1)
    return sparse.coo_matrix(
        (np.ones(2 * (n - 1)), (np.r_[i, i + 1], np.r_[i + 1, i])), shape=(n, n)
    ).tocsr()


def _validate_adjacency(matrix, n: int, dim: int) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(np.asarray(matrix))
    matrix = matrix.tocsr()
    if matrix.shape != (n, n):
        raise ValueError(
            f"adjacency for target dimension {dim} must have shape {(n, n)}, "
            f"got {matrix.shape}."
        )
    matrix = ((matrix + matrix.T) > 0).astype(float)
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix.tocsr()


class ClusterStructure:
    """How the targets are connected.

    Parameters
    ----------
    target_dims : tuple of int
        Shape of the statistic map.
    cluster_dims : int or sequence of int
        Target dimensions along which clusters can extend.
    adjacency : dict, optional
        Adjacency matrices for cluster dimensions that are not lattices,
        :code:`{dim: matrix}` with :code:`matrix` of shape
        :code:`(target_dims[dim], target_dims[dim])` (dense or
        :mod:`scipy.sparse`, e.g. from :func:`mne.channels.find_ch_adjacency`
        or :func:`mne.spatial_src_adjacency`). Cluster dimensions not in
        this dictionary use a lattice adjacency.
    """

    def __init__(
        self,
        target_dims: Sequence[int],
        cluster_dims: Union[int, Sequence[int]],
        adjacency: Adjacency = None,
    ) -> None:
        self.target_dims = tuple(int(d) for d in target_dims)
        n_dims = len(self.target_dims)

        if cluster_dims is None:
            raise ValueError("cluster_dims must be given (e.g. 0 for time).")
        dims = np.atleast_1d(cluster_dims).astype(int)
        dims = tuple(
            sorted(
                set(int(d) % n_dims if -n_dims <= d < n_dims else int(d) for d in dims)
            )
        )
        for d in dims:
            if not 0 <= d < n_dims:
                raise ValueError(
                    f"cluster_dims {cluster_dims} out of range for data with "
                    f"{n_dims} target dimension(s)."
                )
        if len(dims) == 0:
            raise ValueError("At least one cluster dimension is needed.")
        self.cluster_dims = dims

        adjacency = {} if adjacency is None else dict(adjacency)
        self.adjacency = {}
        for dim, matrix in adjacency.items():
            dim = int(dim) % n_dims
            if dim not in self.cluster_dims:
                raise ValueError(
                    f"adjacency given for dimension {dim}, which is not in "
                    f"cluster_dims {self.cluster_dims}."
                )
            self.adjacency[dim] = _validate_adjacency(
                matrix, self.target_dims[dim], dim
            )

        if self.adjacency:
            self._graph = self._build_graph()
        else:
            # Pure lattice: scipy.ndimage.label with a structure that only
            # connects neighbours along the cluster dimensions
            structure = np.zeros((3,) * n_dims, dtype=bool)
            centre = (1,) * n_dims
            structure[centre] = True
            for d in self.cluster_dims:
                for offset in (0, 2):
                    index = list(centre)
                    index[d] = offset
                    structure[tuple(index)] = True
            self._structure = structure
            self._graph = None

    @property
    def is_lattice(self) -> bool:
        return self._graph is None

    def _build_graph(self) -> sparse.csr_matrix:
        """Adjacency over the flattened targets: the sum over cluster
        dimensions of the Kronecker product of that dimension's adjacency
        with identities for every other dimension."""
        graph = None
        for d in self.cluster_dims:
            n = self.target_dims[d]
            A = self.adjacency.get(d, lattice_adjacency(n))
            term = sparse.identity(1, format="csr")
            for k, size in enumerate(self.target_dims):
                factor = A if k == d else sparse.identity(size, format="csr")
                term = sparse.kron(term, factor, format="csr")
            graph = term if graph is None else graph + term
        return graph.tocsr()

    def label(self, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Label the connected components of a boolean mask.

        Parameters
        ----------
        mask : np.ndarray
            Boolean array. Shape is (\\*target_dims).

        Returns
        -------
        labels : np.ndarray
            Integer labels, 0 outside the mask, 1..n_clusters inside.
        n_clusters : int
            Number of clusters.
        """
        if self.is_lattice:
            labels, n_clusters = ndimage.label(mask, structure=self._structure)
            return labels, n_clusters
        flat = mask.reshape(-1)
        indices = np.where(flat)[0]
        labels = np.zeros(flat.shape, dtype=np.int32)
        if len(indices) == 0:
            return labels.reshape(mask.shape), 0
        sub = self._graph[indices][:, indices]
        n_clusters, component = connected_components(sub, directed=False)
        labels[indices] = component + 1
        return labels.reshape(mask.shape), n_clusters


def find_clusters(
    stat: np.ndarray,
    threshold: float,
    structure: ClusterStructure,
    tail: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find clusters of supra-threshold targets.

    Parameters
    ----------
    stat : np.ndarray
        Statistic map. Shape is (\\*target_dims).
    threshold : float
        Cluster-forming threshold (positive).
    structure : ClusterStructure
        How the targets are connected.
    tail : int, optional
        :code:`1` clusters :code:`stat > threshold`, :code:`-1` clusters
        :code:`stat < -threshold`, :code:`0` does both, keeping positive and
        negative clusters separate.

    Returns
    -------
    labels : np.ndarray
        Cluster label of each target, 0 outside clusters. Shape is
        (\\*target_dims).
    cluster_stats : np.ndarray
        Sum of the statistic in each cluster (signed). Shape is
        (n_clusters,). Cluster :code:`i` has label :code:`i + 1`.
    """
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.zeros(stat.shape, dtype=np.int32)
    cluster_stats = []
    signs = {1: (1,), -1: (-1,), 0: (1, -1)}[tail]
    for sign in signs:
        mask = sign * stat > threshold
        lab, n = structure.label(mask)
        if n == 0:
            continue
        sums = ndimage.sum_labels(stat, lab, index=np.arange(1, n + 1))
        labels[lab > 0] = lab[lab > 0] + len(cluster_stats)
        cluster_stats.extend(sums)
    return labels, np.array(cluster_stats, dtype=float)


def max_cluster_stat(
    stat: np.ndarray, threshold: float, structure: ClusterStructure, tail: int = 0
) -> float:
    """Largest absolute cluster statistic in a map (0 if no clusters)."""
    _, cluster_stats = find_clusters(stat, threshold, structure, tail)
    return float(np.abs(cluster_stats).max()) if len(cluster_stats) else 0.0


def cluster_stat_map(
    stat: np.ndarray, threshold: float, structure: ClusterStructure, tail: int = 0
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Map each target to the absolute statistic of its cluster.

    Returns
    -------
    stat_map : np.ndarray
        :code:`|cluster statistic|` of the cluster each target belongs to,
        0 outside clusters. Shape is (\\*target_dims).
    clusters : list of np.ndarray
        Boolean mask of each cluster, largest absolute statistic first.
    cluster_stats : np.ndarray
        Signed statistic of each cluster, in the same order.
    """
    labels, cluster_stats = find_clusters(stat, threshold, structure, tail)
    order = np.argsort(-np.abs(cluster_stats))
    clusters = [labels == i + 1 for i in order]
    cluster_stats = cluster_stats[order]
    stat_map = np.zeros(stat.shape)
    for mask, value in zip(clusters, cluster_stats):
        stat_map[mask] = abs(value)
    return stat_map, clusters, cluster_stats


def tfce(
    stat: np.ndarray,
    structure: ClusterStructure,
    tail: int = 0,
    start: float = 0.0,
    step: float = 0.2,
    E: float = 0.5,
    H: float = 2.0,
) -> np.ndarray:
    """Threshold-free cluster enhancement [1]_.

    Each target's statistic is replaced by the integral over thresholds
    :code:`h` of :code:`extent(h) ** E * h ** H`, where :code:`extent(h)` is
    the size of the cluster containing the target at threshold :code:`h`.

    Parameters
    ----------
    stat : np.ndarray
        Statistic map. Shape is (\\*target_dims).
    structure : ClusterStructure
        How the targets are connected.
    tail : int, optional
        :code:`1` enhances positive values, :code:`-1` negative values and
        :code:`0` both (the output is signed).
    start : float, optional
        First threshold.
    step : float, optional
        Spacing of the thresholds.
    E : float, optional
        Extent exponent.
    H : float, optional
        Height exponent.

    Returns
    -------
    enhanced : np.ndarray
        TFCE map. Shape is (\\*target_dims).

    References
    ----------
    .. [1] Smith, S. M. & Nichols, T. E. (2009). Threshold-free cluster
           enhancement. NeuroImage, 44(1), 83-98.
    """
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)
    enhanced = np.zeros(stat.shape)
    signs = {1: (1,), -1: (-1,), 0: (1, -1)}[tail]
    for sign in signs:
        signed = sign * stat
        top = signed.max()
        if not np.isfinite(top) or top <= start:
            continue
        thresholds = np.arange(start + step, top + step, step)
        for h in thresholds:
            labels, n = structure.label(signed > h)
            if n == 0:
                break
            extent = np.bincount(labels.reshape(-1))[labels]
            inside = labels > 0
            enhanced[inside] += sign * extent[inside] ** E * h**H * step
    return enhanced


def max_tfce(
    stat: np.ndarray, structure: ClusterStructure, tail: int = 0, **kwargs
) -> float:
    """Largest absolute TFCE value in a map."""
    return float(np.abs(tfce(stat, structure, tail, **kwargs)).max())
