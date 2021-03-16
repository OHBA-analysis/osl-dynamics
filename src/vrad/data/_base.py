import logging
from typing import List, Union

import numpy as np
import yaml
from sklearn.cluster import KMeans

from vrad.data.analysis import Analysis
from vrad.data.io import IO
from vrad.data.manipulation import Manipulation
from vrad.data.tf import TensorFlowDataset
from vrad.utils import misc

_rng = np.random.default_rng()


class Data(IO, Manipulation, TensorFlowDataset, Analysis):
    """Data Class.

    The Data class enables the input and processing of data. When given a list of
    files, it produces a set of numpy memory maps which contain their raw data.
    It also provides methods for batching data and creating TensorFlow Datasets.

    Parameters
    ----------
    inputs : list of str or str
        Filenames to be read.
    matlab_field : str
        If a MATLAB filename is passed, this is the field that corresponds to the data.
        Optional. By default we read the field 'X'.
    sampling_frequency : float
        Sampling frequency of the data in Hz. Optional.
    store_dir : str
        Directory to save results and intermediate steps to. Optional, default is /tmp.
    epoched : bool
        Flag indicating if the data has been epoched. Optional, default is False.
        If True, inputs must be a list of lists.
    n_embeddings : int
        Number of embeddings. Optional. Can be passed if data has already been prepared.
    n_pca_components : int
        Number of PCA components. Optional. Can be passed if data has already been
        prepared.
    whiten : bool
        Was whitening applied during the PCA? Optional.
    prepared : bool
        Flag indicating if data has already been prepared. Optional.
    """

    def __init__(
        self,
        inputs: list,
        matlab_field: str = "X",
        sampling_frequency: float = None,
        store_dir: str = "tmp",
        epoched: bool = False,
        n_embeddings: int = 0,
        n_pca_components: int = None,
        whiten: bool = None,
        prepared: bool = False,
    ):
        # Unique identifier for the data object
        self._identifier = id(inputs)

        # Load data by initialising an IO object
        # This assigns self.raw_data_memmaps as well as other attributes
        IO.__init__(self, inputs, matlab_field, sampling_frequency, store_dir, epoched)

        # Use raw data for the subject data
        self.subjects = self.raw_data_memmaps

        # Initialise a Manipulation object so we can prepare the data
        Manipulation.__init__(self, n_embeddings, n_pca_components, whiten, prepared)

        # Initialise a TensorFlowDataset object so we have methods to create
        # training/prediction datasets
        TensorFlowDataset.__init__(self)

        # Initialise an Analysis object so we have methods to analyse the training
        # data after fitting a model
        Analysis.__init__(self)

    def __iter__(self):
        return iter(self.subjects)

    def __getitem__(self, item):
        return self.subjects[item]

    def __str__(self):
        info = [
            f"{self.__class__.__name__}",
            f"id: {self._identifier}",
            f"n_subjects: {self.n_subjects}",
            f"n_epochs: {self.n_epochs}",
            f"n_samples: {self.n_samples}",
            f"n_channels: {self.n_channels}",
            f"prepared: {self.prepared}",
        ]
        return "\n ".join(info)

    @property
    def raw_data(self) -> List:
        """Return raw data as a list of arrays."""
        return self.raw_data_memmaps

    @property
    def n_channels(self) -> int:
        """Number of channels in the data files."""
        return self.subjects[0].shape[-1]

    @property
    def n_epochs(self) -> Union[list, int]:
        """Number of epochs."""
        if self.epoched:
            n_epochs = [subject.shape[0] for subject in self.subjects]
            if len(set(n_epochs)) == 1:
                return n_epochs[0]
            else:
                return n_epochs

    @property
    def n_samples(self) -> int:
        """Number of samples for each subject."""
        if self.epoched:
            return self.subjects[0].shape[-2]
        else:
            return sum([subject.shape[-2] for subject in self.subjects])

    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return len(self.subjects)

    @classmethod
    def from_yaml(cls, file, **kwargs):
        instance = misc.class_from_yaml(cls, file, kwargs)

        with open(file) as f:
            settings = yaml.load(f, Loader=yaml.Loader)

        if issubclass(cls, Data):
            try:
                cls._process_from_yaml(instance, file, **kwargs)
            except AttributeError:
                pass

        training_dataset = instance.training_dataset(
            sequence_length=settings["sequence_length"],
            batch_size=settings["batch_size"],
        )
        prediction_dataset = instance.prediction_dataset(
            sequence_length=settings["sequence_length"],
            batch_size=settings["batch_size"],
        )

        return {
            "data": instance,
            "training_dataset": training_dataset,
            "prediction_dataset": prediction_dataset,
        }

    def covariance_sample(
        self,
        segment_length: Union[int, List[int]],
        n_segments: Union[int, List[int]],
        n_clusters: int = None,
    ) -> np.ndarray:
        """Get covariances of a random selection of a time series.

        Given a time series, `data`, randomly select a set of samples of length(s)
        `segment_length` with `n_segments` of each selected. If `n_clusters` is not
        specified each of these covariances will be returned. Otherwise, a K-means
        clustering algorithm is run to return that `n_clusters` covariances.

        Lack of overlap between samples is *not* guaranteed.

        Parameters
        ----------
        data: np.ndarray
            The time series to be analyzed.
        segment_length: int or list of int
            Either the integer number of samples for each covariance, or a list with a
            range of values.
        n_segments: int or list of int
            Either the integer number of segments to select,
             or a list specifying the number
            of each segment length to be sampled.
        n_clusters: int
            The number of K-means clusters to find
            (default is not to perform clustering).

        Returns
        -------
        covariances: np.ndarray
            The calculated covariance matrices of the samples.
        """
        segment_lengths = misc.listify(segment_length)
        n_segments = misc.listify(n_segments)

        if len(n_segments) == 1:
            n_segments = n_segments * len(segment_lengths)

        if len(segment_lengths) != len(n_segments):
            raise ValueError(
                "`segment_lengths` and `n_samples` should have the same lengths."
            )

        covariances = []
        for segment_length, n_sample in zip(segment_lengths, n_segments):
            data = self.subjects[_rng.choice(self.n_subjects)]
            starts = _rng.choice(data.shape[0] - segment_length, n_sample)
            samples = data[np.asarray(starts)[:, None] + np.arange(segment_length)]

            transposed = samples.transpose(0, 2, 1)
            m1 = transposed - transposed.sum(2, keepdims=1) / segment_length
            covariances.append(np.einsum("ijk,ilk->ijl", m1, m1) / (segment_length - 1))
        covariances = np.concatenate(covariances)

        if n_clusters is None:
            return covariances

        flat_covariances = covariances.reshape((covariances.shape[0], -1))

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flat_covariances)

        kmeans_covariances = kmeans.cluster_centers_.reshape(
            (n_clusters, *covariances.shape[1:])
        )

        return kmeans_covariances
