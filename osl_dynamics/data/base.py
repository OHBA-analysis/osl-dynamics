"""Base class for handling data.

"""

import numpy as np
import yaml

from osl_dynamics.data.rw import RW
from osl_dynamics.data.processing import Processing
from osl_dynamics.data.tf import TensorFlowDataset
from osl_dynamics.utils import misc


class Data(RW, Processing, TensorFlowDataset):
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
        By default we read the field 'X'.
    sampling_frequency : float
        Sampling frequency of the data in Hz.
    store_dir : str
        Directory to save results and intermediate steps to. Default is /tmp.
    n_embeddings : int
        Number of embeddings. Can be passed if data has already been prepared.
    time_axis_first : bool
        Is the input data of shape (n_samples, n_channels)?
    load_memmaps: bool
        Should we load the data into the memmaps?
    keep_memmaps_on_close : bool
        Should we keep the memmaps?
    """

    def __init__(
        self,
        inputs,
        matlab_field="X",
        sampling_frequency=None,
        store_dir="tmp",
        n_embeddings=None,
        time_axis_first=True,
        load_memmaps=True,
        keep_memmaps_on_close=False,
    ):
        # Unique identifier for the Data object
        self._identifier = id(self)

        # Load data by initialising an RW object
        RW.__init__(
            self,
            inputs,
            matlab_field,
            sampling_frequency,
            store_dir,
            time_axis_first,
            load_memmaps=load_memmaps,
            keep_memmaps_on_close=keep_memmaps_on_close,
        )
        if hasattr(self, "n_embeddings"):
            n_embeddings = self.n_embeddings

        # Initialise a Processing object so we have method we can use to prepare
        # the data
        Processing.__init__(
            self, n_embeddings, keep_memmaps_on_close=keep_memmaps_on_close
        )

        # Initialise a TensorFlowDataset object so we have methods to create datasets
        TensorFlowDataset.__init__(self)

    def __iter__(self):
        return iter(self.subjects)

    def __getitem__(self, item):
        return self.subjects[item]

    def __str__(self):
        info = [
            f"{self.__class__.__name__}",
            f"id: {self._identifier}",
            f"n_subjects: {self.n_subjects}",
            f"n_samples: {self.n_samples}",
            f"n_channels: {self.n_channels}",
        ]
        return "\n ".join(info)

    @property
    def raw_data(self):
        """Return raw data as a list of arrays."""
        return self.raw_data_memmaps

    @property
    def n_channels(self):
        """Number of channels in the data files."""
        return self.subjects[0].shape[-1]

    @property
    def n_samples(self):
        """Number of samples for each subject."""
        return sum([subject.shape[-2] for subject in self.subjects])

    @property
    def n_subjects(self):
        """Number of subjects."""
        return len(self.subjects)

    def set_sampling_frequency(self, sampling_frequency):
        """Sets the sampling_frequency attribute.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency in Hz.
        """
        self.sampling_frequency = sampling_frequency

    def time_series(self, concatenate=False):
        """Time series data for all subjects.

        Parameters
        ----------
        concatenate : bool
            Should we return the time series for each subject concatenated?

        Returns
        -------
        ts : list or np.ndarray
            Time series data for each subject.
        """
        if concatenate or self.n_subjects == 1:
            return np.concatenate(self.subjects)
        else:
            return self.subjects

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
