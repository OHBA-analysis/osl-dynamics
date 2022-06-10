"""Classes related to MATLAB SPM.

"""

import numpy as np
from osl_dynamics.data import rw


class SPM:
    """Class to read SPM files.

    Parameters
    ----------
    filename : str
        Path to SPM file.
    load_data : bool
        Should we load the data?
    """

    def __init__(self, filename, load_data=True):
        D = rw.loadmat(filename)

        self.n_samples = D["Nsamples"]
        self.n_channels = len(D["channels"])
        self.channel_labels = [channel["label"] for channel in D["channels"]]
        self.channel_type = [channel["type"] for channel in D["channels"]]
        self.sampling_frequency = D["Fsample"]

        self.good_channels = self.get_good_channels(D["channels"])
        self.good_samples = self.get_good_samples(D["trials"]["events"])

        self.bad_channels = np.invert(self.good_channels)
        self.bad_channel_labels = [
            self.channel_labels[i] for i, bad in enumerate(self.bad_channels) if bad
        ]

        self.discontinuities = self.get_discontinuities()

        self.spm_filename = filename
        self.data_filename = D["data"]["fname"]

        if load_data:
            self.data = self.load_data_file()
            self.data = self.data[self.good_samples]

    def __str__(self):
        info = [
            f"{self.__class__.__name__}",
            f"From file: {self.spm_filename}",
            f"n_samples: {self.n_samples}",
            f"n_channels: {self.n_channels}",
        ]
        return "\n ".join(info)

    def get_discontinuities(self):

        discontinuities = [1]
        for i in range(self.n_samples - 1):
            if self.good_samples[i] != self.good_samples[i + 1]:
                discontinuities.append(1)
            else:
                discontinuities[-1] += 1

        if self.good_samples[0]:
            discontinuities = discontinuities[::2]
        else:
            discontinuities = discontinuities[1::2]

        return np.array(discontinuities)

    def get_good_channels(self, channels):
        return np.array([channel["bad"] == 0 for channel in channels])

    def get_good_samples(self, events):
        if isinstance(events, dict):
            events = [events]

        good_samples = np.ones(self.n_samples, dtype=bool)
        for event in events:
            if event["type"] == "artefact_OSL" and "MEG" in event["value"]:
                start = round(event["time"] * self.sampling_frequency) - 1
                duration = round(event["duration"] * self.sampling_frequency)
                end = start + duration
                good_samples[start:end] = False

        return good_samples

    def load_data_file(self):
        data = np.fromfile(self.data_filename, dtype=np.float32).reshape(
            self.n_samples, self.n_channels
        )
        return data
