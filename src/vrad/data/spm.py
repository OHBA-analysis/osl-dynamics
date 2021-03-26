import numpy as np
from vrad.data import io


class SPM:
    """Class to read SPM files.

    Parameters
    ----------
    filename : str
        Path to SPM file.
    """

    def __init__(self, filename: str):
        D = io.loadmat(filename)

        self.n_samples = D["Nsamples"]
        self.n_channels = len(D["channels"])
        self.channel_label = [channel["label"] for channel in D["channels"]]
        self.channel_type = [channel["type"] for channel in D["channels"]]
        self.sampling_frequency = D["Fsample"]

        self.good_channels = self.get_good_channels(D["channels"])
        self.good_samples = self.get_good_samples(D["trials"]["events"])

        self.discontinuities = self.get_discontinuities(D["trials"]["events"])

        self.spm_filename = filename
        self.data_filename = D["data"]["fname"]
        self.data = self.load_data()

        # Only keep good samples
        self.data = self.data[self.good_samples]

    def __str__(self):
        info = [
            f"{self.__class__.__name__}",
            f"From file: {self.spm_filename}",
            f"n_samples: {self.n_samples}",
            f"n_channels: {self.n_channels}",
            f"data.shape: {self.data.shape}",
        ]
        return "\n ".join(info)

    def get_discontinuities(self, events: list) -> np.ndarray:
        discontinuities = []
        for event in events:
            if event["type"] == "artefact_OSL" and event["value"] == "MEGGRAD":
                start = round(event["time"] * self.sampling_frequency) - 1
                duration = round(event["duration"] * self.sampling_frequency)
                end = start + duration
                discontinuities.append(start)
                discontinuities.append(end)
        if discontinuities[-1] != self.n_samples:
            discontinuities.append(self.n_samples)
        return np.diff(discontinuities)[1::2]

    def get_good_channels(self, channels: list) -> np.ndarray:
        return np.array([channel["bad"] == 0 for channel in channels])

    def get_good_samples(self, events: list) -> np.ndarray:
        good_samples = np.ones(self.n_samples, dtype=bool)
        for event in events:
            if event["type"] == "artefact_OSL" and event["value"] == "MEGGRAD":
                start = round(event["time"] * self.sampling_frequency) - 1
                duration = round(event["duration"] * self.sampling_frequency)
                end = start + duration
                good_samples[start:end] = False
        return good_samples

    def load_data(self) -> np.ndarray:
        data = np.fromfile(self.data_filename, dtype=np.float32).reshape(
            self.n_samples, self.n_channels
        )
        return data
