import numpy as np
import yaml
from tqdm import tqdm
from vrad.analysis import spectral
from vrad.data import Data, manipulation
from vrad.utils.misc import MockArray


class PreprocessedData(Data):
    """Class for loading preprocessed data.

    Contains methods which can be used to prepare the data for training a model.
    This includes methods to perform time embedding and PCA.

    Parameters
    ----------
    inputs : list of str or str
        Filenames to be read.
    sampling_frequency : float
        Sampling frequency of the data in Hz. Optional, default is 1.0.
    store_dir : str
        Directory to save results and intermediate steps to. Optional, default
        is /tmp.
    prepared_data_file : str
        Filename to save memory map to. Optional.
    """

    def __init__(
        self,
        inputs: list,
        sampling_frequency: float = 1.0,
        store_dir: str = "tmp",
        prepared_data_file: str = None,
    ):
        super().__init__(inputs, sampling_frequency, store_dir)
        if prepared_data_file is None:
            self.prepared_data_file = f"dataset_{self._identifier}.npy"

    def prepare_memmap_filenames(self):
        self.te_pattern = "te_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(len(self.inputs))), identifier=self._identifier
        )
        self.prepared_data_pattern = (
            "prepared_data_"
            "{{i:0{width}d}}_"
            "{identifier}.npy".format(
                width=len(str(len(self.inputs))), identifier=self._identifier
            )
        )

        # Time embedded data memory maps
        self.te_memmaps = []
        self.te_filenames = [
            str(self.store_dir / self.te_pattern.format(i=i))
            for i, _ in enumerate(self.inputs)
        ]

        # Prepared data memory maps (time embedded and pca'ed)
        self.prepared_data_memmaps = []
        self.prepared_data_filenames = [
            str(self.store_dir / self.prepared_data_pattern.format(i=i))
            for i, _ in enumerate(self.inputs)
        ]
        self.prepared_data_mean = []
        self.prepared_data_std = []

    def prepare(
        self, n_embeddings: int, n_pca_components: int = None, whiten: bool = False,
    ):
        """Prepares data to train the model with.

        Performs standardization, time embedding and principle component analysis.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data.
        n_pca_components : int
            Number of PCA components to keep. Optional, default is no PCA.
        whiten : bool
            Should we whiten the PCA'ed data? Optional, default is False.
        """
        self.prepare_memmap_filenames()

        # Standardise and time embed the data for each subject
        for memmap, new_file in zip(
            tqdm(self.raw_data_memmaps, desc="Time embedding", ncols=98),
            self.te_filenames,
        ):
            memmap = manipulation.standardize(memmap, axis=0)
            te_shape = (
                memmap.shape[0] - (n_embeddings - 1),
                memmap.shape[1] * n_embeddings,
            )
            te_memmap = MockArray.get_memmap(new_file, te_shape, dtype=np.float32)
            te_memmap = manipulation.time_embed(
                memmap, n_embeddings, output_file=te_memmap
            )
            self.te_memmaps.append(te_memmap)

        # Perform principle component analysis (PCA)
        if n_pca_components is not None:

            print("Calculating PCA")
            covariance = np.zeros([te_memmap.shape[1], te_memmap.shape[1]])
            for te_memmap in self.te_memmaps:
                covariance += np.transpose(te_memmap - te_memmap.mean(axis=0)) @ (
                    te_memmap - te_memmap.mean(axis=0)
                )
            u, s, vh = np.linalg.svd(covariance)
            u = u[:, :n_pca_components].astype(np.float32)
            s = s[:n_pca_components].astype(np.float32)
            if whiten:
                u = u @ np.diag(1.0 / np.sqrt(s))
            self.pca_weights = u

            # Apply PCA to the data for each subject and standardise again
            for te_memmap, prepared_data_file in zip(
                tqdm(self.te_memmaps, desc="Applying PCA", ncols=98),
                self.prepared_data_filenames,
            ):
                pca_te_shape = (
                    te_memmap.shape[0],
                    n_pca_components,
                )
                pca_te_memmap = MockArray.get_memmap(
                    prepared_data_file, pca_te_shape, dtype=np.float32
                )
                pca_te_memmap = te_memmap @ self.pca_weights
                self.prepared_data_mean.append(np.mean(pca_te_memmap, axis=0))
                self.prepared_data_std.append(np.std(pca_te_memmap, axis=0))
                pca_te_memmap = manipulation.standardize(pca_te_memmap, axis=0)
                self.prepared_data_memmaps.append(pca_te_memmap)

        # Otherwise, the time embedded data is the prepared data
        else:
            for te_memmap, prepared_data_file in zip(
                self.te_memmaps, self.prepared_data_filenames,
            ):
                self.prepared_data_mean.append(np.mean(te_memmap, axis=0))
                self.prepared_data_std.append(np.std(te_memmap, axis=0))
                te_memmap = manipulation.standardize(te_memmap, axis=0)
                self.prepared_data_memmaps.append(te_memmap)

        # Update subjects to return the prepared data
        self.subjects = self.prepared_data_memmaps

        self.n_embeddings = n_embeddings
        self.n_pca_components = n_pca_components
        self.whiten = whiten
        self.prepared = True

    def trim_raw_time_series(
        self, n_embeddings: int = None, sequence_length: int = None
    ) -> np.ndarray:
        """Trims the raw preprocessed data time series.

        Removes the data points that are removed when the data is prepared,
        i.e. due to time embedding and separating into sequences, but does not
        perform time embedding or batching into sequences on the time series.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data.
        sequence_length : int
            Length of the segement of data to feed into the model.

        Returns
        -------
        np.ndarray
            Trimed time series.
        """
        trimmed_raw_time_series = []
        for memmap in self.raw_data_memmaps:
            if n_embeddings is not None:
                # Remove data points which are removed due to time embedding
                memmap = memmap[n_embeddings // 2 : -n_embeddings // 2]
            if sequence_length is not None:
                # Remove data points which are removed due to separating into sequences
                n_sequences = memmap.shape[0] // sequence_length
                memmap = memmap[: n_sequences * sequence_length]
            trimmed_raw_time_series.append(memmap)
        return trimmed_raw_time_series

    def autocorrelation_functions(self, covariances: np.ndarray) -> np.ndarray:
        """Calculates the autocorrelation function the state covariance matrices.

        An autocorrelation function is calculated for each state for each subject.

        Parameters
        ----------
        covariances : np.ndarray
            State covariance matrices.
            Shape is (n_subjects, n_states, n_channels, n_channels).
            These must be subject specific covariances.

        Returns
        -------
        np.ndarray
            Autocorrelation function.
            Shape is (n_subjects, n_states, n_channels, n_channels, n_acf).
        """
        n_subjects = len(covariances)
        n_states = covariances[0].shape[0]

        # Validation
        if self.n_embeddings <= 0:
            raise ValueError(
                "To calculate an autocorrelation function we have to train on time "
                + "embedded data."
            )

        autocorrelation_function = []
        for n in range(n_subjects):
            # Reverse the final standardisation
            for i in range(n_states):
                covariances[n][i] = (
                    np.diag(self.prepared_data_std[n])
                    @ covariances[n][i]
                    @ np.diag(self.prepared_data_std[n])
                )

            # Reverse the PCA
            te_cov = []
            for i in range(n_states):
                te_cov.append(self.pca_weights @ covariances[n][i] @ self.pca_weights.T)
            te_cov = np.array(te_cov)

            # Reverse the first standardisation
            for i in range(n_states):
                te_cov[i] = (
                    np.diag(np.repeat(self.raw_data_std[n], self.n_embeddings))
                    @ te_cov[i]
                    @ np.diag(np.repeat(self.raw_data_std[n], self.n_embeddings))
                )

            # Calculate the autocorrelation function
            autocorrelation_function.append(
                spectral.autocorrelation_function(
                    te_cov, self.n_embeddings, self.n_raw_data_channels
                )
            )

        return autocorrelation_function

    def _process_from_yaml(self, file, **kwargs):
        with open(file) as f:
            settings = yaml.load(f, Loader=yaml.Loader)

        prep_settings = settings.get("prepare", {})
        if prep_settings.get("do", False):
            self.prepare(
                n_embeddings=prep_settings.get("n_embeddings"),
                n_pca_components=prep_settings.get("n_pca_components", None),
                whiten=prep_settings.get("whiten", False),
            )
