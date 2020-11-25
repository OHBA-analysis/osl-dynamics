import numpy as np
from tqdm import tqdm
from vrad.data import manipulation
from vrad.data._base import Data
from vrad.utils.misc import MockArray


class PreprocessedData(Data):
    """Class for loading preprocessed data.

    Contains methods which can be used to prepare the data for training a model.
    This includes methods to perform time embedding and PCA.

    Parameters
    ----------
    inputs : list of str or str
        Filenames to be read.
    store_dir : str
        Directory to save results and intermediate steps to.
    output_file : str
        Filename to save memory map to.
    """

    def __init__(self, inputs, store_dir="tmp", output_file=None):
        super().__init__(inputs, store_dir)
        if output_file is None:
            self.output_file = f"dataset_{self._identifier}.npy"

    def prepare_memmap_filenames(self):
        self.te_pattern = "te_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(len(self.inputs))), identifier=self._identifier
        )
        self.output_pattern = "output_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(len(self.inputs))), identifier=self._identifier
        )

        # Time embedded data memory maps
        self.te_memmaps = []
        self.te_filenames = [
            str(self.store_dir / self.te_pattern.format(i=i))
            for i, _ in enumerate(self.inputs)
        ]

        # Prepared data memory maps (time embedded and pca'ed)
        self.output_memmaps = []
        self.output_filenames = [
            str(self.store_dir / self.output_pattern.format(i=i))
            for i, _ in enumerate(self.inputs)
        ]

    def prepare(
        self, n_embeddings: int, n_pca_components: int, whiten: bool = False,
    ):
        """Prepares data to train the model with.

        Performs standardization, time embedding and principle component analysis.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data.
        n_pca_components : int
            Number of PCA components to keep.
        whiten : bool
            Should we whiten the PCA'ed data? Optional, default is False.
        """
        self.prepare_memmap_filenames()

        # Standardise and time embed the data for each subject
        for memmap, discontinuities, new_file in zip(
            tqdm(self.raw_data_memmaps, desc="Time embedding", ncols=98),
            self.discontinuities,
            self.te_filenames,
        ):
            memmap = manipulation.standardize(memmap, discontinuities)
            te_shape = (
                memmap.shape[0] - (n_embeddings + 1) * len(discontinuities),
                memmap.shape[1] * (n_embeddings + 2),
            )
            te_memmap = MockArray.get_memmap(new_file, te_shape, dtype=np.float32)
            te_memmap = manipulation.time_embed(
                memmap, discontinuities, n_embeddings, output_file=te_memmap
            )
            self.te_memmaps.append(te_memmap)

        # Update discontinuity indices
        for i in range(len(self.discontinuities)):
            self.discontinuities[i] -= n_embeddings

        # Perform principle component analysis (PCA)
        print("Calculating PCA")
        covariance = np.zeros([te_memmap.shape[1], te_memmap.shape[1]])
        for te_memmap in self.te_memmaps:
            covariance += np.transpose(te_memmap - te_memmap.mean(axis=0)) @ (
                te_memmap - te_memmap.mean(axis=0)
            )
        u, s, vh = np.linalg.svd(covariance)
        u = u[:, :n_pca_components]
        s = s[:n_pca_components]
        if whiten:
            u = u @ np.diag(1.0 / np.sqrt(s))

        # Apply PCA to the data for each subject and standardise again
        for te_memmap, discontinuities, output_file in zip(
            tqdm(self.te_memmaps, desc="Applying PCA", ncols=98),
            self.discontinuities,
            self.output_filenames,
        ):
            pca_te_shape = (
                te_memmap.shape[0] - (n_embeddings + 1) * len(discontinuities),
                n_pca_components,
            )
            pca_te_memmap = MockArray.get_memmap(
                output_file, pca_te_shape, dtype=np.float32
            )
            pca_te_memmap = te_memmap @ u
            pca_te_memmap = manipulation.standardize(pca_te_memmap, discontinuities)
            self.output_memmaps.append(pca_te_memmap)

        # Update subjects to return the prepared data
        self.subjects = self.output_memmaps

        self.prepared = True
        self.n_embeddings = n_embeddings
        self.n_pca_components = n_pca_components
        self.whiten = whiten

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
