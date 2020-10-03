import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from vrad.data import manipulation
from vrad.data._base import Data
from vrad.utils.misc import MockArray, array_to_memmap


class PreprocessedData(Data):
    """Class for loading preprocessed data.

    Contains methods which can be used to prepare the data for training a model.
    This includes methods to perform time embedding and PCA.
    """

    def __init__(self, inputs, store_dir="tmp", output_file=None):
        super().__init__(inputs, store_dir)
        if output_file is None:
            self.output_file = f"dataset_{self._identifier}.npy"

    def prepare_memmap_filenames(self):
        self.te_pattern = "te_data_{{i:0{width}d}}.npy".format(
            width=len(str(len(self.inputs)))
        )
        self.output_pattern = "output_data_{{i:0{width}d}}.npy".format(
            width=len(str(len(self.inputs)))
        )

        # Time embedded data memory maps
        self.te_memmaps = []
        self.te_filenames = [
            str(self.store_dir / self.te_pattern.format(i=i))
            for i, _ in enumerate(self.inputs)
        ]

        # Prepared data memory maps
        self.output_memmaps = []
        self.output_filenames = [
            str(self.store_dir / self.output_pattern.format(i=i))
            for i, _ in enumerate(self.inputs)
        ]

    def prepare(
        self, n_embeddings: int, n_pca_components: int, whiten: bool,
    ):
        self.prepare_memmap_filenames()

        # Time embed the data for each subject
        for memmap, new_file in zip(
            tqdm(self.raw_data_memmaps, desc="Time embedding", ncols=98),
            self.te_filenames,
        ):
            te_shape = (
                memmap.shape[0],
                memmap.shape[1] * len(range(-n_embeddings // 2, n_embeddings // 2 + 1)),
            )
            te_memmap = MockArray.get_memmap(new_file, te_shape, dtype=np.float32)

            te_memmap = manipulation.time_embed(
                memmap, n_embeddings, output_file=te_memmap
            )
            te_memmap = manipulation.scale(te_memmap)

            self.te_memmaps.append(te_memmap)

        # Perform principle component analysis (PCA)
        pca = PCA(n_pca_components, svd_solver="full", whiten=whiten)
        for te_memmap in tqdm(self.te_memmaps, desc="Calculating PCA", ncols=98):
            pca.fit(te_memmap)

        # Apply PCA to the data for each subject
        for output_file, te_memmap in zip(
            tqdm(self.output_filenames, desc="Applying PCA", ncols=98), self.te_memmaps
        ):
            pca_result = pca.transform(te_memmap)
            pca_result = array_to_memmap(output_file, pca_result)
            self.output_memmaps.append(pca_result)

        # Update subjects to return the prepared data
        self.subjects = self.output_memmaps

        self.prepared = True
        self.n_embeddings = n_embeddings
        self.n_pca_components = n_pca_components
        self.whiten = whiten
