import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from vrad.data import manipulation
from vrad.data._base import Data
from vrad.utils.misc import MockArray, array_to_memmap


class PreprocessedData(Data):
    def __init__(self, inputs, store_dir="tmp"):
        super().__init__(inputs, store_dir)

    def prepare_memmaps(self):
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

        pca = PCA(n_pca_components, svd_solver="full", whiten=whiten)
        for te_memmap in tqdm(self.te_memmaps, desc="Calculating PCA", ncols=98):
            pca.fit(te_memmap)
        for output_file, te_memmap in zip(
            tqdm(self.output_filenames, desc="Applying PCA", ncols=98), self.te_memmaps
        ):
            pca_result = pca.transform(te_memmap)
            pca_result = array_to_memmap(output_file, pca_result)
            self.output_memmaps.append(pca_result)

        self.prepared = True
