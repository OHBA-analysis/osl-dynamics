import pathlib

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.python.data import Dataset
from tqdm import tqdm
from vrad.data import io, manipulation
from vrad.utils.misc import MockArray, array_to_memmap


class BigData:
    def __init__(self, files, store_dir="tmp", output_file="dataset.npy"):
        self.files = files
        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.input_pattern = "input_data_{{i:0{width}d}}.npy".format(
            width=len(str(len(self.files)))
        )
        self.te_pattern = "te_data_{{i:0{width}d}}.npy".format(
            width=len(str(len(self.files)))
        )
        self.output_pattern = "output_data_{{i:0{width}d}}.npy".format(
            width=len(str(len(self.files)))
        )

        # Preprocessed data memory maps
        self.data_memmaps = []
        self.data_filenames = [
            str(self.store_dir / self.input_pattern.format(i=i))
            for i, _ in enumerate(files)
        ]

        # Time embedded data memory maps
        self.te_memmaps = []
        self.te_filenames = [
            str(self.store_dir / self.te_pattern.format(i=i))
            for i, _ in enumerate(files)
        ]

        # Prepared data memory maps
        self.output_memmaps = []
        self.output_filenames = [
            str(self.store_dir / self.output_pattern.format(i=i))
            for i, _ in enumerate(files)
        ]

        self.output_file = output_file

        # Flag to indicate whether output_memmaps are been generated
        self.prepared = False

        # Load the preprocessed data
        self.load_data()

    @property
    def raw_data(self):
        return self.data_memmaps

    @property
    def n_channels(self):
        if self.prepared:
            return self.output_memmaps[0].shape[1]
        else:
            return self.data_memmaps[0].shape[1]

    def load_data(self):
        for in_file, out_file in zip(
            tqdm(self.files, desc="Loading files", ncols=98), self.data_filenames
        ):
            np.save(out_file, io.load_data(in_file)[0])
            self.data_memmaps.append((np.load(out_file, mmap_mode="r+")))

    @staticmethod
    def num_batches(arr, sequence_length: int, step_size: int = None):
        step_size = step_size or sequence_length
        final_slice_start = arr.shape[0] - sequence_length + 1
        index = np.arange(0, final_slice_start, step_size)[:, None] + np.arange(
            sequence_length
        )
        return len(index)

    def count_batches(self, sequence_length, step_size=None):
        return np.array(
            [
                self.num_batches(memmap, sequence_length, step_size)
                for memmap in self.output_memmaps
            ]
        )

    def prepare(
        self, n_embeddings: int, n_pca_components: int, whiten: bool,
    ):
        for memmap, new_file in zip(
            tqdm(self.data_memmaps, desc="Time embedding", ncols=98), self.te_filenames
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

    def training_dataset(self, sequence_length, batch_size=32, step_size=None):
        subjects = self.output_memmaps or self.data_memmaps
        num_batches = self.count_batches(sequence_length, step_size)

        subject_datasets = []
        for i in range(len(subjects)):
            subject = subjects[i]
            subject_data = Dataset.from_tensor_slices(subject).batch(
                sequence_length, drop_remainder=True
            )
            subject_tracker = Dataset.from_tensor_slices(
                np.zeros(num_batches[i], dtype=np.float32) + i
            )
            subject_datasets.append(Dataset.zip((subject_data, subject_tracker)))

        full_dataset = subject_datasets[0]
        for subject_dataset in subject_datasets[1:]:
            full_dataset = full_dataset.concatenate(subject_dataset)

        return full_dataset.batch(batch_size).prefetch(-1)

    def prediction_dataset(self, sequence_length, batch_size=32, step_size=None):
        subjects = self.output_memmaps or self.data_memmaps

        subject_datasets = [
            Dataset.from_tensor_slices(subject)
            .batch(sequence_length, drop_remainder=True)
            .batch(batch_size)
            .prefetch(-1)
            for subject in subjects
        ]

        return subject_datasets
