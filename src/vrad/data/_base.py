import pathlib
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.python.data import Dataset
from tqdm import tqdm
from vrad.data import io, manipulation
from vrad.utils.misc import MockArray, array_to_memmap


class Data:
    raw_data_pattern = "input_data_{{i:0{width}d}}.npy"
    n_embeddings = None

    def __init__(self, inputs, store_dir="tmp", output_file="dataset.npy"):
        self.inputs = inputs
        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_pattern = self.raw_data_pattern.format(
            width=len(str(len(inputs)))
        )

        # raw data memory maps
        self.raw_data_filenames = [
            str(self.store_dir / self.raw_data_pattern.format(i=i))
            for i, _ in enumerate(inputs)
        ]

        self.output_file = output_file

        # Load the preprocessed data
        self.raw_data_memmaps = self.load_data()
        self.subjects = self.raw_data_memmaps

    def __iter__(self):
        return iter(self.subjects)

    def __getitem__(self, item):
        return self.subjects[item]

    def validate_subjects(self):
        """Check all Subjects have the same shape."""
        n_channels = [subject.shape[1] for subject in self.subjects]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All subjects should have the same number of channels.")

    @property
    def raw_data(self) -> List:
        """Return raw data as a list of arrays."""
        return self.raw_data_memmaps

    @property
    def n_channels(self) -> int:
        """Return the number of channels in the current data state."""
        return self.subjects[0].shape[1]

    def load_data(self):
        """Import data into a list of Subjects."""
        memmaps = []
        for in_file, out_file in zip(
            tqdm(self.inputs, desc="Loading files", ncols=98), self.raw_data_filenames
        ):
            memmaps.append(io.load_data(out_file, mmap_location=out_file)[0])
        return memmaps

    def count_batches(self, sequence_length, step_size=None):
        return np.array(
            [
                manipulation.num_batches(memmap, sequence_length, step_size)
                for memmap in self.subjects
            ]
        )

    def training_dataset(self, sequence_length, batch_size=32, step_size=None):
        num_batches = self.count_batches(sequence_length, step_size)

        subject_datasets = []
        for i in range(len(self.subjects)):
            subject = self.subjects[i]
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

    def prediction_dataset(self, sequence_length, batch_size=32):
        start = self.n_embeddings // 2 if self.n_embeddings else None
        end = -start if start else None

        subject_datasets = [
            Dataset.from_tensor_slices(subject[start:end])
            .batch(sequence_length, drop_remainder=True)
            .batch(batch_size)
            .prefetch(-1)
            for subject in self.subjects
        ]

        return subject_datasets

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
        self.n_embeddings = n_embeddings
