import pathlib
from glob import glob
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.python.data import Dataset
from tqdm import tqdm, trange
from vrad.data import io, manipulation
from vrad.data.subject import Subject
from vrad.utils.misc import MockArray, array_to_memmap

_rng = np.random.default_rng()


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

        self.data_memmaps = []
        self.data_filenames = [
            str(self.store_dir / self.input_pattern.format(i=i))
            for i, _ in enumerate(files)
        ]

        self.te_memmaps = []
        self.te_filenames = [
            str(self.store_dir / self.te_pattern.format(i=i))
            for i, _ in enumerate(files)
        ]

        self.output_memmaps = []
        self.output_filenames = [
            str(self.store_dir / self.output_pattern.format(i=i))
            for i, _ in enumerate(files)
        ]

        self.output_file = output_file

        self.load_data()

        self.n_components = None

    def load_data(self):
        for in_file, out_file in zip(
            tqdm(self.files, desc="loading_files"), self.data_filenames
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
        self,
        n_embeddings: int,
        n_components: int,
        whiten: bool,
        random_seed: int = None,
    ):
        self.n_components = n_components

        for new_file, memmap in zip(
            self.te_filenames, tqdm(self.data_memmaps, desc="time embedding")
        ):
            te_shape = (
                memmap.shape[0],
                memmap.shape[1] * len(range(-n_embeddings // 2, n_embeddings // 2 + 1)),
            )
            te_memmap = MockArray.get_memmap(new_file, te_shape, dtype=np.float32)

            te_memmap = manipulation.time_embed(
                memmap, n_embeddings, random_seed, output_file=te_memmap
            )
            te_memmap = manipulation.scale(te_memmap)

            self.te_memmaps.append(te_memmap)

        pca_object = PCA(n_components, svd_solver="full", whiten=whiten)
        for te_memmap in tqdm(self.te_memmaps, desc="calculating pca"):
            pca_object.fit(te_memmap)
        for te_memmap, output_file in zip(
            self.te_memmaps, tqdm(self.output_filenames, desc="applying pca")
        ):
            pca_result = pca_object.transform(te_memmap)
            pca_result = array_to_memmap(output_file, pca_result)
            self.output_memmaps.append(pca_result)

        return pca_object

    def training_dataset(self, sequence_length, step_size=None):
        subjects = self.output_memmaps or self.data_memmaps

        subject_datasets = [
            Dataset.from_tensor_slices(subject).batch(
                sequence_length, drop_remainder=True
            )
            for subject in subjects
        ]
        full_dataset = subject_datasets[0]
        for subject_dataset in subject_datasets[1:]:
            full_dataset = full_dataset.concatenate(subject_dataset)

        return full_dataset

    def prediction_dataset(self, sequence_length, step_size=None):
        subjects = self.output_memmaps or self.data_memmaps

        subject_datasets = [Dataset.from_tensor_slices(subject) for subject in subjects]
        full_dataset = subject_datasets[0]
        for subject_dataset in subject_datasets[1:]:
            full_dataset = full_dataset.concatenate(subject_dataset)

        full_dataset = full_dataset.batch(sequence_length, drop_remainder=True)

        return full_dataset


def prepare_many(
    files,
    n_embeddings: int,
    n_pca_components: int,
    whiten: bool,
    output_directory: str = "tmp",
    random_seed: int = None,
):
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(tqdm(files, desc="time embed")):
        subject = Subject(file)
        # Perform time embedding
        subject.time_embed(n_embeddings, random_seed=random_seed)

        # Rescale (z-transform) the time series
        subject.scaler = StandardScaler()
        subject.time_series = subject.scaler.fit_transform(subject.time_series)
        np.save(output_directory / f"te_subject_{i:04d}", subject.time_series)

    pca_object = PCA(n_pca_components, svd_solver="full", whiten=whiten)
    for i in trange(len(files), desc="find pca"):
        te_subject = np.load(output_directory / f"te_subject_{i:04d}.npy")
        pca_object.fit(te_subject)
    for i in trange(len(files), desc="apply pca"):
        pca_subject = pca_object.transform(
            np.load(output_directory / f"te_subject_{i:04d}.npy")
        )
        np.save(output_directory / f"pca_subject_{i:04d}.npy", pca_subject)

    return pca_object


def get_new_order(files, sequence_length):
    batch_lengths = np.array(
        [
            Subject(file).num_batches(sequence_length)
            for file in tqdm(files, desc="Reindexing")
        ]
    )
    cumsum_batch_lengths = batch_lengths.cumsum()
    new_order = _rng.permutation(batch_lengths.sum())
    insertion_points = np.searchsorted(cumsum_batch_lengths, new_order, side="right")
    for_subtraction = np.pad(cumsum_batch_lengths, (1, 0))
    internal_index = new_order - for_subtraction[insertion_points]
    return insertion_points, internal_index


def subjects_to_memmap(
    prepared_files,
    sequence_length,
    n_components,
    insertion_points,
    output_directory="tmp",
):
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    complete_memmap = np.memmap(
        output_directory / "memmap.npy",
        dtype=np.float32,
        mode="w+",
        shape=(len(insertion_points), sequence_length, n_components),
    )
    for i, file in enumerate(tqdm(prepared_files, desc="writing")):
        subject = Subject(file)
        complete_memmap[np.argwhere(insertion_points == i)[:, 0]] = subject.batch(
            sequence_length
        )
        complete_memmap.flush()
    return complete_memmap


def prep_pipeline(
    files: List[str],
    n_embeddings: int,
    n_pca_components: int,
    whiten: bool,
    sequence_length: int,
    output_directory: str = "tmp",
):
    output_directory = pathlib.Path(output_directory)

    pca_object = prepare_many(
        files, n_embeddings, n_pca_components, whiten, output_directory.absolute()
    )

    proc_files = sorted(glob(str((output_directory / "pca_subject*.npy").absolute())))

    insertion_points, internal_index = get_new_order(proc_files, sequence_length)

    memmap = subjects_to_memmap(
        proc_files,
        sequence_length,
        n_pca_components,
        insertion_points,
        output_directory,
    )

    with (output_directory / "dimensions.txt").open(mode="w") as f:
        print(f"n_channels: {n_pca_components}")
        print(f"sequence_length: {sequence_length}")

    return memmap, pca_object


def memmap_dataset(
    filename: str, sequence_length: int, n_channels: int, batch_size: int
):
    training_data = np.memmap(filename, dtype=np.float32, mode="r+")
    training_data = training_data.reshape((-1, sequence_length, n_channels))
    training_dataset = Dataset.from_tensor_slices(training_data).prefetch(-1)
    empty_dataset = Dataset.from_tensor_slices(np.zeros(training_data.shape[0]))

    # TODO: Subject tracking hasn't been implemented yet
    training_dataset = Dataset.zip((training_dataset, empty_dataset)).batch(batch_size)

    return training_dataset
