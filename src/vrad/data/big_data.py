import pathlib
from glob import glob
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.python.data import Dataset
from tqdm import tqdm, trange
from vrad.data.subject import Subject

_rng = np.random.default_rng()


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
