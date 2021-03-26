from tqdm import tqdm
from vrad.data.io import write_h5_file
from vrad.data.spm import SPM


def make_channels_consistent(spm_filenames: list, output_folder: str = "."):
    """Removes channels that are not present in all subjects.

    Parameters
    ----------
    spm_filenames : list of str
        Path to SPM files containing the preprocessed data.
    output_folder : str
        Path to folder to write preprocessed data to. Optional, default
        is the current working directory.
    """
    # Get the channel labels
    channel_labels = []
    for filename in tqdm(spm_filenames, desc="Loading files", ncols=98):
        spm = SPM(filename, load_data=False)
        channel_labels.append(spm.channel_labels)

    # Find channels that are common to all SPM files
    common_channels = set(channel_labels[0]).intersection(*channel_labels)

    # Write data to file only keeping the common channels
    for i in tqdm(range(len(spm_filenames)), desc="Writing files", ncols=98):
        spm = SPM(spm_filenames[i], load_data=True)
        channels = [label in common_channels for label in spm.channel_labels]
        X = spm.data[:, channels]
        T = spm.discontinuities

        output_filename = output_folder + f"/subject{i+1}.h5"
        write_h5_file(data={"X": X, "T": T}, filename=output_filename)
