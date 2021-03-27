import numpy as np
from tqdm import tqdm
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

    # Find channels that are common to all SPM files only keeping the MEG
    # Recordings. N.b. the ordering of this list is random.
    common_channels = set(channel_labels[0]).intersection(*channel_labels)
    common_channels = [channel for channel in common_channels if "M" in channel]

    # Write the channel labels to file in the correct order
    with open(output_folder + "/channels.dat", "w") as file:
        for channel in spm.channel_labels:
            if channel in common_channels:
                file.write(channel + "\n")

    # Write data to file only keeping the common channels
    for i in tqdm(range(len(spm_filenames)), desc="Writing files", ncols=98):
        spm = SPM(spm_filenames[i], load_data=True)
        channels = [label in common_channels for label in spm.channel_labels]

        output_filename = output_folder + f"/subject{i}.npy"
        output_data = spm.data[:, channels].astype(np.float32)
        np.save(output_filename, output_data)
