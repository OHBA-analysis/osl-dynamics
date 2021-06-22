import numpy as np
from vrad import files
from vrad.utils import plotting

# Get example files
data_file = str(files.example.directory / "ctf275_data.npy")
channel_file = str(files.example.directory / "ctf275_channel_names.npy")

# N.b. that if you are reading in a cell array of file names from MATLAB, then you
# can use the following syntax to achieve the requisite formatting for the topoplot
# to work: chan_names = spio.loadmat('chan_names.mat') # file containing the
# [channels x 1] channel names chan_names = chan_names['ans'][0]  # access the
# variables in the cell array (called "ans" here) ctf275_channel_names = [chan_name[
# 0] for chan_name in chan_names] # extract the channel names

# Load example files
ctf275_data = np.load(data_file)
ctf275_channel_names = np.load(channel_file)

# Produce the figure using the "CTF275_helmet" layout provided by the FieldTrip toolbox
# www.fieldtriptoolbox.org.
# Layouts follow the naming of those in FieldTrip.
plotting.topoplot(
    layout="CTF275_helmet",
    data=ctf275_data,
    channel_names=ctf275_channel_names,
    plot_boxes=False,
    show_deleted_sensors=True,
    show_names=False,
    title="Example Plot",
    colorbar=True,
    cmap="plasma",
    n_contours=25,
)
