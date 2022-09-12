"""Example code for creating a sensor space topoplot.

"""

import numpy as np
from osl_dynamics import files
from osl_dynamics.utils import plotting

# Scanner channel names
ctf275_channel_names = np.load(files.scanner.path / "ctf275_channel_names.npy")

# N.b. that if you are reading in a cell array of file names from MATLAB, then you
# can use the following syntax to achieve the requisite formatting for the topoplot
# to work:
#
# file containing the [channels x 1] channel names:
# chan_names = spio.loadmat('chan_names.mat')
#
# access the variables in the cell array (called "ans" here):
# chan_names = chan_names['ans'][0]
#
# extract the channel names:
# ctf275_channel_names = [chan_name[0] for chan_name in chan_names]

# Generate random data to plot
ctf275_data = np.random.normal(size=ctf275_channel_names.shape)

# Produce the figure using the "CTF275_helmet" layout provided by the FieldTrip
# toolbox: www.fieldtriptoolbox.org.
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
    filename="ctf275.png",
)
