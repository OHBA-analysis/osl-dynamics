"""Get training data.

This script will download source reconstructed resting-state MEG data.
This data is part of the MRC MEGUK dataset. It is CTF data from the
Nottingham site.
"""

import os

# We will download example data hosted on osf.io/by2tc.
# Note, osfclient must be installed. This can be installed with pip:
#
#     pip install osfclient


def get_data(name, output_dir):
    if os.path.exists(output_dir):
        print(f"{output_dir} already downloaded. Skipping..")
        return
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {output_dir}")
    os.remove(f"{name}.zip")
    print(f"Data downloaded to: {output_dir}")


# Download the dataset (approximately 600 MB)
#
# This will unzip the notts_rest_55_subj.zip file into a
# directory called "training_data"
get_data("notts_rest_55_subj", output_dir="training_data")
