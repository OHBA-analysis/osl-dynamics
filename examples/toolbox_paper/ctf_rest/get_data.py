"""Get training data.

This script will download source reconstructed resting-state (eyes open)
MEG data.

This data is part of the MRC MEGUK dataset. It is CTF data from the
Nottingham site. 65 subjects are part of this dataset.
"""

import os

def get_data(name, output_dir):
    if os.path.exists(output_dir):
        print(f"{output_dir} already downloaded. Skipping..")
        return
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {output_dir}")
    os.remove(f"{name}.zip")
    print(f"Data downloaded to: {output_dir}")

# We will download example data hosted on osf.io/by2tc.
# (approximately 708 MB)
#
# This will unzip the notts_mrc_meguk_giles.zip file into a
# directory called "training_data". There are two subdirectories:
# - "bursts", which contains single channel data for the burst
#   detection pipeline.
# - "networks", which contains multi-channel data for the dynamic
#   network analysis pipelines.
get_data("notts_mrc_meguk_giles", output_dir="training_data")
