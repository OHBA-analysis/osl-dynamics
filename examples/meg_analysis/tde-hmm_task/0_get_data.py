"""Get training data.

This script will download source reconstructed task MEG data collected using
an Elekta scanner.

The data is from the Wakeman-Henson dataset: https://www.nature.com/articles/sdata20151

It consists of 19 subjects performing a visual perception task.
Each subject performed 6 runs.
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
#
# This will unzip the wakeman_henson.zip file into a
# directory called "data"
get_data("wakeman_henson_glasser", output_dir="data")
