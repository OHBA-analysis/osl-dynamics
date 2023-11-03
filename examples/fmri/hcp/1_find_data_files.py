"""Make a list of the data files we want to train on.

"""

import os
from glob import glob

paths = sorted(
    glob(
        "/gpfs3/well/win-hcp/HCP-YA/HCP_PTN1200/node_timeseries"
        "/3T_HCP1200_MSMAll_d50_ts2/*.txt"
    )
)

print("Found", len(paths), "files")

with open("data_files.txt", "w") as file:
    for path in paths:
        file.write(path + "\n")
