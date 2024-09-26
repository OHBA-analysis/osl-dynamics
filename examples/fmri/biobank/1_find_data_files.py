"""Make a list of the data files we want to train on.

"""

import os
import scipy.io
from tqdm.auto import trange

# %% Subject IDs

# Load matlab file containing subject IDs
mat = scipy.io.loadmat(
    "/gpfs3/well/win-biobank/projects/imaging/data/data3/ANALYSIS/workspace13d.mat",
    only_include="subject_IDs_unique",
)
subject_IDs = mat["subject_IDs_unique"]
exit()
# %% Paths to files

# Get paths to files that exist (this takes about 30 seconds)
paths = []
for i in trange(subject_IDs.shape[0], desc="Finding files"):
    id = subject_IDs[i]
    if str(id)[0] == "2":
        path = f"/gpfs3/well/win-biobank/projects/imaging/data/data3/subjectsAll/2{int(id)}/fMRI/rfMRI_25.dr/dr_stage1.txt"
        if os.path.exists(path):
            paths.append(path)

print("Found", len(paths), "files")

# %% Save

with open("data_files.txt", "w") as file:
    for path in paths:
        file.write(path + "\n")
