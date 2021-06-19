import numpy as np
from vrad.analysis.connectivity import plot_connectivity
from vrad.utils.parcellation import Parcellation

edges = np.random.uniform(low=-10.0, high=10.0, size=(38, 38))
select = np.logical_and(edges >= 0, edges <= 1.0)
parcellation = Parcellation(
    "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
)

plot_connectivity(
    edges,
    parcellation,
    inflation=0,
    selection=select,
)
