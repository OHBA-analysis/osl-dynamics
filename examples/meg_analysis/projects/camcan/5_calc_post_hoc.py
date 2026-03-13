"""Post-hoc analysis.

"""

import os
from sys import argv

if len(argv) != 2:
    print("Please pass the number of states and run id, e.g. python 5_calc_post_hoc.py 8")
    exit()

n_states = int(argv[1])

output_dir = f"results/inf_params/{n_states:02d}_states"

import pickle
import numpy as np
from glob import glob

from osl_dynamics.data import Data
from osl_dynamics.inference import modes
from osl_dynamics.analysis import spectral

# Load parcellated data
files = sorted(glob("/well/woolrich/projects/camcan/spring23/src/*/sflip_parc-raw.fif"))[:20]
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=16)
x = data.trim_time_series(n_embeddings=15, sequence_length=400)

# Load state probability time courses
alp = pickle.load(open(f"{output_dir}/alp.pkl", "rb"))

# Calculate multitaper spectra
f, psd, coh, w = spectral.multitaper_spectra(
    data=x,
    alpha=alp,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    standardize=True,
    return_weights=True,
    n_jobs=16,
)
np.save(f"{output_dir}/f.npy", f)
np.save(f"{output_dir}/psd.npy", psd)
np.save(f"{output_dir}/coh.npy", coh)
np.save(f"{output_dir}/w.npy", w)

# Calculate non-negative matrix factorisation on the stacked coherences
nnmf = spectral.decompose_spectra(coh, n_components=2)
np.save(f"{output_dir}/nnmf_2.npy", nnmf)

# Convert probabilities to a state time course
stc = modes.argmax_time_courses(alp)

# Calculate subject-specific summary stats
fo = modes.fractional_occupancies(stc)
lt = modes.mean_lifetimes(stc, sampling_frequency=250)
intv = modes.mean_intervals(stc, sampling_frequency=250)
sr = modes.switching_rates(stc, sampling_frequency=250)
np.save(f"{output_dir}/fo.npy", fo)
np.save(f"{output_dir}/lt.npy", lt)
np.save(f"{output_dir}/intv.npy", intv)
np.save(f"{output_dir}/sr.npy", sr)
