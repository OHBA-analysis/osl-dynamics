"""Static analysis of Nottingham movie data.

"""

print("Setting up")
import os
import numpy as np
from osl_dynamics.analysis import static
from osl_dynamics.data import Data
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power, connectivity

#%% Setup

run = 1

#subjects_dir = '/well/woolrich/projects/notts_movie_opm'
subjects_dir = '/Users/woolrich/homedir/vols_data/notts_movie_opm'

subjects_to_do = np.arange(0, 10)
sessions_to_do = np.arange(0, 2)
subj_sess_2exclude = np.zeros([10, 2]).astype(bool)

sampling_frequency = 150  # Hz

# Source reconstruction files used to create the source space data
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# Filenames
subjects = []
sf_files = []
recon_dir = os.path.join(subjects_dir, 'recon')
for sub in subjects_to_do:
    for ses in sessions_to_do:
        if not subj_sess_2exclude[sub, ses]:
            sub_dir = 'sub-' + ('{}'.format(subjects_to_do[sub]+1)).zfill(3)
            ses_dir = 'ses-' + ('{}'.format(sessions_to_do[ses]+1)).zfill(3)
            subject = sub_dir + '_' + ses_dir
            sf_file = os.path.join(recon_dir, subject + '/sflip_parc.npy')
            subjects.append(subject)
            sf_files.append(sf_file)

static_dir = f"{recon_dir}/static"
os.makedirs(static_dir, exist_ok=True)

#%% Plot AECs

for temporal_filter in [[4, 7], [8, 12], [13, 30]]:

    # Prepare data
    data = Data(sf_files, sampling_frequency=sampling_frequency)
    methods = {
        "filter": {"low_freq": temporal_filter[0], "high_freq": temporal_filter[1]},
        "amplitude_envelope": {},
        "standardize": {},
    }
    data.prepare(methods)
    ts = data.time_series()

    # Calculate functional connectivity (Pearson correlation)
    conn_map = static.functional_connectivity(ts)

    # Plot group mean
    conn_map = np.mean(conn_map, axis=0)
    connectivity.save(
        connectivity_map=conn_map,
        filename=f"{static_dir}/aec_group_{temporal_filter[0]}_{temporal_filter[1]}_.png",
        parcellation_file=parcellation_file,
        threshold=0.97,
    )

#%% Calculate static PSDs

# Load data
data = Data(sf_files)
ts = data.time_series()

# Calculate spectra
f, psd = static.welch_spectra(
    data=ts,
    window_length=500,
    sampling_frequency=sampling_frequency,
    standardize=True,
)
np.save(f"{static_dir}/f.npy", f)
np.save(f"{static_dir}/psd.npy", psd)

#%% Plot PSDs

# Load spectra (calculated with static/calc_spectra.py)
f = np.load(f"{static_dir}/f.npy")
psd = np.load(f"{static_dir}/psd.npy")

# Average over channels
p = np.mean(psd, axis=1)
plotting.plot_line(
    [f] * p.shape[0],
    p,
    labels=[f"Subject {i + 1}" for i in range(p.shape[0])],
    filename=f"{static_dir}/psd.png",
)

# Subject specific power maps
varmap = power.variance_from_spectra(f, psd)
power.save(
    power_map=varmap,
    filename=f"{static_dir}/power_subj_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,  # display the differences in power relative to the group mean
)

# Group level power map
group_varmap = np.mean(varmap, axis=0)
power.save(
    power_map=group_varmap,
    filename=f"{static_dir}/power_group_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
)

# Plot group level power map for different freq bands
psd = np.load(f"{static_dir}/psd.npy")
for frequency_range in [[7, 13], [13, 30]]:
    varmap = power.variance_from_spectra(f, psd, frequency_range=frequency_range)
    group_varmap = np.mean(varmap, axis=0)
    power.save(
        power_map=group_varmap,
        filename=f"{static_dir}/power_group_{frequency_range[0]}_{frequency_range[1]}_.png",
        mask_file=mask_file,
        parcellation_file=parcellation_file,
    )
