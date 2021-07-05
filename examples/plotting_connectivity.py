"""Example code for plotting connectivity.

"""

print("Setting up")
import numpy as np
from vrad.analysis import connectivity, power, spectral, states
from vrad.data import Data, io, OSL_HMM
from vrad.utils import plotting

# Load an HMM fit
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-10_K-6/hmm.mat"
)
cov = hmm.covariances
alp = hmm.alpha(concatenate=True)

n_embeddings = 15
pca_components = io.loadmat(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/prepared_data/pca_components.mat"
)
sampling_frequency = 250
frequency_range = [1, 45]

mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Connectivity using correlations taken from the raw covariance matrix
raw_cov = states.raw_covariances(
    state_covariances=cov,
    n_embeddings=n_embeddings,
    pca_components=pca_components,
    zero_lag=False,
)

conn_map = abs(raw_cov)
connectivity.save(
    connectivity_map=conn_map,
    threshold=0.98,
    filename="raw_.png",
    parcellation_file=parcellation_file,
)

# Connectivity using covariance calculated from autocorrelation functions
acf = states.autocorrelation_functions(
    state_covariances=cov,
    n_embeddings=n_embeddings,
    pca_components=pca_components,
)
f, psd, coh = spectral.state_covariance_spectra(
    acf,
    sampling_frequency=sampling_frequency,
    frequency_range=frequency_range,
)

conn_map = abs(np.sum(psd, axis=-1))
connectivity.save(
    connectivity_map=conn_map,
    threshold=0.98,
    filename="acf_.png",
    parcellation_file=parcellation_file,
)

# Connectivity using the multitaper method
preprocessed_data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject{i}.mat"
        for i in range(1, 11)
    ]
)
ts = preprocessed_data.trim_raw_time_series(n_embeddings=n_embeddings, concatenate=True)

f, psd, coh = spectral.multitaper_spectra(
    data=ts,
    alpha=alp,
    sampling_frequency=sampling_frequency,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=frequency_range,
)

wideband_components = spectral.decompose_spectra(coh, n_components=2)
plotting.plot_line([f, f], wideband_components, filename="wideband.png")

conn_map = connectivity.covariance_from_spectra(f, psd, wideband_components)
connectivity.save(
    connectivity_map=conn_map,
    threshold=0.98,
    filename="mt_wideband_.png",
    parcellation_file=parcellation_file,
    component=0,  # only plot the first spectral component
)

# Delete temporary directory
preprocessed_data.delete_dir()
