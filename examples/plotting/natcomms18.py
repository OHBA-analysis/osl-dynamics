"""Example code for reproducing plots from D. Vidaurre's 2018 Nature Comms paper.

"""

print("Setting up")
import os
import numpy as np
from osl_dynamics.analysis import connectivity, power, spectral
from osl_dynamics.data import OSL_HMM, Data, rw
from osl_dynamics.utils import plotting

# Make directory for plots
os.makedirs("figures", exist_ok=True)

# Load an HMM fit
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-55_K-12/hmm.mat"
)
alp = hmm.alpha()

n_embeddings = 15
pca_components = rw.loadmat(
    "/well/woolrich/projects/uk_meg_notts/eo"
    + "/natcomms18/prepared_data/pca_components.mat"
)
sampling_frequency = 250
frequency_range = [1, 45]

mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Load source reconstructed data
preprocessed_data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject{i}.mat"
        for i in range(1, 56)
    ]
)
ts = preprocessed_data.trim_raw_time_series(n_embeddings=n_embeddings)

# Calculate subject-specific PSDs and coherences using multitaper method
f, psd, coh, w = spectral.multitaper_spectra(
    data=ts,
    alpha=alp,
    sampling_frequency=sampling_frequency,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=frequency_range,
    return_weights=True,
)

# Group average PSD and coherence
gpsd = np.average(psd, axis=0, weights=w)
gcoh = np.average(coh, axis=0, weights=w)

# Fit two spectral components to the subject-specific coherences
wideband_components = spectral.decompose_spectra(coh, n_components=2)
plotting.plot_line([f, f], wideband_components, filename="figures/wideband.png")

# Calculate power and connectivity maps using PSDs and coherences
power_map = power.variance_from_spectra(f, gpsd, wideband_components)
conn_map = connectivity.mean_coherence_from_spectra(
    f,
    gcoh,
    wideband_components,
    fit_gmm=True,
)

# Just plot the first component (second is noise)
power.save(
    power_map=power_map,
    filename="figures/mt_wideband0_power_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
    component=0,
)
connectivity.save(
    connectivity_map=conn_map,
    threshold=0.925,
    filename="figures/mt_wideband0_conn_.png",
    parcellation_file=parcellation_file,
    component=0,
)

# Fit four spectral components to the subject-specific coherences
narrowband_components = spectral.decompose_spectra(coh, n_components=4)
plotting.plot_line(
    [f, f, f, f], narrowband_components, filename="figures/narrowband.png"
)

# Calculate power and connectivity maps using PSDs and coherences
power_map = power.variance_from_spectra(f, gpsd, narrowband_components)
conn_map = connectivity.mean_coherence_from_spectra(
    f, gcoh, narrowband_components, fit_gmm=True
)

# Plot the first 3 components
for component in range(3):
    power.save(
        power_map=power_map,
        filename=f"figures/mt_narrowband{component}_power_.png",
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        subtract_mean=True,
        component=component,
    )
    connectivity.save(
        connectivity_map=conn_map,
        threshold=0.925,
        filename=f"figures/mt_narrowband{component}_conn_.png",
        parcellation_file=parcellation_file,
        component=component,
    )
