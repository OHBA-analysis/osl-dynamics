"""Example code for reproducing plots from D. Vidaurre's 2018 Nature Comms paper.

"""

print("Setting up")
from dynemo.analysis import connectivity, power, spectral
from dynemo.data import OSL_HMM, Data, io
from dynemo.utils import plotting

# Load an HMM fit
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-55_K-12/hmm.mat"
)
cov = hmm.covariances
alp = hmm.alpha(concatenate=True)

n_embeddings = 15
pca_components = io.loadmat(
    "/well/woolrich/projects/uk_meg_notts/eo/"
    "natcomms18/prepared_data/pca_components.mat"
)
sampling_frequency = 250
frequency_range = [1, 45]

mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Calculate PSDs and coherence using multitaper method
preprocessed_data = Data(
    [
        f"/well/woolrich/projects/uk_meg_notts/eo/natcomms18/src_rec/subject{i}.mat"
        for i in range(1, 56)
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

# Non-frequency specific power maps
power_map = power.variance_from_spectra(f, psd)
power.save(
    power_map=power_map,
    filename="mt_fullrange_power_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)

# Non-frequency specific connectivity
conn_map = connectivity.covariance_from_spectra(f, psd)
connectivity.save(
    connectivity_map=conn_map,
    threshold=0.98,
    filename="mt_fullrange_conn_.png",
    parcellation_file=parcellation_file,
)

# Fit two spectral components to the coherence
wideband_components = spectral.decompose_spectra(coh, n_components=2)
plotting.plot_line([f, f], wideband_components, filename="wideband.png")

power_map = power.variance_from_spectra(f, psd, wideband_components)
conn_map = connectivity.mean_coherence_from_spectra(f, coh, wideband_components)
for component in range(2):
    power.save(
        power_map=power_map,
        filename=f"mt_wideband{component}_power_.png",
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        subtract_mean=True,
        component=component,
    )
    connectivity.save(
        connectivity_map=conn_map,
        threshold=0.98,
        filename=f"mt_wideband{component}_conn_.png",
        parcellation_file=parcellation_file,
        component=component,
    )

# Fit four spectral components to the coherence
narrowband_components = spectral.decompose_spectra(coh, n_components=4)
plotting.plot_line([f, f, f, f], narrowband_components, filename="narrowband.png")

power_map = power.variance_from_spectra(f, psd, narrowband_components)
conn_map = connectivity.mean_coherence_from_spectra(f, coh, narrowband_components)
for component in range(4):
    power.save(
        power_map=power_map,
        filename=f"mt_narrowband{component}_power_.png",
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        subtract_mean=True,
        component=component,
    )
    connectivity.save(
        connectivity_map=conn_map,
        threshold=0.98,
        filename=f"mt_narrowband{component}_conn_.png",
        parcellation_file=parcellation_file,
        component=component,
    )
