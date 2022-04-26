"""Example code for plotting different types of power maps.

"""

print("Setting up")
from osl_dynamics.analysis import power, spectral, modes, workbench
from osl_dynamics.data import OSL_HMM, Data, io
from osl_dynamics.utils import plotting

# Load an HMM fit
hmm = OSL_HMM(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-10_K-6/hmm.mat"
)
cov = hmm.covariances
alp = hmm.alpha(concatenate=True)

n_embeddings = 15
pca_components = io.loadmat(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18"
    "/prepared_data/pca_components.mat"
)
sampling_frequency = 250
frequency_range = [1, 45]

mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# Use elements of the mode covariance matrices for the power maps
power_map = modes.raw_covariances(
    mode_covariances=cov,
    n_embeddings=n_embeddings,
    pca_components=pca_components,
)
power.save(
    power_map=power_map,
    filename="var.nii.gz",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)
workbench.setup("/well/woolrich/projects/software/workbench/bin_rh_linux64")
workbench.render("var.nii.gz", "tmp", gui=False, image_name="var_.png")

# Calculate power maps using power spectra calculated using the mode covariances
acf = modes.autocorrelation_functions(
    mode_covariances=cov,
    n_embeddings=n_embeddings,
    pca_components=pca_components,
)
f, psd, _ = spectral.mode_covariance_spectra(
    acf,
    sampling_frequency=sampling_frequency,
    frequency_range=frequency_range,
)

power_map = power.variance_from_spectra(f, psd)
power.save(
    power_map=power_map,
    filename="acf_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)

# Calculate power maps using the multitaper method
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

power_map = power.variance_from_spectra(f, psd)
power.save(
    power_map=power_map,
    filename="mt_fullrange_.png",
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
)

wideband_components = spectral.decompose_spectra(coh, n_components=2)
plotting.plot_line([f, f], wideband_components, filename="wideband.png")

power_map = power.variance_from_spectra(f, psd, wideband_components)
for component in range(2):
    power.save(
        power_map=power_map,
        filename=f"mt_wideband{component}_.png",
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        subtract_mean=True,
        component=component,
    )

narrowband_components = spectral.decompose_spectra(coh, n_components=4)
plotting.plot_line([f, f, f, f], narrowband_components, filename="narrowband.png")

power_map = power.variance_from_spectra(f, psd, narrowband_components)
for component in range(4):
    power.save(
        power_map=power_map,
        filename=f"mt_narrowband{component}_.png",
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        subtract_mean=True,
        component=component,
    )
