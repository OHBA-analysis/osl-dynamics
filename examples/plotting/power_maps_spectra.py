"""Example code for plotting power maps calculated using the state/mode spectra
estimated from the source reconstructed data and inferred state/mode time course.

The spectra can be calculate with a multitaper (in the case of a state time
course) or regression (in the case of a mode time course).

See examples/spectra/hmm_multitaper.py for how to calculate a multitaper
and examples/spectra/dynemo_regression_spectra.py for how to calculate regression spectra.

In this script we assume this has been done and the subject-specific spectra
have been saved with spectra.save("spectra.npz").
"""

from osl_dynamics.analysis import power, spectral

# Load the subject-specific spectra
spectra = spectral.Spectra.load("spectra.npz")

# Calculate the group average
group = spectra.group_average()

# Calculate power maps from the spectra: the change from the static power
# (regression coefficients or state PSD minus the static PSD)
# (frequency_range is an optional argument)
power_map = group.power_maps(frequency_range=[1, 30])

# Save the power maps as images
power.save(
    power_map,
    parcellation_file="atlas-Giles_nparc-42_space-MNI_res-8x8x8.nii.gz",
    filename="maps_.png",
    plot_kwargs={"symmetric_cbar": True},
)
