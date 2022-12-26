"""Example code for calculating summary statistics from a state time course.

"""

print("Setting up")
import numpy as np
from osl_dynamics.analysis import modes
from osl_dynamics.data import HMM_MAR

# ---------
# Load data

# MATLAB HMM-MAR fit
hmm = HMM_MAR(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-10_K-6/hmm.mat"
)

# State time course shape is (n_subjects, n_samples, n_states)
stc = hmm.state_time_course()

# -----------------------------------
# Subject-specific summary statistics

print()
print("Subject-specific summary stats")
print("------------------------------")
print()

# Fractional occupancy
fo = modes.fractional_occupancies(stc)

print("Fractional occupancy:")
print(fo)
print()

# Mean lifetimes
lt = modes.mean_lifetimes(stc, sampling_frequency=250)

print("Mean lifetimes (ms):")
print(lt * 1e3)
print()

# Mean intervals
intv = modes.mean_intervals(stc, sampling_frequency=250)

print("Mean intervals (s):")
print(intv)
print()

# ------------------------------
# Group-level summary statistics

print("Group-level summary stats")
print("-------------------------")
print()

stc = np.concatenate(stc)

# Fractional occupancy
fo = modes.fractional_occupancies(stc)

print("Fractional occupancy:")
print(fo)
print()

# Mean lifetimes
lt = modes.mean_lifetimes(stc, sampling_frequency=250)

print("Mean lifetimes (ms):")
print(lt * 1e3)
print()

# Mean intervals
intv = modes.mean_intervals(stc, sampling_frequency=250)

print("Mean intervals (s):")
print(intv)
