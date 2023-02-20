"""Example code for calculating summary statistics from a state time course.

"""

import pickle

from osl_dynamics.inference import modes

# Get the inferred state probabilities from the HMM
#
# We saved these using the get_inf_params.py script.
alp = pickle.load(open("model/alp.pkl", "rb"))

# Calculate a state time course by taking the most probable state at each time point
stc = modes.argmax_time_courses(alp)

# Calculate fractional occupancy
fo = modes.fractional_occupancies(stc)

print("Fractional occupancy:")
print(fo)
print()

# Calculate mean lifetimes
lt = modes.mean_lifetimes(stc, sampling_frequency=250)

print("Mean lifetimes (ms):")
print(lt * 1e3)
print()

# Calculate mean intervals
intv = modes.mean_intervals(stc, sampling_frequency=250)

print("Mean intervals (s):")
print(intv)
