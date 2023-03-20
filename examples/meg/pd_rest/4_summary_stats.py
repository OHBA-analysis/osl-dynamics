"""Calculate summary statistics for each state and do max-stat permutation
testing for significance.

"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import glmtools as glm
from osl_dynamics.inference import modes

# Directories
inf_params_dir = "results/inf_params"
summary_stats_dir = "results/summary_stats"

os.makedirs(summary_stats_dir, exist_ok=True)

#%% Load state time course

# State probabilities
alp = pickle.load(open(inf_params_dir + "/alp.pkl", "rb"))

# Calculate a state time course by taking the most likely state
stc = modes.argmax_time_courses(alp)

#%% Calculate summary statistics

# Fractional occupancy
fo = modes.fractional_occupancies(stc)

# Mean lifetime
lt = modes.mean_lifetimes(stc, sampling_frequency=250)

# Mean interval
intv = modes.mean_intervals(stc, sampling_frequency=250)

# Mean switching rate
sr = modes.switching_rates(stc, sampling_frequency=250)

np.save(summary_stats_dir + "/fo.npy", fo)
np.save(summary_stats_dir + "/lt.npy", lt)
np.save(summary_stats_dir + "/intv.npy", intv)
np.save(summary_stats_dir + "/sr.npy", sr)

#%% Max-stat permutation testing

# Load confounds
behav_data = pd.read_csv("data/BehaveData_SourceAnalysis.csv")

# Remove rows with nan and store indices of good rows
behav_data = behav_data[["Group", "Handedness", "Gender", "Age", "Education"]]
in_ind = np.prod(behav_data.notna().values, axis=1)
in_ind = np.ma.make_mask(in_ind)
behav_data = behav_data[in_ind]

category_list = behav_data["Group"].values
age = behav_data["Age"].values
gender = behav_data["Gender"].values
handedness = behav_data["Handedness"].values
education = behav_data["Education"].values

# Fit a GLM for each summary statistic
for metric in ["fo", "lt", "intv", "sr"]:
    print("Running GLM for", metric)

    data = np.load(f"{summary_stats_dir}/{metric}.npy")
    data = data[:, np.newaxis, :]
    data = data[in_ind, :, :]

    # Define dataset for GLM
    data = glm.data.TrialGLMData(
        data=data,
        category_list=category_list,
        age=age,
        gender=gender,
        handedness=handedness,
        education=education,
        num_observations=data.shape[0],
    )

    # Specify regressors and contrasts
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="HC", rtype="Categorical", codes=1)
    DC.add_regressor(name="PD", rtype="Categorical", codes=2)
    DC.add_regressor(name="Gender", rtype="Parametric", datainfo="gender", preproc="z")
    DC.add_regressor(name="Handedness", rtype="Parametric", datainfo="handedness", preproc="z")
    DC.add_regressor(name="Education", rtype="Parametric", datainfo="education", preproc="z")
    DC.add_regressor(name="Age", rtype="Parametric", datainfo="covariate", preproc="z")

    DC.add_simple_contrasts()
    DC.add_contrast(name="HC < PD", values=[-1, 1, 0, 0, 0, 0])

    # Create design martix
    des = DC.design_from_datainfo(data.info)

    # Fit GLM
    model = glm.fit.OLSModel(des, data)

    # Permutation Test Pooling Across States
    contrast = 6  # select "HC < PD"
    metric_perm = "tstats"  # add the t-stats to the null rather than the copes
    nperms = 10000  # for a real analysis 1000+ is best but 100 is a reasonable approximation.
    pooled_dims = 2  # Pool max t-stats across this dimension to correct for multiple comparisons
    nprocesses = 4  # number of parallel processing cores to use

    # Calculate GLM
    perm = glm.permutations.MaxStatPermutation(
        des,
        data,
        contrast,
        nperms,
        metric=metric_perm,
        pooled_dims=pooled_dims,
        nprocesses=nprocesses,
    )

    # Create variables for further plotting
    tstats = model.tstats[contrast]
    thresh = perm.get_thresh([97.5, 99.5])
    sig_mask = abs(tstats) > thresh[0]  # Mask of tstats < .05
    nulls = perm.nulls

    # Get p-values - two sided/undirected
    percentiles = stats.percentileofscore(nulls, tstats)
    pvalues = 1 - percentiles / 100

    # Results for further
    mdict = {
        "tstats": tstats,
        "thresh": thresh,
        "sig_mask": sig_mask,
        "model": model,
        "perm": perm,
        "metric": metric,
        "pvalues": pvalues,
    }
    pickle.dump(mdict, open(f"{summary_stats_dir}/{metric}_GroupComp_GLM.pkl", "wb"))

#%% Plot

def get_group_assignments():
    subIDs = [
        "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14",
        "15", "16", "17", "18", "19", "51", "52", "53", "54", "56", "57", "58", "59",
        "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "101", "102",
        "103", "104", "105", "106", "107", "109", "110", "116", "117", "118", "151",
        "153", "154", "155", "156", "157", "159", "160", "161", "162", "163", "164",
        "165", "166", "167", "168", "169", "170",
    ]
    HCsub = np.array(range(0, subIDs.index("71") + 1))
    PDsub = np.array(range(HCsub[-1] + 1, subIDs.index("170") + 1))
    allSub = np.hstack([HCsub, PDsub])
    assignments = np.ones_like(allSub)
    assignments[allSub > len(HCsub) - 1] = 2
    labels = []
    for a in assignments:
        if a == 1:
            labels.append("HC")
        else:
            labels.append("PD")
    return labels

def save(filename):
    print("Saving", filename)
    plt.savefig(filename)
    plt.close()

def print_significance(tstats, thresh):
    tstats = abs(tstats)
    sig = []
    for tstat in tstats:
        if tstat > thresh[1]:
            sig.append("**")
        elif tstat > thresh[0]:
            sig.append("*")
        else:
            sig.append("-")
    print(" ".join(sig))

# Healthy control or PD patient
assignments = get_group_assignments()

# Load summary stats
fo = np.load(summary_stats_dir + "/fo.npy")
lt = np.load(summary_stats_dir + "/lt.npy")
intv = np.load(summary_stats_dir + "/intv.npy")
sr = np.load(summary_stats_dir + "/sr.npy")

# Create a pandas DataFrame containing the data
# This will help with plotting
ss_dict = {
    "Fractional Occupancy": [],
    "Mean Lifetime (s)": [],
    "Mean Interval (s)": [],
    "Switching Rate (/s)": [],
    "State": [],
    "Group": [],
}
n_subjects, n_states = fo.shape
for subject in range(n_subjects):
    for state in range(n_states):
        ss_dict["Fractional Occupancy"].append(fo[subject, state])
        ss_dict["Mean Lifetime (s)"].append(lt[subject, state])
        ss_dict["Mean Interval (s)"].append(intv[subject, state])
        ss_dict["Switching Rate (/s)"].append(sr[subject, state])
        ss_dict["State"].append(state + 1)
        ss_dict["Group"].append(assignments[subject])
ss_df = pd.DataFrame(ss_dict)

# Get results from the GLM
tstats = {}
thresh = {}
for metric in ["fo", "lt", "intv", "sr"]:
    glm = pickle.load(open(f"{summary_stats_dir}/{metric}_GroupComp_GLM.mat", "rb"))
    tstats[metric] = np.squeeze(glm["tstats"])
    thresh[metric] = np.squeeze(glm["thresh"])

# Plot fractional occupancies
sns.violinplot(data=ss_df, x="State", y="Fractional Occupancy", hue="Group", split=True)
save("plots/fo.png")
print_significance(tstats["fo"], thresh["fo"])

# Plot mean lifetimes
sns.violinplot(data=ss_df, x="State", y="Mean Lifetime (s)", hue="Group", split=True)
save("plots/lt.png")
print_significance(tstats["lt"], thresh["lt"])

# Plot mean intervals
sns.violinplot(data=ss_df, x="State", y="Mean Interval (s)", hue="Group", split=True)
save("plots/intv.png")
print_significance(tstats["intv"], thresh["intv"])

# Plot switching rates
sns.violinplot(data=ss_df, x="State", y="Switching Rate (/s)", hue="Group", split=True)
save("plots/sr.png")
print_significance(tstats["sr"], thresh["sr"])
