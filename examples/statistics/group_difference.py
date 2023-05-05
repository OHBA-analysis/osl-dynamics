"""Example script for comparing two groups.

"""

import numpy as np

from osl_dynamics.analysis import statistics

n_subjects_group1 = 24
n_subjects_group2 = 8
n_subjects = n_subjects_group1 + n_subjects_group2
n_features = 10

# Simulate data
group1 = np.random.randn(n_subjects_group1, n_features)
group2 = np.random.randn(n_subjects_group2, n_features)

# There is a difference in the second, third and last feature for group2
group2[:, [1, 2, n_features - 1]] += 1.5

group_data = np.concatenate([group1, group2])

# Labels for group assignment:
# - 1 indicates assignment to group1
# - 2 indicates assignment to group2
assignments = np.ones(n_subjects)
assignments[-n_subjects_group2:] += 1

# Get the p-value for the difference between the features for each group
pvalues = statistics.group_diff_max_stat_perm(group_data, assignments, n_perm=100)
significant = pvalues < 0.05

print("significant features are:", np.squeeze(np.argwhere(significant)))
