"""Example script for comparing three or more groups.

An F-test asks whether any group differs from the others (a one-way ANOVA
with permutations). Pairwise tests then show which groups differ.
"""

import numpy as np

from osl_dynamics import stats

n_per_group = 12
n_features = 10

# Simulate data for three groups
assignments = np.repeat([1, 2, 3], n_per_group)
data = np.random.randn(3 * n_per_group, n_features)

# Group 3 differs in the second and third feature
data[np.ix_(assignments == 3, [1, 2])] += 1.5

# Is there any difference between the groups? (F-test)
result = stats.group_diff(data, assignments, n_perm=1000)
print(result)
print("features with a group difference:", np.argwhere(result.significant()).ravel())

# Which groups differ? Pairwise tests, Bonferroni corrected over the three pairs
pairs = [(1, 2), (1, 3), (2, 3)]
for groups in pairs:
    pairwise = stats.group_diff(data, assignments, groups=groups, n_perm=1000)
    significant = pairwise.pvalues < 0.05 / len(pairs)
    print(f"groups {groups}: features", np.argwhere(significant).ravel())
