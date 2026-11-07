"""Example script for comparing conditions measured in the same subjects.

A within-subject design: the conditions are shuffled within each subject.
With three conditions an F-test asks whether any condition differs; pairwise
tests then show which.
"""

import numpy as np

from osl_dynamics import stats

n_subjects = 15
n_conditions = 3
n_features = 10

# Simulate data for each subject and condition, with subject differences
data = np.random.randn(n_subjects, n_conditions, n_features)
data += np.random.randn(n_subjects, 1, 1) * 2

# Condition 2 differs in the first two features
data[:, 2, :2] += 1.0

# Is there any difference between the conditions? (F-test)
result = stats.condition_diff(data, n_perm=1000)
print(result)
print("features with a condition difference:", np.argwhere(result.significant()).ravel())

# Which conditions differ? Condition 2 minus condition 0
pairwise = stats.condition_diff(data, conditions=(2, 0), n_perm=1000)
print("condition 2 - 0, significant features:", np.argwhere(pairwise.significant()).ravel())

# Effect size of the pairwise difference with a bootstrap confidence interval
effect = pairwise.effect_size(n_boot=1000)
print("Cohen's d:", np.round(effect.d, 2))
print("95% CI:", np.round(effect.d_ci, 2))
