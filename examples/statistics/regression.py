"""Example script for testing the effect of a continuous variable (e.g. age).

"""

import numpy as np

from osl_dynamics import stats

# Simulate fractional occupancies (subjects, states) that increase with age
# for state 3, and a nuisance covariate (sex)
n_subjects, n_states = 40, 8
age = np.random.uniform(20, 80, size=n_subjects)
sex = np.random.randint(0, 2, size=n_subjects)

fo = np.random.normal(size=(n_subjects, n_states))
fo[:, 2] += 0.05 * (age - age.mean())

# Test for a linear effect of age on each state, controlling for sex, with
# the maximum statistic across states to control the family-wise error rate
result = stats.regression(fo, age, name="Age", covariates={"sex": sex}, n_perm=1000)
print(result)

print("Effect per SD of age:", result.copes.round(3))
print("p-values:", result.pvalues.round(3))
print("Significant states:", np.where(result.significant())[0] + 1)
