"""
Statistical Significance Testing
================================
 
In this tutorial we'll covers how to perform permutation tests for checking whether differences between groups are significant. This tutorial covers:
 
1. Generating Data
2. Permutation testing with a Bonferroni correction for multiple comparisons
3. Maximum statistic permutation testing

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/trfky>`_ for the expected output.
"""

#%%
# Generating Data
# ^^^^^^^^^^^^^^^
# 
# Representing the subjects
# *************************
# 
# It is common to estimate a vector for individual subjects. For example, this vector can be:
# 
# - The power at each parcel/ROI/voxel.
# - The upper or lower triangle of a connectivity matrix, which gives a vector containing all pairwise connection strengths.
# - Power spectral density (PSD) for different frequencies.
# 
# Let's say we have `n` subjects and a `d`-dimensional vector for each subject. This can be stored as a 2D array of shape `(n, d)`. Let's randomly sample values for our vectors using a multivariate normal distribution (zero mean, identity covariance).

import numpy as np

n = 100  # number of subjects
d = 50  # dimensionality of the vector
X = np.random.normal(size=(n, d))
print(X.shape)

#%%
# Separating into two groups
# **************************
# 
# Often the each subject can be assigned to a group. A common analysis is to compare healthy vs diseased groups. Let's say the first 80 subjects are assigned to the first group and the last 20 belong to the second group. We can separate the groups by indexing the `X` array.

# True label for each subject:
# - 0 indices group 1
# - 1 indices group 2
assignments = np.zeros(n)
assignments[-20:] = 1  # assign the last 20 subjects to group 2
print(assignments)

# Select the subjects that belong to group 1
X1 = X[assignments == 0]
print(X1.shape)

# Select the subjects that belong to group 2
X2 = X[assignments == 1]
print(X2.shape)

#%%
# We can see from the shape of the `X1` and `X2` arrays that they contain the expected number of subjects.
# 
# Differences between groups
# **************************
# 
# When we do a statistical significance test to compare groups, what we're doing is seeing if the samples from each group originate from the same or different distributions. When we generated the data we used the same multivariate normal distribution to generate the vectors for each group. Performing a statistical significance test on these vectors should show there's no differences between groups.
# 
# To help illustrate how to perform significance testing, let's change the distribution for the subjects in group 2.

X[assignments == 1] += 1

#%%
# Here, by adding 1 to all of the elements of vectors in group 2, the distribution of data has a mean of 1. This is in contrast to the distribution of data in group 1, where the mean of each element is 0. We show now be able to detect all of the elements in the vectors for group 2 are significantly different to group 1.
#
# Permutation testing with a Bonferroni correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# To compare the groups we'll use a permutation test. This involves randomly shuffling group assignments and calculating a metric (called a **test statistic**). To perform a permutation test, we will do the following:
# 
# 1. Build a **null distribution**, by:
#
#    - Randomly shuffling the real group assignments.
#    - Calculating the mean vector for each group.
#    - Calculating the absolute difference.
#    - We repeat this many times.
#
# 2. The null distribution contains the values the absolute difference between mean vectors that can occur purely due to chance. We want to show our observed difference between the mean vectors of each group lies in an improbable part of this distribution. We do this by specifying a p-value and looking up a threshold. For example, if we want to verify the difference is significant with a p-value < 0.5, we would use the 95th percentile of the null distribution for the threshold.
# 3. An important detail we need to account for is the fact that we're applying a statistical test to each elements of the difference vector simultaneously. The more tests we perform the more false positives we should expect. This is the problem of **multiple comparisons**. A conservative approach or handling this is to use a Bonferroni correction, where we apply a more stringent threshold for a particular p-value. Instead of using the 95th percentile, we use the `100 * (1 - alpha)`th percentile, where `alpha = p_value / m`.
# 4. Finally, we verify if the observed difference vector is above the threshold, if so it is significant. Note, it maybe the case only some of the elements of the vector are significant.
# 
# Let's implement a permutation test with a Bonferroni correction.

def null_distribution(vectors, real_assignments, n_perm):
    # Randomly generate group assignments by shuffling the real assignments
    # Note, for the first permutation we use the real group assignments
    group_assignments = [real_assignments]
    for i in range(n_perm - 1):
        random_assignments = np.copy(real_assignments)
        np.random.shuffle(random_assignments)
        group_assignments.append(random_assignments)

    # Make sure we don't have any duplicate permutations
    group_assignments = np.unique(group_assignments, axis=0)

    # Calculate null distribution
    null = []
    for assignment in group_assignments:
        # Assign subjects to their group
        group1 = vectors[assignment == 0]
        group2 = vectors[assignment == 1]

        # Calculate group means and absolute difference
        mean1 = np.mean(group1, axis=0)
        mean2 = np.mean(group2, axis=0)
        abs_diff = np.abs(mean1 - mean2)
        null.append(abs_diff)

    return np.array(null)

def bonferroni_percentile(p_value, m):
    alpha = p_value / m  # Bonferroni correct significance level
    return 100 * (1 - alpha)

# Generate a null distribution
null = null_distribution(X, assignments, n_perm=1000)

# Calculate a threshold for significance
p_value = 0.05
m = X.shape[-1]  # number of multiple comparisons
percentile = bonferroni_percentile(p_value, m)
thres = np.percentile(null, percentile, axis=0)

# See which elements are significant
X1 = np.mean(X[assignments == 0], axis=0)
X2 = np.mean(X[assignments == 1], axis=0)
abs_diff = np.abs(X1 - X2)
sig = abs_diff > thres

print("Significant elements:")
print(sig)

#%%
# Although we changed the distribution of all elements of the vectors in group 2, we didn't find all of them were significant. This is because the Bonferroni correction is a very stringent test.
# 
# Bonferroni correction with small datasets
# *****************************************
# 
# Note, the Bonferroni correction requires us to be able to resolve the percentiles of the null distribution with a high resolution. For example, if we used a p-value of 0.05 the percentile threshold should be:

bonferroni_percentile(p_value=0.05, m=X.shape[1])

#%%
# For a p-value of 0.01 it should be:

bonferroni_percentile(p_value=0.01, m=X.shape[1])

#%%
# Therefore, we need to be able to pick out the 99.98 percentile from the null distribution, depending on the number of subjects you have you may not be able to resolve the null distribution to this level.
# 
# **The Bonferroni correction approach should be used with caution on small dataset.**
#
# Maximum statistic permutation testing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Another approach that may work better in practice is a maximum statistic permutation test. With a maximum statistic permutation test rather than recording the distribution for each element of the vector separately, we record the maximum value of the absolute difference - we call this the **maximum statistic**. This results in a single distribution that we use to threshold all elements of the difference vector with.
# 
# Let's implement a max stat permutation test.

def null_distribution(vectors, real_assignments, n_perm):
    # Randomly generate group assignments by shuffling the real assignments
    # Note, for the first permutation we use the real group assignments
    group_assignments = [real_assignments]
    for i in range(n_perm - 1):
        random_assignments = np.copy(real_assignments)
        np.random.shuffle(random_assignments)
        group_assignments.append(random_assignments)

    # Make sure we don't have any duplicate permutations
    group_assignments = np.unique(group_assignments, axis=0)

    # Calculate null distribution
    null = []
    for assignment in group_assignments:
        # Assign subjects to their group
        group1 = vectors[assignment == 0]
        group2 = vectors[assignment == 1]

        # Calculate group means and absolute difference
        mean1 = np.mean(group1, axis=0)
        mean2 = np.mean(group2, axis=0)
        abs_diff = np.abs(mean1 - mean2)

        # Keep max stat
        null.append(abs_diff.max())

    return np.array(null)

# Generate a null distribution
null = null_distribution(X, assignments, n_perm=1000)

# Calculate a threshold for significance
p_value = 0.05
thres = np.percentile(null, 100 * (1 - p_value))

# See which elements are significant
X1 = np.mean(X[assignments == 0], axis=0)
X2 = np.mean(X[assignments == 1], axis=0)
abs_diff = np.abs(X1 - X2)
sig = abs_diff > thres

print("Significant elements:")
print(sig)

#%%
# Again, we see not all elements come out as significant. This is due to the limited number of subjects we have, which limits our statistical power. Let's see what happens if we increase the number of subjects to 1000.

# Generate example data
X = np.random.normal(size=(1000,50))
assignments = np.zeros(X.shape[0])
assignments[-200:] = 1

# Add an effect to group 2
X[assignments == 1] += 1

# Calculate threshold for significance
null = null_distribution(X, assignments, n_perm=1000)
p_value = 0.05
thres = np.percentile(null, 100 * (1 - p_value))

# See which elements are significant
X1 = np.mean(X[assignments == 0], axis=0)
X2 = np.mean(X[assignments == 1], axis=0)
abs_diff = np.abs(X1 - X2)
sig = abs_diff > thres

print("Significant elements:")
print(sig)

#%%
# Now we can see all elements are marked as significant.
#
# Wrap Up
# ^^^^^^^
# 
# - We have shown how to perform two types of permutation testing for comparing group differences.
