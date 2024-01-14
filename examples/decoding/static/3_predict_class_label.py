"""Predict a class label using neuroimaging features.

Note, this script doesn't require osl-dynamics, it uses scikit-learn
to train a machine learning model.

This script should achieve an accuracy of ~0.78.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

#%% Load data

# Input features
power = np.load("../data/static_features/power.npy")
aec = np.load("../data/static_features/aec.npy")

# Only keep upper triangle of AEC
m, n = np.triu_indices(aec.shape[-1], k=1)
aec = aec[..., m, n]

X = np.concatenate([power, aec], axis=-1)
X = X.reshape(X.shape[0], -1)

'''
# Use TDE covariances as input features
tde_cov = np.load("../data/static_features/tde_cov.npy")
m, n = np.triu_indices(tde_cov.shape[-1])
X = tde_cov[..., m, n]
'''

# Target variable
age = np.load("../data/age.npy")

# Separate into a young (<=53) and old (>53) group
y = np.zeros_like(age)
y[age > 53] = 1

# Sanity check the shapes
print(f"X.shape = {X.shape}")
print(f"y.shape = {y.shape}")
print()

#%% Predict young vs old

# Create folds
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# Pipeline for the prediction model
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(whiten=True)),
        ("clf", LogisticRegression()),
    ]
)
param_grid = {
    "pca__n_components": [150, 200, 250],
}

scores = []
for fold, (train_indices, test_indices) in enumerate(kf.split(y)):
    X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    clf = GridSearchCV(pipeline, param_grid, n_jobs=8)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    scores.append(score)

    print(f"Fold {fold}: best_params={clf.best_params_} accuracy={score}")

print()
print(f"Mean accuracy: {np.mean(scores)}")
