"""Predict a continuous value using neuroimaging features.

Note, this script doesn't require osl-dynamics, it uses scikit-learn
to train a machine learning model.

This script should achieve an R2 value of ~0.55.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet

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
#
# This should achieve an R2 value of ~0.5
tde_cov = np.load("../data/static_features/tde_cov.npy")
m, n = np.triu_indices(tde_cov.shape[-1])
X = tde_cov[..., m, n]
'''

# Target variable
y = np.load("../data/age.npy")

# Sanity check the shapes
print(f"X.shape = {X.shape}")
print(f"y.shape = {y.shape}")
print()

#%% Predict age

# Create folds
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# Pipeline for the prediction model
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(whiten=True)),
        ("reg", ElasticNet()),
    ]
)
param_grid = {
    "pca__n_components": [150, 200, 250],
    "reg__alpha": np.logspace(-5, 3, 9),
}

scores = []
for fold, (train_indices, test_indices) in enumerate(kf.split(y)):
    X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    reg = GridSearchCV(pipeline, param_grid, n_jobs=8)
    reg.fit(X_train, y_train)

    score = reg.score(X_test, y_test)
    scores.append(score)

    print(f"Fold {fold}: best_params={reg.best_params_} R2={score}")

print()
print(f"Mean R2: {np.mean(scores)}")
