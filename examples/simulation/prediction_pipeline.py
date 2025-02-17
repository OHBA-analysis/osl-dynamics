import numpy as np

from osl_dynamics.analysis import prediction

# Create a pipeline builder
pipeline_builder = prediction.PipelineBuilder()

# Print the list of available scalers
print("available scalers: ", pipeline_builder.available_scalers)

# Print the list of available dimensionality reduction techniques
print("available dim reductions: ", pipeline_builder.available_dim_reductions)

# Print the list of available predictors
print("available predictors: ", pipeline_builder.available_predictors)

# Get the model (Standard Scaler, Elastic Net and no dimensionality reduction)
model = pipeline_builder.build_model(scaler="standard", predictor="elastic_net")

# Now get the params grid for the model
params_grid = pipeline_builder.get_params_grid(
    predictor_params={
        "alpha": np.logspace(-3, 3, 7),
        "l1_ratio": np.linspace(0.1, 0.999, 10),
    }
)


# Now simulate some random data
n_samples = 5000
n_features = 10
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

# The model is a pipeline, so we can fit it as usual
model.fit(X, y)
print("Score of fitting model with default parameters: ", model.score(X, y))

# We can also use the model selection class to perform a grid search and nested cross-validation
model_selection = prediction.ModelSelection(model=model, params_grid=params_grid)

# Grid search
model_selection.model_selection(X, y)
print(
    "Score of best model after grid search: ",
    model_selection.cross_validation_scores(X, y),
)
# best model after grid search can be accessed with model_selection.best_model
# best model parameters can be accessed with model_selection.best_params

# Nested cross-validation
outer_score = model_selection.nested_cross_validation(X, y)
print("Score of nested cross-validation: ", outer_score)
