"""
Training Models
===============
This is a tutorial that covers how to train an osl-dynamics model. In this tutorial we use simulated data.
For scripts that use real neuroimaging data see the `examples directory <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples>`_.

"""

#%%
# We start by importing the necessary packages. In this tutorial we will train the DyNeMo model.

from osl_dynamics import data
from osl_dynamics.models.dynemo import Config, Model

#%%
# The Config Object
# ^^^^^^^^^^^^^^^^^
#
# Model hyperparameters are all contained in the Config object. The API reference for each model lists the attributes for each model's Config object.
# 
# In this example we will use the following configuration.

config = Config(
    n_modes=5,
    n_channels=11,
    sequence_length=200,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

#%%
# Notice we have set `learn_means=False`, which means we are forcing the mean to be zero in our observation model.

#%%
# Training Data
# ^^^^^^^^^^^^^^^^^^^^^^
#
# In the `Data Object and Preparation` tutorial we saved some simulated data to a numpy file `X.npy`. Let's use this data to train our model.
# We load this data using the Data object and prepare a TensorFlow training dataset:

training_data = data.Data("X.npy")
training_data.prepare()
training_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
)

#%%
# Build and Train a Model
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# We build a model using the Model class and Config object:

model = Model(config)

#%%
# We can treat the model object as a normal TensorFlow Keras Model object, e.g. to view a summary, we can use

model.summary()

#%%
# To train the model we use the `fit` method:

model.fit(training_dataset, epochs=config.n_epochs)

#%%
# Saving Trained Models
# ^^^^^^^^^^^^^^^^^^^^^
#
# To save a trained model we can use the `save_weights` method:

model.save_weights("trained_model/weights")

#%%
# See the 'Analyse Trained Models' tutorial for how to load a trained model.
