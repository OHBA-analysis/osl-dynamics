"""
DyNeMo: Long-Range Dependencies Simulation
==========================================

In this tutorial we will train a DyNeMo on simulated data and demonstrate its ability to learn long-range temporal dependencies. This tutorial covers:

1. Simulating Data
2. Training DyNeMo
3. Getting Inferred Model Parameters
4. Sampling from the Generative Model

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/6mrh3>`_ for the expected output.
"""

#%%
# Simulating Data
# ^^^^^^^^^^^^^^^
# 
# Let's start by simulating some training data.
# 
# Hidden Semi-Markov Model
# ************************
# 
# We will simulate long-range temporal structure by using a HSMM. This differs from a vanilla HMM by specifying non-exponential a distribution for lifetimes. This enables us to simulate very long-lived states that would be improbable with an HMM. We can simulate an HSMM with a multivariate normal observation model using the `simulation.HSMM_MVN <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/simulation/hsmm/index.html#osl_dynamics.simulation.hsmm.HSMM_MVN>`_ class in osl-dynamics. This class uses a Gamma distribution for the state lifetimes, which is parameterised with a shape and scale parameter.

from osl_dynamics.simulation import HSMM_MVN

# Simulate the data
sim = HSMM_MVN(
    n_samples=25600,
    n_channels=11,
    n_modes=3,
    means="zero",
    covariances="random",
    observation_error=0.0,
    gamma_shape=10,
    gamma_scale=5,
    random_seed=123,
)

# Standardize the data (z-transform)
sim.standardize()

#%%
# We can access the simulated time series via the `sim.time_series` attribute.

sim_ts = sim.time_series
print(sim_ts.shape)

#%%
# We can see we have the expected number of samples and channels. Now let's examine the simulate state time course.

from osl_dynamics.utils import plotting

# Get the simulated state time course
sim_stc = sim.state_time_course

# Plot
plotting.plot_alpha(sim_stc, n_samples=2000)

#%%
# We can see there are long-lived states with lifetimes of approximately 50 samples, which wouldn't occur with a vanilla HMM.
# 
# Loading into the Data class
# ***************************
# 
# We can create a Data object by simply passing the simulated numpy array to the `Data class <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_.

from osl_dynamics.data import Data

training_data = Data(sim_ts)

#%%
# Training DyNeMo
# ^^^^^^^^^^^^^^^
# 
# Now we have simulated our training data. Let's create a DyNeMo model. We first need to specify a Config object. See the `API reference guide <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/dynemo/index.html#osl_dynamics.models.dynemo.Config>`_ for a list of arguments that can be passed to this class. We will use the following arguments in this tutorial.

from osl_dynamics.models.dynemo import Config

config = Config(
    n_modes=3,
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
    n_kl_annealing_epochs=100,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
)

#%%
# We build a model by passing the Config object to the `Model class <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/models/dynemo/index.html#osl_dynamics.models.dynemo.Model>`_.

from osl_dynamics.models.dynemo import Model

model = Model(config)
model.summary()

#%%
# Now we can train the model by calling the `fit` method. Note, we will also pass `use_tqdm=True`, which will tell the `fit` method to use a tqdm progress bar instead of the default TensorFlow progress bar. This argument is only for visualisation the progress bar, it does not affect the training.

print("Training model:")
model.fit(training_data, use_tqdm=True)

#%%
# Getting the Inferred Parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Now we have trained the model, let's examine the inferred parameters. Let's start with the inferred mixing coefficients.

# DyNeMo inferred alpha
inf_alp = model.get_alpha(training_data)

#%%
# DyNeMo learns a mixture of modes, whereas the ground truth simulation was a state description. Let's see how binarized the inferred mixing coefficients are.

plotting.plot_alpha(inf_alp, n_samples=2000)

#%%
# We can see although DyNeMo is a mixture model, it's able to correctly infer a state description if the ground truth is binary states. Let's binarize the mixing coefficients anyway to obtained mutually exclusive states using `modes.argmax_time_courses <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/inference/modes/index.html#osl_dynamics.inference.modes.argmax_time_courses>`_ function.

from osl_dynamics.inference import modes

inf_stc = modes.argmax_time_courses(inf_alp)

#%%
# There is a trivial identifiability problem with DyNeMo where we can arbitrarily re-order the modes. This means the inferred modes may not be in the same order as the simulation. We can use the `modes.match_modes <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/inference/modes/index.html#osl_dynamics.inference.modes.match_modes>`_ function to re-order the modes to get a better correspondence with the simulation. This function finds the ordering that maximises the Pearson correlation between pairs of time courses.

from osl_dynamics.inference import metrics

# Match the inferred modes to the simulation
sim_stc = sim.state_time_course
_, order = modes.match_modes(sim_stc, inf_stc, return_order=True)
inf_stc = inf_stc[:, order]

# Calculate the dice coefficient between mode time courses
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

#%%
# The dice coefficient can be thought of as inference accuracy, a value of 1 indicates perfect inference. We can see we were able to infer the simulated states very accurately. Let's plot the simualted and inferred state time courses side by side.

plotting.plot_alpha(sim_stc, inf_stc, y_labels=["Ground Truth", "DyNeMo"])

#%%
# Let's also look at the lifetime distribution of the states.

plotting.plot_mode_lifetimes(sim_stc, x_label="Lifetime", y_label="Occurrence")
plotting.plot_mode_lifetimes(inf_stc, x_label="Lifetime", y_label="Occurrence")

#%%
# We see the simulated and inferred lifetime distribution match very well.
# 
# Finally, let's have a look at the simulated and inferred covariances.

import numpy as np

# Ground truth vs inferred covariances
sim_cov = sim.covariances
inf_cov = model.get_covariances()[order]

# Plot
plotting.plot_matrices(np.concatenate([sim_cov, inf_cov]))

#%%
# Again, we see a good correspondence between the simulated and inferred parameters.
# 
# Sampling from the Generative Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# In the previous section, we showed DyNeMo was able to infer hidden states with long-range temporal structure from the data. However, this does not neccessarily mean it is able to generate data with the same long-range temporal structure. To access this we need to sample new data from the trained model. We're interested in the temporal structure in the mixing coefficients. We can sample from the trained model using the `sample_alpha` method. Let's do this.

# Sample from model RNN
sam_alp = model.sample_alpha(2560)

# Hard classify the mode time courses to give mutually exclusive states
sam_stc = modes.argmax_time_courses(sam_alp)

# Plot
plotting.plot_alpha(sam_alp)

#%%
# We can see long lived states similar to the ones used to simulate the training data. Let's plot the lifetime distribution of the samples.

plotting.plot_mode_lifetimes(
    sam_stc,
    x_label="Lifetime",
    x_range=[0, 150],
    y_label="Occurrence",
)

#%%
# We see this resembles the lifetime distribution used to simulate the data, demonstrates DyNeMo was able to learn long-range temporal dependencies in the training data. The distribution is a bit noisy due to the limited length of the sampled time course. Note, if we trained an HMM on the same data, it would not be able generate samples with this lifetime distribution.
# 
# Wrap Up
# ^^^^^^^
# 
# - We have shown how to simulate HSMM data.
# - We trained DyNeMo on HSMM data and showed it was able to learn the temporal structure in the hidden states.
