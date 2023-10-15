import pickle
import os
import numpy as np

from osl_dynamics.analysis import connectivity
from rotation.utils import group_high_pass_filter

def HMM_training(dataset,n_states,n_channels,save_dir,compute_state=False):
    from osl_dynamics.models.hmm import Config, Model
    # Create a config object
    config = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=600,
        learn_means=True,
        learn_covariances=True,
        batch_size=64,
        learning_rate=1e-3,
        n_epochs=30,
    )
    
    # Initiate a Model class and print a summary
    model = Model(config)
    model.summary()
    
    # Initialization
    init_history = model.random_state_time_course_initialization(dataset, n_epochs=2, n_init=10)
    
    # Full training
    history = model.fit(dataset)
    model.save(save_dir)

    loss_history = history["loss"]
    np.save(f'{save_dir}/loss_history.npy',np.array(loss_history))

    
    # Compute state
    if compute_state:
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}/alpha.pkl', "wb"))
        


def Dynemo_training(dataset, n_modes, n_channels, save_dir,compute_state=False):
    from osl_dynamics.models.dynemo import Config, Model
    # Create a config object
    config = Config(
        n_modes=n_modes,
        n_channels=n_channels,
        sequence_length=100,
        inference_n_units=64,
        inference_normalization="layer",
        model_n_units=64,
        model_normalization="layer",
        learn_alpha_temperature=True,
        initial_alpha_temperature=1.0,
        learn_means=True,
        learn_covariances=True,
        do_kl_annealing=True,
        kl_annealing_curve="tanh",
        kl_annealing_sharpness=5,
        n_kl_annealing_epochs=10,
        batch_size=64,
        learning_rate=0.01,
        n_epochs=30,  # for the purposes of this tutorial we'll just train for a short period
    )

    # Initiate a Model class and print a summary
    model = Model(config)
    model.summary()

    # Initialization
    init_history = model.random_subset_initialization(dataset, n_epochs=2, n_init=10,take=1.0)

    # Full training
    history = model.fit(dataset)
    model.save(save_dir)

    loss_history = history["loss"]
    np.save(f'{save_dir}/loss_history.npy', np.array(loss_history))
    
    # Compute state
    if compute_state:
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}/alpha.pkl', "wb"))


def MAGE_training(dataset, n_modes, n_channels, save_dir, compute_state=False):
    from osl_dynamics.models.mdynemo import Config, Model
    # Create a config object
    config = Config(
        n_modes=n_modes,
        n_channels=n_channels,
        sequence_length=100,
        inference_n_units=64,
        inference_normalization="layer",
        model_n_units=64,
        model_normalization="layer",
        learn_alpha_temperature=True,
        initial_alpha_temperature=1.0,
        learn_means=True,
        learn_stds=True,
        learn_fcs=True,
        do_kl_annealing=True,
        kl_annealing_curve="tanh",
        kl_annealing_sharpness=5,
        n_kl_annealing_epochs=10,
        batch_size=64,
        learning_rate=0.01,
        n_epochs=30,  # for the purposes of this tutorial we'll just train for a short period
    )

    # Initiate a Model class and print a summary
    model = Model(config)
    model.summary()

    # Initialization
    init_history = model.random_subset_initialization(dataset, n_epochs=2, n_init=10, take=1.0)

    # Full training
    history = model.fit(dataset)
    model.save(save_dir)

    loss_history = history["loss"]
    np.save(f'{save_dir}/loss_history.npy', np.array(loss_history))

    # Compute state
    if compute_state:
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}/alpha.pkl', "wb"))

def SWC_computation(dataset,window_length,step_size,save_dir):
    '''

    Parameters
    ----------
    dataset: (osl_dynamics.data.Data): Dataset for training
    window_length: (int) sliding window length
    step_size: (int) step size of sliding window
    save_dir: save directions

    Returns
    -------

    '''

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ts = dataset.time_series()
    filtered_ts = group_high_pass_filter(ts)
    # Calculate the sliding window connectivity
    swc = connectivity.sliding_window_connectivity(filtered_ts, window_length=window_length, step_size=step_size, conn_type="corr")
    np.save(f'{save_dir}/cor_swc.npy',swc,allow_pickle=True)

    swc_cov = connectivity.sliding_window_connectivity(filtered_ts, window_length=window_length, step_size=step_size,
                                                   conn_type="cov")
    np.save(f'{save_dir}/cov_swc.npy', swc_cov, allow_pickle=True)
