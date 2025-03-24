import os
import numpy as np
from osl_dynamics import simulation
from osl_dynamics.array_ops import apply_hrf
def hmm_iid(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/hmm_iid/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.9,
        means='zero',
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)
    np.save(f'{save_dir}truth/tpm.npy', sim.hmm.trans_prob)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def hmm_hrf(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/hmm_hrf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.9,
        means='zero',
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)
    np.save(f'{save_dir}truth/tpm.npy', sim.hmm.trans_prob)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', apply_hrf(data[i],tr))
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])


def dynemo_iid(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/dynemo_iid/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.MixedSine_MVN(
        n_samples=n_subjects * n_samples,
        n_modes=n_states,
        n_channels=n_channels,
        relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.2],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1, 2, 3, 4, 6, 8],
        sampling_frequency=250,
        means="zero",
        covariances="random",
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])

def dynemo_hrf(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/dynemo_hrf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.MixedSine_MVN(
        n_samples=n_subjects * n_samples,
        n_modes=n_states,
        n_channels=n_channels,
        relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.2],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1, 2, 3, 4, 6, 8],
        sampling_frequency=250,
        means="zero",
        covariances="random",
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', apply_hrf(data[i],tr))
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])

def swc_iid(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/swc_iid/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.SWC_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        stay_time=100,
        means="zero",
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def swc_hrf(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/swc_hrf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.SWC_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        stay_time=100,
        means="zero",
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', apply_hrf(data[i],tr))
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def soft_mixing(save_dir):
    save_dir = f'{save_dir}/soft_mixing/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    n_subjects = 64
    n_channels = 80
    n_modes = 6
    sim = simulation.MixedSine_MVN(
        n_samples=25600,
        n_modes=n_modes,
        n_channels=n_channels,
        relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.1],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1, 2, 3, 4, 6, 8],
        sampling_frequency=250,
        means="zero",
        covariances="random",
    )

    time_course = sim.mode_time_course
    data = sim.time_series
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_modes)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])

def dynemo_fair(save_dir):
    save_dir = f'{save_dir}/dynemo_fair/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    n_subjects = 64
    n_channels = 80
    n_modes = 6
    sim = simulation.MixedSine_MVN(
        n_samples=25600,
        n_modes=n_modes,
        n_channels=n_channels,
        #relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.2],
        relative_activation=[1.0,1.2,1.4,1.6,1.8,2.0],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
        sampling_frequency=250,
        means="zero",
        covariances="random",
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_modes)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])
def dynemo_ukb(save_dir):
    import pickle
    from osl_dynamics.simulation.mvn import MVN

    save_dir = f'{save_dir}/dynemo_ukb/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    n_subjects = 500
    ukb_dir = './results_final/real/ICA_50_UKB/dynemo_state_6/repeat_1/inf_params/'
    with open(f"{ukb_dir}/alp.pkl", "rb") as f:  # "rb" means read binary mode
        alpha = pickle.load(f)[:n_subjects]
    covariances = np.load(f'{ukb_dir}/covs.npy')

    np.save(f'{save_dir}truth/state_covariances.npy', covariances)

    mvn = MVN(means='zero',covariances=covariances)

    for i in range(n_subjects):
        time_course = alpha[i]
        data = mvn.simulate_data(time_course)
        np.savetxt(f'{save_dir}{10001 + i}.txt', data)
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course)

def dynemo_meg(save_dir):
    import pickle
    from osl_dynamics.simulation.mvn import MVN

    save_dir = f'{save_dir}/dynemo_meg/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    n_subjects = 10
    meg_dir = './dynemo-paper-resting-state/data/'
    with open(f"{meg_dir}/alp.pkl", "rb") as f:  # "rb" means read binary mode
        alpha = pickle.load(f)
    alpha_trimmed = [a[:60000] for a in alpha]
    covariances = np.load(f'{meg_dir}/covs.npy')

    np.save(f'{save_dir}truth/state_covariances.npy', covariances)

    mvn = MVN(means='zero',covariances=covariances)

    for i in range(n_subjects):
        time_course = alpha_trimmed[i]
        data = mvn.simulate_data(time_course)
        np.savetxt(f'{save_dir}{10001 + i}.txt', data)
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course)

def main(simulation_list=None):
    save_dir = './data/node_timeseries/simulation_final/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config = {
        'save_dir': save_dir,
        'n_subjects':500,
        'n_states': 6,
        'n_channels': 50,
        'n_samples':1200,
        'tr':0.72
    }

    if 'hmm_iid' in simulation_list:
        hmm_iid(**config)
    if 'hmm_iid' in simulation_list:
        hmm_hrf(**config)
    if 'dynemo_iid' in simulation_list:
        dynemo_iid(**config)
    if 'dynemo_hrf' in simulation_list:
        dynemo_hrf(**config)
    if 'swc_iid' in simulation_list:
        swc_iid(**config)
    if 'swc_hrf' in simulation_list:
        swc_hrf(**config)

    if 'soft_mixing' in simulation_list:
        soft_mixing(save_dir)
    if 'dynemo_UKB' in simulation_list:
        dynemo_ukb(save_dir)

    if 'dynemo_fair' in simulation_list:
        dynemo_fair(save_dir)

    if 'dynemo_meg' in simulation_list:
        dynemo_meg(save_dir)