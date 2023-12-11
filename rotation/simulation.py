import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

def perturb_covariances(covariances:np.ndarray,perturbation_factor: float=0.002,random_seed:int=None):
    """
    Perturb the state covariances
    Parameters
    ----------
    covariances: np.ndarray
        The shape is n_states, n_channels, n_channels. The covariances matrix to perturb
    perturbation_factor: float
        Control the scaling of perturbation.
    random_seed: int
        Random seed to generate the perturbation.

    Returns
    -------
    perturbed_covariances: np.ndarray
        Shape is the same as covariances. Perturbed covariances.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    perturbed_covariances = np.zeros_like(covariances)

    for i in range(covariances.shape[0]):
        # Generate a random perturbation matrix
        perturbation_matrix = np.random.normal(0, perturbation_factor, covariances.shape[1:])

        # Add the perturbation to the original covariance matrix
        perturbed_cov_matrix = covariances[i] + perturbation_matrix

        # Ensure the resulting matrix is still positive definite
        perturbed_cov_matrix = np.dot(perturbed_cov_matrix, perturbed_cov_matrix.T)

        # Store the perturbed matrix in the result array
        perturbed_covariances[i] = perturbed_cov_matrix

    return perturbed_covariances
def HMM_single_subject_simulation(save_dir:str, n_scans:int, n_states:int, n_samples:int,n_channels:int,
                                  trans_prob:np.ndarray,means:np.ndarray,covariances:np.ndarray,subj_name:str='10001'):
    tf_ops.gpu_growth()
    time_series = []
    state_time_course = []
    for i in range(n_scans):
        sim = simulation.HMM_MVN(
            n_samples=n_samples,
            n_states=n_states,
            n_channels=n_channels,
            trans_prob=trans_prob,
            means=means,
            covariances=covariances,
        )
        time_series.append(sim.time_series)
        state_time_course.append(sim.state_time_course)
    time_series = np.concatenate(time_series,axis=0)
    state_time_course = np.concatenate(state_time_course)
    np.savetxt(f'{save_dir}{subj_name}.txt',time_series)
    np.save(f'{save_dir}{subj_name}_state_time_course.npy',state_time_course)
    np.save(f'{save_dir}{subj_name}_state_covariances.npy', covariances)