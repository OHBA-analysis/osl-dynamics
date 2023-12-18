import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting
from rotation.utils import cov2stdcor, stdcor2cov
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

def perturb_covariances(covariances:np.ndarray,perturbation_factor: float=0.002,random_seed:int=None):
    """
    Perturb the state covariances, the default setting is to only perturb the correlations
    (perserving the standard deviations)
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

    def make_valid_correlation_matrix(matrix):
        """
        Ensure that the input matrix is a valid correlation matrix.

        Args:
            matrix (numpy.ndarray): Square matrix to be validated.

        Returns:
            numpy.ndarray: Valid correlation matrix.
        """
        # Ensure diagonals are 1
        np.fill_diagonal(matrix, 1.0)

        # Ensure the matrix is symmetric
        matrix = 0.5 * (matrix + matrix.T)

        # Ensure all eigenvalues are non-negative
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        matrix = np.dot(eigenvectors, np.dot(np.diag(np.maximum(eigenvalues, 0)), eigenvectors.T))

        return matrix

    stds,cors = cov2stdcor(covariances)
    perturbed_cors = np.zeros_like(cors)

    for i in range(cors.shape[0]):
        # Generate a random perturbation matrix
        perturbation_matrix = np.random.normal(0, perturbation_factor, cors.shape[1:])

        # Add the perturbation to the original correlation matrix
        perturbed_matrix = cors[i] + perturbation_matrix

        # Ensure the resulting matrix is still a valid correlation matrix
        perturbed_matrix = make_valid_correlation_matrix(perturbed_matrix)

        # Store the perturbed matrix in the result array
        perturbed_cors[i] = perturbed_matrix

    return stdcor2cov(stds, perturbed_cors)


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