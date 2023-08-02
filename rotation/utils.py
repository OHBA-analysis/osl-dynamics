import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from nibabel.nifti1 import Nifti1Image

def parse_index(index:int,models:list,list_channels:list,list_states:list,training:bool=False):
    '''
    This function is used in the array job. Given an index,
    return the model, n_channels, n_states accordingly
    
    Parameters:
    index: (int) the input index in the array job
    models: (list) the model list
    list_channels: (list) the n_channel list
    list_states: (list) the n_state list
    training: (bool) Whether we are in the training mode.
    
    Returns:
        tuple: A tuple containing the following
            - model (string): The model to use
            - n_channels (int): The number of channels to use
            - n_states (int): The number of states to use
    '''

    N_n_channels = len(list_channels)
    N_n_states = len(list_states)
    
    model = models[index // (N_n_channels * N_n_states)]
    index = index % (N_n_channels * N_n_states)

    # For SWC, we do not need to specify n_states
    if (model == 'SWC') & training:
        n_channels = list_channels[index]
        return model, n_channels, None
    
    n_channels = list_channels[index // N_n_states]
    n_states = list_states[index % N_n_states]
    
    return model, n_channels, n_states

def plot_FO(fo_matrix:np.ndarray,plot_dir:str):
    """
    Plot the histogram of fractional occupancy (FO)
    and save the plot to plot_dir
    Parameters
    ----------
    fo_matrix: the fractional occupancy matrix
    plot_dir: the save direction

    Returns
    -------
    """
    n_subj, n_state = fo_matrix.shape

    # Calculate the layout of subplots
    n_cols = n_state // 4

    # Create the subplots
    fig, axes = plt.subplots(n_cols, 4, figsize=(15, 5 * n_cols), sharex=True)

    # Flatten the axes if x=1 to avoid issues with indexing
    #if n_cols == 1:
    axes = np.array(axes).flatten()

    # Iterate over states and plot histograms in each subplot
    for state_idx in range(n_state):
        row = state_idx // 4
        col = state_idx % 4

        ax = axes[row * 4 + col]

        # Plot histogram for the current state
        ax.hist(fo_matrix[:, state_idx], bins=20, edgecolor='black')
        ax.set_title(f'State {state_idx + 1}')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(f'{plot_dir}/fo_hist.jpg')
    plt.savefig(f'{plot_dir}/fo_hist.pdf')
    plt.close()

def stdcor2cov(stds: np.ndarray, corrs:np.ndarray):
    """
    Convert from M standard deviations vectors (N) or diagonal matrices (N*N) and M correlation matrices (N*N)
    to M covariance matrices
    Parameters
    ----------
    stds: np.ndarray
    standard deviation vectors with shape (M, N) or (M,N,N) (diagonal)
    cors: np.ndarray
    correlation matrices with shape (M, N, N)

    Returns
    -------
    covariances: np.ndarray
    covariance matrices with shape (M, N, N)
    """

    if stds.ndim == 2:
        return (np.expand_dims(stds,-1) @ np.expand_dims(stds,-2)) * corrs
    elif stds.ndim == 3:
        return stds @ corrs @ stds
    else:
        raise ValueError('Check the dimension of your standard deviation!')

def cov2stdcor(covs:np.ndarray):
    """
    convert from covariance matrices (M*N*N) to
    stds (M*N) and correlation matrices (M * N * N)
    Parameters
    ----------
    covs: numpy.ndarray (M*N*N)

    Returns
    -------
    stds: numpy.ndarray (M*N)
    corrs: numpy.ndarray (M*N*N)
    """
    # Validation
    if covs.ndim < 2:
        raise ValueError("input covariances must have more than 1 dimension.")

    # Extract batches of standard deviations
    stds = np.sqrt(np.diagonal(covs, axis1=-2, axis2=-1))
    normalisation = np.expand_dims(stds, -1) @ np.expand_dims(stds, -2)
    return stds, covs / normalisation

def first_eigenvector(matrix: np.ndarray):
    """
    Compute the first eigenvector (corresponding to the largest eigenvector)
    of a symmetric matrix
    Parameters
    ----------
    matrix: numpy.ndarray
    N * N. Symmetric matrix.

    Returns
    -------
    eigenvector: numpy.ndarray
    N: the first eigenvector.
    """
    _, eigenvector = eigsh(matrix,k=1,which='LM')
    eigenvector = np.squeeze(eigenvector)
    # Ensure that the returned eigenvector has norm 1
    return eigenvector / (np.linalg.norm(eigenvector) + 1e-10)

def IC2brain(spatial_map:Nifti1Image,IC_metric:np.ndarray):
    """
    Project the IC_metric map to a brain map according to spatial maps of IC components.
    For example, IC_metric can be the mean activation of states, or the rank-one decomposition
    of FC matrix corresponding to different states.
    Parameters
    ----------
    spatial_map: Nifti1Image, represent the spatial map obtained from groupICA
    IC_metric: np.ndarray(K, N),
    where K is the number of independent components, and N is the number of states

    Returns
    -------
    brain_map: Nifti1Image, represent the whole brain map of different states.
    """
    spatial_map_data = spatial_map.get_fdata()
    spatial_map_shape = spatial_map_data.shape
    K, N = IC_metric.shape
    # Assert the matrix multiplication is plausible
    assert spatial_map_shape[-1] == K

    # Implement the multiplication
    brain_map_data = np.matmul(spatial_map_data,IC_metric)
    #Store brain map to a Nifti1Image type
    brain_map = Nifti1Image(brain_map_data, affine=spatial_map.affine, header=spatial_map.header)

    return brain_map

