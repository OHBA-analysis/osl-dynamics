from tqdm import trange
import warnings
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

from osl_dynamics.inference.metrics import riemannian_distance

def parse_index(index:int,models:list,list_channels:list,list_states:list,training:bool=False):
    '''
    This function is used in the array job. Given an index,
    return the model, n_channels, n_states accordingly
    
    Parameters:
    index: int
        the input index in the array job
    models: list
        the model list
    list_channels: list
        the n_channel list
    list_states: list
        the n_state list
    training: bool
        Whether we are in the training mode.
    
    Returns:
    model: string
        The model to use
    n_channels: int
        The number of channels to use
    n_states: int
        The number of states to use
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

def plot_FO(fo_matrix:np.ndarray,plot_dir:str,file_name:str=None):
    """
    Plot the histogram of fractional occupancy (FO)
    and save the plot to plot_dir
    Parameters
    ----------
    fo_matrix: np.ndarray
        the fractional occupancy matrix
    plot_dir: str
        the save direction
    file_name: str
        the file name
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

    if file_name is None:
        file_name = 'fo_hist'
    # Show the plot
    plt.savefig(f'{plot_dir}/{file_name}.jpg')
    plt.savefig(f'{plot_dir}/{file_name}.pdf')
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
    covs: np.ndarray
        shape M*N*N

    Returns
    -------
    stds: np.ndarray
        shape M*N
    corrs: np.ndarray
        shape M*N*N
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

def IC2surface(spatial_map: nib.cifti2.cifti2.Cifti2Image,IC_metric:np.ndarray)->nib.cifti2.cifti2.Cifti2Image:
    """
     Project the IC_metric map to a brain map according to spatial maps of IC components.
    For example, IC_metric can be the mean activation of states, or the rank-one decomposition
    of FC matrix corresponding to different states.
    Different from IC2brain, this function works on CIFTI spatial maps.

    Parameters
    ----------
    spatial_map: Cifti2Image, represent the spatial map obtained from groupICA
    IC_metric: np.ndarray(K, N),
    where K is the number of independent components, and N is the number of states
    Returns
    -------
    new_cifti: Cifti2Image, represent the whole brain map of different states.
    """
    spatial_map_data = spatial_map.get_fdata()
    header = spatial_map.header
    K, N = IC_metric.shape
    new_data = np.matmul(IC_metric.T, spatial_map_data)

    # Define the new axes: the state i(i=1,...,N)
    new_axes = nib.cifti2.cifti2_axes.ScalarAxis([f'State {i + 1}' for i in range(N)])

    # Define the new header using new_axes + axes[1] from old header
    new_header = nib.cifti2.cifti2.Cifti2Header.from_axes((new_axes, header.get_axis(1)))

    # Combine new data and new header to obtain new CIFTI object
    new_cifti = nib.cifti2.cifti2.Cifti2Image(new_data, new_header)

    return new_cifti


def fisher_z_transform(v:np.ndarray)->np.ndarray:
    """
    Fisher z-transform each element of the input
    z = \frac{1}{2}\ln(\frac{1+r}{1-r})
    Parameters
    ----------
    v: (numpy.ndarray) Elements in [-1,1]

    Returns
    -------
    Fisher z-transformed matrix
    """
    return 0.5 * np.log((1 + v)/(1 - v))
def fisher_z_correlation(M1:np.ndarray,M2:np.ndarray)->float:
    """
    Compute the Fisher z-transformed vector correlation.
    Fisher z-transformation: z = \frac{1}{2}\ln(\frac{1+r}{1-r})
    Suppose M1 and M2 are in shape N*N, semi-positive definite
    Step 1: Obtain the lower triangular element of M1 and M2,
    unwrap to vectors v1, v2 with length N * (N - 1) / 2
    Step 2: Fisher z-transform vectors v1, v2 to z1, z2
    Step 3: return the Pearson's correlation between z1, z2
    Parameters
    ----------
    M1: (numpy.ndarray) correlation/matrix 1
    M2: (numpy.ndarray) matrix 2

    Returns
    -------
    dist: (float) distance between M1 and M2
    """
    N = M1.shape[0]
    assert N == M2.shape[0]

    # Obtain the upper triangular elements
    upper_indices = np.triu_indices(N, k=1)
    v1 = M1[upper_indices]
    v2 = M2[upper_indices]

    # Fisher-z-transformation
    z1 = fisher_z_transform(v1)
    z2 = fisher_z_transform(v2)

    # return the correlation
    return np.cov(z1,z2,ddof=0)[0,1]/ (np.std(z1,ddof=0) * np.std(z2,ddof=0))

def pairwise_fisher_z_correlations(matrices:np.ndarray)->np.ndarray:
    """
    Compute the pairwise Fisher z-transformed correlations of matrices
    See function fisher_z_correlation for details
    Parameters
    ----------
    matrices: numpy.ndarray with shape M*N*N

    Returns
    -------
    correlations: numpy.ndarray with shape M * M
    """
    N = len(matrices)
    correlation_metrics = np.eye(N)
    for i in trange(N,desc='Compute Fisher-z transformated correlation'):
        for j in range(i + 1, N):
            correlation_metrics[i][j] = fisher_z_correlation(
                matrices[i], matrices[j]
            )
            correlation_metrics[j][i] = correlation_metrics[i][j]

    return correlation_metrics

def high_pass_filter(data:np.ndarray,T:float,cutoff_frequency:float,order:int)->np.ndarray:
    """

    Parameters
    ----------
    data: (np.ndarray) N_channels*N_timepoints
    T: (float) temporal resolution of the signal
    cutoff_frequency: (float) cutoff frequency of the filter
    order: (int) order of the filter
    Returns
    -------
    filtered_data: np.ndarray, having the same format as data
    """
    from scipy.signal import butter,filtfilt
    N = data.shape[1]
    fs = 1 / T # Sampling frequency
    nyquist = 0.5 * fs #Nyquist frequency
    b_butter,a_butter = butter(order,cutoff_frequency,btype="highpass",fs=fs)
    filtered_data = filtfilt(b_butter,a_butter,data)

    return filtered_data

def group_high_pass_filter(ts:list,T:float=0.7,cutoff_frequency:float=0.25,order:int=16) -> list:
    """
    Apply high pass filter to the group time series
    Parameters
    ----------
    ts: (list) group time series, each element should be a np.ndarray (N_timepoint, N_channels)
    T: (float) temporal resolution of the signal
    cutoff_frequency: (float) cutoff frequency of the filter
    order: (int) order of the high-pass filter

    Returns
    -------
    returned_ts: (list) filtered signal with the same format as ts
    """
    returned_ts = []
    for i in trange(len(ts),desc='high pass filtering signals'):
        returned_ts.append(high_pass_filter(ts[i].T,
                                            T=T,
                                            cutoff_frequency=cutoff_frequency,
                                            order=order).T
                           )

    return returned_ts
def regularisation(matrices:np.ndarray,eps:float=1e-6)->np.ndarray:
    """
    Regularise a bunch of matrices by adding eps to the diagonal
    Parameters
    ----------
    matrices: np.ndarray with shape (M,N,N) or (N,N)
    eps: float

    Returns
    -------
    regularised_matrices: np.ndarray with shape (M,N,N) or (N,N)
    """
    if matrices.ndim == 2:
        N = len(matrices)
    elif matrices.ndim == 3:
        _,N,_ = matrices.shape
    else:
        return ValueError('Matrices dimension is incorrect!')

    return matrices + np.eye(N) * eps

def twopair_vector_correlation(vectors_1:np.ndarray,vectors_2:np.ndarray)->np.ndarray:
    """
    Compute the pairwise correlation \rho_{ij}
    between ith vectors_1 and jth vectors_j
    Parameters
    ----------
    matrices_1: (M1*N), M1 is the number of vectors,
     N is the vector length
    marices_2: (M2*N) M2 is the number of vectors,
     N is the vector length

    Returns
    -------
    correlation_map: M1*M2
    """
    M1, N1 = vectors_1.shape
    M2, N2 = vectors_2.shape
    assert N1 == N2
    correlations = np.empty((M1, M2))

    for i in range(M1):
        for j in range(M2):
            correlation = np.corrcoef(vectors_1[i, :], vectors_2[j, :])[0, 1]
            correlations[i, j] = correlation
    return correlations

def twopair_riemannian_distance(matrices_1:np.ndarray,matrices_2:np.ndarray,eps_value:list=[0,1e-9,1e-8,1e-7,1e-6,1e-5],threshold:float=1e-3)->np.ndarray:
    """
    Compute the Riemannian distance between two sets of matrices
    Parameters
    ----------
    matrices_1: (np.ndarray) M1*N*N, M1 is the number of matrices,N*N is the matrix size
    matrices_2: (np.ndarray) M2*N*N, M2 is the number of matrices,N*N is the matrix size
    eps: (list) regularisation factor
    threshold: (float) Threshold to apply when there are negative eigenvalues. Must be positive.
    Returns
    riemannian_distances: (M1*M2) two pair riemannian distance
    -------
    """
    M1,N1,P1 = matrices_1.shape
    M2,N2,P2 = matrices_2.shape
    assert N1 == P1
    assert N1 == N2
    assert P1 == P2

    matrices_1.astype(np.float64)
    matrices_2.astype(np.float64)
    riemannian_distances = np.zeros([M1,M2])
    for eps in eps_value:
        try:
            for i in trange(M1, desc="Computing Riemannian distances"):
                for j in range(M2):
                    riemannian_distances[i][j] = riemannian_distance(
                        regularisation(matrices_1[i],eps),
                        regularisation(matrices_2[j],eps),
                        threshold=threshold
                    )
            return riemannian_distances
        except np.linalg.LinAlgError:
            warnings.warn(f'Calculation of Riemannian distance failed for eps={eps}')

    raise ValueError('Riemannian distance failed for all eps values!')

def twopair_fisher_z_transformed_correlation(matrices_1:np.ndarray,matrices_2:np.ndarray)->np.ndarray:
    """
    Compute the Fisher_z_transformed correlation between two sets of matrices
    Parameters
    ----------
    matrices_1:  (np.ndarray) M1*N*N, M1 is the number of matrices,N*N is the matrix size
    matrices_2:  (np.ndarray) M2*N*N, M2 is the number of matrices,N*N is the matrix size

    Returns
    fisher_z_transformed_correlation: (M1*M2) two pair fisher_z_correlation
    -------
    """
    M1 = len(matrices_1)
    M2 = len(matrices_2)
    fisher_z_transformed_correlation = np.zeros([M1,M2])
    for i in trange(M1, desc='Compute twopair Fisher-z transformated correlation'):
        for j in range(M2):
            fisher_z_transformed_correlation[i][j] = fisher_z_correlation(
                matrices_1[i], matrices_2[j]
            )

    return fisher_z_transformed_correlation


def hungarian_pair(matrix:np.ndarray,distance:bool=False):
    """
    Use the hungarian algorithm to pair different "states" according to the matrix
    Parameters
    ----------
    matrix: (M*M) distance/correlation matrix as the measure for pairing
    distance: (str) True: matrix measures distance.
                    False: matrix measures correlation

    Returns
    -------

    """
    if distance:
        row_indices, col_indices = linear_sum_assignment(matrix)
    else:
        row_indices, col_indices = linear_sum_assignment(np.max(matrix) - matrix)

    indices = {'row':row_indices.tolist(),'col':col_indices.tolist()}
    matrix_reordered = matrix[row_indices,:]
    matrix_reordered = matrix_reordered[:,col_indices]
    return indices, matrix_reordered
def heatmap_reorder_matrix(matrix:np.ndarray,plot_dir:str,plot_statistics:str,indices:dict,model_name:str,n_channels:int,n_states:int,split_strategy:str=''):
    """
    Plot the reordered matrix while putting the row and column indices onto the x and y ticks

    Parameters
    ----------
    matrix: np.ndarray
        matrix to plot
    plot_dir: str
        the directory to save the plot
    plot_statistics: str
        'means_correlation' or 'FCs_distance'
    indices: dict
        indices with keys: 'row' and 'col'
    model_name: str
        model name
    n_channels: int
        number of channels
    n_states: int
        number of states
    split_strategy: str
        Specific to split half plot, split_strategy = '1','2','3','4'

    Returns
    -------
    """

    # When plotting, change the model name from MAGE to mDynemo
    if model_name == 'MAGE':
        model_name = 'mDynemo'

    # Set up the figure and axis
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font size for labels

    # Create a heatmap of the correlation matrix
    sns.heatmap(matrix, cmap="coolwarm", square=True,
                linewidths=.5, cbar_kws={"shrink": 0.75}, fmt=".2f")
    plt.xticks(np.arange(len(indices['col'])) + 0.5, indices['col'], fontsize=13)
    plt.yticks(np.arange(len(indices['row'])) + 0.5, indices['row'], fontsize=13)

    plt.title(f'{model_name}_ICA_{n_channels}_state_{n_states}_split_{split_strategy}, {plot_statistics}', fontsize=20)
    if split_strategy is not None:
        plt.savefig(f'{plot_dir}{plot_statistics}_plot_split_{split_strategy}.jpg')
        plt.savefig(f'{plot_dir}{plot_statistics}_plot_split_{split_strategy}.pdf')
    else:
        plt.savefig(f'{plot_dir}{plot_statistics}_plot.jpg')
        plt.savefig(f'{plot_dir}{plot_statistics}_plot.pdf')
    plt.close()
