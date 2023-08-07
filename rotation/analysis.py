import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import nibabel as nib

from osl_dynamics.array_ops import cov2corr
import osl_dynamics.data
from osl_dynamics.inference.metrics import pairwise_frobenius_distance,\
    pairwise_matrix_correlations, pairwise_riemannian_distances, pairwise_congruence_coefficient
from rotation.utils import plot_FO, stdcor2cov,cov2stdcor, first_eigenvector, IC2brain

def construct_graph(tpm:np.ndarray):
    """
    Construct the graph based on input matrix
    Parameters
    ----------
    tpm: transition probability matrix

    Returns
    -------
    G: (networkx.classes.digraph.DiGraph)
    """

    # Work with directed weighted graph
    G = nx.DiGraph()

    for i in range(len(tpm)):
        G.add_node(i)  # Add nodes to the graph

        for j in range(len(tpm)):
            weight = tpm[i][j]
            if (weight > 0) & (i != j):
                G.add_edge(i, j, weight=weight)

    return G
def HMM_analysis(dataset:osl_dynamics.data.Data, save_dir:str,
                 spatial_map_dir:str,n_channels:int, n_states:int):
    """
    Post-training analysis of HMM model
    Parameters
    ----------
    dataset: (osl_dynamics.data.Data) dataset to work on
    save_dir: (str) directory to save results
    spatial_map_dir: (str) directory of groupICA spatial maps
    n_channels: (int) number of channels
    n_states: (int) number of states

    Returns
    -------
    """
    from osl_dynamics.models import load
    from osl_dynamics.utils import plotting
    from osl_dynamics.inference import modes

    model = load(save_dir)

    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))

    # Summary statistics analysis
    plot_dir = f'{save_dir}plot/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        # Plot the state probability time course for the first subject
        with open(f'{save_dir}alpha.pkl', 'rb') as file:
            alpha = pickle.load(file)

        plotting.plot_alpha(alpha[0], n_samples=1200)
        plt.savefig(f'{plot_dir}state_prob_example.jpg')
        plt.savefig(f'{plot_dir}state_prob_example.pdf')
        plt.close()

        # Hard classify the state probabilities
        stc = modes.argmax_time_courses(alpha)
        # Plot the state time course for the first subject (8 seconds)
        plotting.plot_alpha(stc[0], n_samples=1200)
        plt.savefig(f'{plot_dir}state_hard_example.jpg')
        plt.savefig(f'{plot_dir}state_hard_example.pdf')
        plt.close()

        # Calculate fractional occupancies
        fo = modes.fractional_occupancies(stc)
        np.save(f'{save_dir}fo.npy', fo)
        # Plot the distribution of fractional occupancy (FO) across subjects
        plotting.plot_violin(fo.T, x_label="State", y_label="FO")
        plt.savefig(f'{plot_dir}fo_violin.jpg')
        plt.savefig(f'{plot_dir}fo_violin.pdf')
        plt.close()

        # Calculate mean lifetimes (in seconds)
        mlt = modes.mean_lifetimes(stc, sampling_frequency=1 / 0.7)
        np.save(f'{save_dir}mlt.npy', mlt)
        # Plot distribution across subjects
        plotting.plot_violin(mlt.T, x_label="State", y_label="Mean Lifetime (s)")
        plt.savefig(f'{plot_dir}mlt_violin.jpg')
        plt.savefig(f'{plot_dir}mlt_violin.pdf')
        plt.close()

        # Calculate mean intervals (in seconds)
        mintv = modes.mean_intervals(stc, sampling_frequency=1 / 0.7)
        np.save(f'{save_dir}mintv.npy', mintv)
        # Plot distribution across subjects
        plotting.plot_violin(mintv.T, x_label="State", y_label="Mean Interval (s)")
        plt.savefig(f'{plot_dir}mintv_violin.jpg')
        plt.savefig(f'{plot_dir}mintv_violin.pdf')
        plt.close()

    # Analyze the transition probability matrix
    # using Louvain community detection algorithm
    if not os.path.isfile(f'{save_dir}tpm_partition.pkl'):
        tpm = np.load(f'{save_dir}trans_prob.npy')
        G = construct_graph(tpm)
        partition = nx.community.louvain_communities(G)
        print('The final partition is: ', partition)

        with open(f'{save_dir}tpm_partition.pkl', 'wb') as file:
            pickle.dump(partition, file)

    # Obtain the statistics (mean,std,cov,cor) of states
    if not os.path.isfile(f'{save_dir}state_covariances.npy'):
        extract_state_statistics(save_dir,model_name='HMM')

    # Plot the statistics (mean,std,cor) of states
    if not os.path.isfile(f'{plot_dir}plot_state_correlations.pdf'):
        plot_state_statistics(save_dir,plot_dir,
                              model_name='HMM',
                              n_channels=n_channels,
                              n_states=n_states
                              )

    # Analyze the distance between different states/modes
    dist_dir = f'{save_dir}distance/'
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    if not os.path.isfile(f'{dist_dir}congruence_coefficient_cor.npy'):
        compute_distance(save_dir,dist_dir)

    # Plot the distance between different states/modes
    if not os.path.isfile(f'{plot_dir}/distance_plot_cor.pdf'):
        plot_distance(dist_dir,plot_dir,
                      model='HMM',
                      n_channels=n_channels,
                      n_states=n_states)

    # Fractional occupancy analysis
    FO_dir = f'{save_dir}FO_analysis/'
    if not os.path.exists(FO_dir):
        os.makedirs(FO_dir)
        fo_matrix = np.load(f'{save_dir}fo.npy')
        n_subj, n_state = fo_matrix.shape
        fo_corr = np.corrcoef(fo_matrix.T)
        np.save(f'{save_dir}fo_corr.npy',fo_corr)

        # Plot the FO distribution of each state
        plot_FO(fo_matrix,FO_dir)

        from scipy.spatial.distance import squareform
        # Convert correlation matrix to 1D condensed distance
        # Do not check symmetry because there might be round-off
        # errors in fo_corr.
        fo_dist = squareform(1 - fo_corr,checks=False)

        import scipy.cluster.hierarchy as sch
        Z = sch.linkage(fo_dist, method='ward',optimal_ordering=True)
        np.save(f'{save_dir}cluster_Z.npy',Z)

        # Plot the dendrogram
        plt.figure(figsize=(10, 5))
        sch.dendrogram(Z)
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.title('Dendrogram')
        plt.savefig(f'{FO_dir}dendrogram_FO.jpg')
        plt.savefig(f'{FO_dir}dendrogram_FO.pdf')
        plt.close()

    # Compute the mean activation map
    if not os.path.isfile(f'{save_dir}mean_activation_map.nii.gz'):
        mean_mapping(save_dir,spatial_map_dir)

    # Compute rank-one decomposition of FC map
    if not os.path.isfile(f'{save_dir}FC_map.nii.gz'):
        FC_mapping(save_dir,spatial_map_dir)




def Dynemo_analysis(dataset:osl_dynamics.data.Data, save_dir:str,
                    spatial_map_dir:str, n_channels:int,n_states:int):
    """
    Post-training analysis of Dynemo model
    Parameters
    ----------
    dataset: (osl_dynamics.data.Data) dataset to work on
    save_dir: (str) directory to save results
    spatial_map_dir: (str) directory of spatial maps
    n_channels: (int) number of channels
    n_states: (int) number of states

    Returns
    -------
    """
    from osl_dynamics.models import load
    model = load(save_dir)

    # Specify plot directory
    plot_dir = f'{save_dir}plot/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))

    # Obtain the statistics (mean,std, cov,cor) of states
    if not os.path.isfile(f'{save_dir}state_covariances.npy'):
        extract_state_statistics(save_dir, model_name='Dynemo')

    # Plot the statistics (mean,std,cor) of states
    if not os.path.isfile(f'{plot_dir}plot_state_correlations.pdf'):
        plot_state_statistics(save_dir, plot_dir,
                              model_name='Dynemo',
                              n_channels=n_channels,
                              n_states=n_states
                              )

    # Analyze the distance between different states/modes
    dist_dir = f'{save_dir}distance/'
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    if not os.path.isfile(f'{dist_dir}congruence_coefficient_cor.npy'):
        compute_distance(save_dir, dist_dir)

    # Plot the distance between different states/modes
    if not os.path.isfile(f'{plot_dir}/distance_plot_cor.pdf'):
        plot_distance(dist_dir, plot_dir,
                      model='Dynemo',
                      n_channels=n_channels,
                      n_states=n_states)

    # Compute the mean activation map
    if not os.path.isfile(f'{save_dir}mean_activation_map.nii.gz'):
        mean_mapping(save_dir,spatial_map_dir)

    if not os.path.isfile(f'{save_dir}FC_map.nii.gz'):
        FC_mapping(save_dir,spatial_map_dir)

def MAGE_analysis(dataset:osl_dynamics.data.Data, save_dir:str,
                  spatial_map_dir:str, n_channels:int, n_states:int):
    """
    Post-training analysis of MAGE
    Parameters
    ----------
    dataset: (osl_dynamics.data.Data) dataset to work on
    save_dir: (str) directory to save results
    spatial_map_dir: (str) directory of spatial maps
    n_channels: (int) number of channels
    n_states: (int) number of states

    Returns
    -------

    """
    from osl_dynamics.models import load
    model = load(save_dir)

    # Specify plot directory
    plot_dir = f'{save_dir}plot/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))

    # Obtain the statistics (mean,std, cov,cor) of states
    if not os.path.isfile(f'{save_dir}state_covariances.npy'):
        extract_state_statistics(save_dir, model_name='MAGE')

    # Plot the statistics (mean,std,cor) of states
    if not os.path.isfile(f'{plot_dir}plot_state_correlations.pdf'):
        plot_state_statistics(save_dir, plot_dir,
                              model_name='MAGE',
                              n_channels=n_channels,
                              n_states=n_states
                              )

    # Analyze the distance between different states/modes
    dist_dir = f'{save_dir}distance/'
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    if not os.path.isfile(f'{dist_dir}congruence_coefficient_cor.npy'):
        compute_distance(save_dir, dist_dir)

    # Plot the distance between different states/modes
    if not os.path.isfile(f'{plot_dir}/distance_plot_cor.pdf'):
        plot_distance(dist_dir, plot_dir,
                      model='MAGE',
                      n_channels=n_channels,
                      n_states=n_states)


    # Compute the mean activation map
    if not os.path.isfile(f'{save_dir}mean_activation_map.nii.gz'):
        mean_mapping(save_dir,spatial_map_dir)

    if not os.path.isfile(f'{save_dir}FC_map.nii.gz'):
        FC_mapping(save_dir,spatial_map_dir)
def SWC_analysis(save_dir,old_dir,n_channels,n_states):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    swc = np.load(f'{old_dir}/fc_swc.npy')

    swc = np.concatenate(swc)

    # Initiate the K-means class
    kmeans = KMeans(n_clusters=n_states, verbose=0)

    # Get indices that correspond to an upper triangle of a matrix
    # (not including the diagonal)
    i, j = np.triu_indices(n_channels, k=1)

    # Now let's convert the sliding window connectivity matrices to a series of vectors
    swc_vectors = swc[:, i, j]

    # Check the shape of swc vectors
    print(f'SWC vectors shape: {swc_vectors.shape}')

    # Fitting
    kmeans.fit(swc_vectors)  # should take a few seconds

    centroids = kmeans.cluster_centers_
    print(f'centroids shape: {centroids.shape}')

    # Convert from a vector to a connectivity matrix
    kmean_networks = np.empty([n_states, n_channels, n_channels])
    kmean_networks[:, i, j] = centroids
    kmean_networks[:, j, i] = centroids

    np.save(f'{save_dir}/kmean_networks.npy',kmean_networks)

def extract_state_statistics(save_dir:str,model_name:str):
    from osl_dynamics.models import load
    model = load(save_dir)
    if model_name == 'MAGE':
        means, stds, correlations = model.get_means_stds_fcs()
        # Compact stds (M*N*N) to (M*N)
        if stds.ndim == 3:
            stds = np.array([np.diag(matrix) for matrix in stds])
        covariances = stdcor2cov(stds, correlations)
    else:
        means, covariances = model.get_means_covariances()
        stds, correlations = cov2stdcor(covariances)
    np.save(f'{save_dir}state_means.npy', means)
    np.save(f'{save_dir}state_stds.npy', stds)
    np.save(f'{save_dir}state_correlations.npy', correlations)
    np.save(f'{save_dir}state_covariances.npy', covariances)

def plot_state_statistics(save_dir:str, plot_dir:str,model_name:str,n_channels:int,n_states:int):
    """
    plot the mean, std, correlation of states
    Parameters
    ----------
    save_dir: (str) directory to read in mean,std,correlations
    plot_dir: (str) directory to save the plot
    model_name: (str) model name
    n_channels: (int) number of channels
    n_states: (int) number of states

    Returns
    -------

    """
    means = np.load(f'{save_dir}state_means.npy')
    stds = np.load(f'{save_dir}state_stds.npy')
    corrs = np.load(f'{save_dir}state_correlations.npy')

    # means box plot
    fig, ax = plt.subplots(figsize=(10,6))
    boxplot = ax.boxplot(means.T,vert=True)
    ax.set_xticklabels([f'{i+1}'for i in range(n_states)])
    ax.set_xlabel('State',fontsize=12)
    ax.set_ylabel(r'$\mu$')
    plt.tight_layout()
    plt.suptitle(f'Mean, {model_name}_ICA_{n_channels}_state_{n_states}',fontsize=15)
    plt.savefig(f'{plot_dir}plot_state_means.jpg')
    plt.savefig(f'{plot_dir}plot_state_means.pdf')
    plt.close()

    # stds box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    boxplot = ax.boxplot(stds.T, vert=True)
    ax.set_xticklabels([f'{i + 1}' for i in range(n_states)])
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel(r'$\sigma$')
    plt.tight_layout()
    plt.suptitle(f'Standard deviation, {model_name}_ICA_{n_channels}_state_{n_states}', fontsize=15)
    plt.savefig(f'{plot_dir}plot_state_stds.jpg')
    plt.savefig(f'{plot_dir}plot_state_stds.pdf')
    plt.close()


    # Plot correlation matrix
    # Calculate the number of rows and columns for the subplot grid
    num_cols = 4
    num_rows = n_states // num_cols

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

    # Loop through each correlation matrix and plot it in the corresponding subplot
    for i in range(n_states):
        row = i // num_cols
        col = i % num_cols
        if num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        corr_matrix = corrs[i, :, :]
        # Set the diagonal to zero
        corr_matrix = corr_matrix - np.eye(len(corr_matrix))

        # Plot the correlation matrix as an image (heatmap)
        cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # Set subplot title
        ax.set_title(f'Correlation Matrix {i + 1}')
    # Add a colorbar to show the correlation values
    cbar = fig.colorbar(cax, ax=axes, fraction=0.05, pad=0.04)
    cbar.set_label('Correlation')
    # Adjust spacing between subplots
    #plt.tight_layout()
    plt.savefig(f'{plot_dir}plot_state_correlations.jpg')
    plt.savefig(f'{plot_dir}plot_state_correlations.pdf')
    plt.close()


def compute_distance(save_dir:str,dist_dir:str):
    """
    Compute distance of corrletion and covariance matrices
    Parameters
    ----------
    save_dir: (str) directory to read these matrices
    dist_dir: (str) directory to save distances

    Returns
    -------

    """
    #means = np.load(f'{save_dir}state_means.npy')
    #stds = np.load(f'{save_dir}state_stds.npy')
    correlations = np.load(f'{save_dir}state_correlations.npy')
    covariances = np.load(f'{save_dir}state_covariances.npy')

    # Compute four distance/correlation metrics
    np.save(f'{dist_dir}/frobenius_distance_cov.npy', pairwise_frobenius_distance(covariances))
    np.save(f'{dist_dir}/matrix_correlation_cov.npy', pairwise_matrix_correlations(covariances))
    np.save(f'{dist_dir}/riemannian_distance_cov.npy', pairwise_riemannian_distances(covariances))
    np.save(f'{dist_dir}/congruence_coefficient_cov.npy', pairwise_congruence_coefficient(covariances))

    np.save(f'{dist_dir}/frobenius_distance_cor.npy', pairwise_frobenius_distance(correlations))
    np.save(f'{dist_dir}/matrix_correlation_cor.npy', pairwise_matrix_correlations(correlations))
    np.save(f'{dist_dir}/riemannian_distance_cor.npy', pairwise_riemannian_distances(correlations))
    np.save(f'{dist_dir}/congruence_coefficient_cor.npy', pairwise_congruence_coefficient(correlations))

def plot_distance(dist_dir:str,plot_dir:str,model:str,n_channels:int,n_states:int):
    """
    Plot the distance distribution within each N_states, N_channels
    Parameters
    ----------
    dist_dir: (str) directory containing distance matrices
    dist_plot_dir: (str) directory to save the plot
    model: (str) the model name
    n_channels: (int) number of channels
    n_states: (int) number of states

    Returns
    -------
    """
    measures = ['cov','cor']
    for measure in measures:
        frobenius_distance = np.load(f'{dist_dir}frobenius_distance_{measure}.npy')
        correlation_distance = np.load(f'{dist_dir}matrix_correlation_{measure}.npy')
        riemannian_distance = np.load(f'{dist_dir}riemannian_distance_{measure}.npy')
        congruence_distance = np.load(f'{dist_dir}congruence_coefficient_{measure}.npy')

        N_states = len(frobenius_distance)
        f_d = frobenius_distance[np.triu_indices(N_states, k=1)]
        corr_d = correlation_distance[np.triu_indices(N_states, k=1)]
        r_d = riemannian_distance[np.triu_indices(N_states, k=1)]
        cong_d = congruence_distance[np.triu_indices(N_states, k=1)]

        distances = np.array([f_d, corr_d, r_d, cong_d])
        measures = ['Frobenius', 'Correlation', 'Riemannian', 'Congruence']
        n_measures = len(distances)
        # Start plotting
        fig, axes = plt.subplots(n_measures, n_measures, figsize=(12, 12))

        # Loop through each pair of measures and plot histograms on the diagonal and scatter plots on the off-diagonal
        for i in range(n_measures):
            for j in range(n_measures):
                if i == j:
                    # Plot histogram for diagonal entries
                    axes[i, j].hist(distances[i, :], bins=20, color='blue', alpha=0.7)
                else:
                    data_1 = distances[j, :]
                    data_2 = distances[i, :]
                    # Plot scatter plot for off-diagonal entries (i,j)
                    axes[i, j].scatter(data_1, data_2, color='blue', alpha=0.5)
                    axes[i, j].set_xlabel(measures[j])
                    axes[i, j].set_ylabel(measures[i])

        # Add labels to the diagonal plots
        for i in range(n_measures):
            axes[i, i].set_xlabel(measures[i])
            axes[i, i].set_ylabel('Frequency')

        # Title
        plt.suptitle(f'{measure} distance, {model}_ICA_{n_channels}_states_{n_states}',fontsize=20)

        plt.savefig(f'{plot_dir}distance_plot_{measure}.jpg')
        plt.savefig(f'{plot_dir}distance_plot_{measure}.pdf')
        plt.close()

def mean_mapping(save_dir:str,spatial_map_dir:str):
    """
    Obtain mean activation spatial maps of each state/mode in the specified directory
    Parameters
    ----------
    save_dir: (string) directory to work in
    spatial_map_dir: (string) spatial map of independent components

    Returns
    -------
    """
    state_means = np.load(f'{save_dir}state_means.npy')
    spatial_map = nib.load(spatial_map_dir)
    mean_activation_map = IC2brain(spatial_map, state_means.T)
    nib.save(mean_activation_map, f'{save_dir}mean_activation_map.nii.gz')



def FC_mapping(save_dir:str,spatial_map_dir:str):
    """
    obtain the FC spatial maps of each state/mode in the specified directory
    pipeline: correlation/covariance matrix --> rank-one approximation --> brain map
    Parameters
    ----------
    save_dir: (string) directory to work in
    spatial_map_dir (string) spatial map of independent components

    Returns
    -------
    """
    # Obtain the correlation matrices first (or compute from covariance matrices)
    try:
        correlations = np.load(f'{save_dir}state_correlations.npy')
    except FileNotFoundError:
        covariances = np.load(f'{save_dir}state_covariances.npy')
        correlations = cov2corr(covariances)
        # Remember to save back!
        np.save(f'{save_dir}state_correlations.npy', correlations)

    # Rank-one approximation
    r1_approxs = []
    sum_of_degrees = []
    for i in range(len(correlations)):
        correlation = correlations[i,:,:]
        r1_approxs.append(first_eigenvector(correlation))
        np.fill_diagonal(correlation,0)
        sum_of_degrees.append(np.sum(np.abs(correlation),axis=1))
    r1_approxs = np.array(r1_approxs)
    sum_of_degrees = np.array(sum_of_degrees)
    np.save(f'{save_dir}r1_approx_FC.npy', r1_approxs)
    np.save(f'{save_dir}sum_of_degree.npy',sum_of_degrees)

    # Component map to spatial map
    spatial_map = nib.load(spatial_map_dir)
    FC_degree_map = IC2brain(spatial_map, r1_approxs.T)
    nib.save(FC_degree_map, f'{save_dir}FC_map.nii.gz')
    sum_of_degree_map = IC2brain(spatial_map, sum_of_degrees.T)
    nib.save(sum_of_degree_map, f'{save_dir}FC_sum_of_degree_map.nii.gz')

def comparison_analysis(models:list,list_channels:list,list_states:list,save_dir:str):
    """
    Compare results obtained from different models, channel numbers, state numbers.
    Parameters
    ----------
    models: (list) model list.
    list_channels: (list) N_channels list
    list_states: (list) N_states list
    save_dir: (str) where to save comparison results

    Returns
    -------
    """
    # Create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
