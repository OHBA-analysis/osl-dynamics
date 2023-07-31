import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import nibabel as nib

from osl_dynamics.array_ops import cov2corr
from osl_dynamics.inference.metrics import pairwise_frobenius_distance,\
    pairwise_matrix_correlations, pairwise_riemannian_distances, pairwise_congruence_coefficient
from rotation.utils import plot_FO, stdcor2cov,first_eigenvector, IC2brain

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
def HMM_analysis(dataset, save_dir,spatial_map_dir):
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

    # Analyze the distance between different states/modes
    dist_dir = f'{save_dir}/distance/'
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
        model = load(save_dir)
        means, covariances = model.get_means_covariances()
        np.save(f'{save_dir}state_means.npy',means)
        np.save(f'{save_dir}state_covariances.npy',covariances)

        # Compute four distance/correlation metrics
        np.save(f'{dist_dir}/frobenius_distance.npy',pairwise_frobenius_distance(covariances))
        np.save(f'{dist_dir}/matrix_correlation.npy', pairwise_matrix_correlations(covariances))
        np.save(f'{dist_dir}/riemannian_distance.npy', pairwise_riemannian_distances(covariances))
        np.save(f'{dist_dir}/congruence_coefficient.npy', pairwise_congruence_coefficient(covariances))

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
        state_means = np.load(f'{save_dir}state_means.npy')
        spatial_map = nib.load(spatial_map_dir)
        mean_activation_map = IC2brain(spatial_map,state_means.T)

        nib.save(mean_activation_map,f'{save_dir}mean_activation_map.nii.gz')

    # Compute rank-one decomposition of FC map
    if not os.path.isfile(f'{save_dir}FC_map.nii.gz'):
        FC_map(save_dir,spatial_map_dir)




def Dynemo_analysis(dataset, save_dir, spatial_map_dir):
    from osl_dynamics.models import load
    model = load(save_dir)
    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))

    dist_dir = f'{save_dir}/distance/'
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
        model = load(save_dir)
        means, covariances = model.get_means_covariances()
        np.save(f'{save_dir}state_means.npy',means)
        np.save(f'{save_dir}state_covariances.npy',covariances)
        correlations = cov2corr(covariances)

        # Compute four distance/correlation metrics
        np.save(f'{dist_dir}/frobenius_distance.npy', pairwise_frobenius_distance(covariances))
        np.save(f'{dist_dir}/matrix_correlation.npy', pairwise_matrix_correlations(covariances))
        np.save(f'{dist_dir}/riemannian_distance.npy', pairwise_riemannian_distances(covariances))
        np.save(f'{dist_dir}/congruence_coefficient.npy', pairwise_congruence_coefficient(covariances))

    # Compute the mean activation map
    if not os.path.isfile(f'{save_dir}mean_activation_map.nii.gz'):
        state_means = np.load(f'{save_dir}state_means.npy')
        spatial_map = nib.load(spatial_map_dir)
        mean_activation_map = IC2brain(spatial_map, state_means.T)
        nib.save(mean_activation_map, f'{save_dir}mean_activation_map.nii.gz')
def MAGE_analysis(dataset,save_dir, spatial_map_dir):
    from osl_dynamics.models import load
    model = load(save_dir)
    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))

    dist_dir = f'{save_dir}/distance/'
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
        model = load(save_dir)
        means, stds, correlations = model.get_means_stds_fcs()
        covariances = stdcor2cov(stds,correlations)
        np.save(f'{save_dir}state_means.npy', means)
        np.save(f'{save_dir}state_stds.npy',stds)
        np.save(f'{save_dir}state_correlations.npy', correlations)
        np.save(f'{save_dir}state_covariances.npy',covariances)

        # Compute four distance/correlation metrics
        np.save(f'{dist_dir}/frobenius_distance.npy', pairwise_frobenius_distance(covariances))
        np.save(f'{dist_dir}/matrix_correlation.npy', pairwise_matrix_correlations(covariances))
        np.save(f'{dist_dir}/riemannian_distance.npy', pairwise_riemannian_distances(covariances))
        np.save(f'{dist_dir}/congruence_coefficient.npy', pairwise_congruence_coefficient(covariances))

    # Compute the mean activation map
    if not os.path.isfile(f'{save_dir}mean_activation_map.nii.gz'):
        state_means = np.load(f'{save_dir}state_means.npy')
        spatial_map = nib.load(spatial_map_dir)
        mean_activation_map = IC2brain(spatial_map, state_means.T)
        nib.save(mean_activation_map, f'{save_dir}mean_activation_map.nii.gz')
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

def FC_map(save_dir,spatial_map_dir):
    """
    obtain the FC spatial maps of each state/mode in the specified directory
    pipeline: correlation/covariance matrix --> rank-one approximation --> brain map
    Parameters
    ----------
    save_dir: (string) directory to work on
    spatial_map_dir (string) spatial map of indepedent components

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
    for i in range(len(correlations)):
        r1_approxs.append(first_eigenvector(correlations[i, :, :]))
    r1_approxs = np.array(r1_approxs)
    np.save(f'{save_dir}r1_approx_FC.npy', r1_approxs)

    # Component map to spatial map
    spatial_map = nib.load(spatial_map_dir)
    FC_degree_map = IC2brain(spatial_map, r1_approxs.T)
    nib.save(FC_degree_map, f'{save_dir}FC_map.nii.gz')