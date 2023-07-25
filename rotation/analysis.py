import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import networkx as nx


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
def HMM_analysis(dataset, save_dir):
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
    alpha = model.get_alpha(dataset)
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
    np.save(f'{save_dir}fo.npy',fo)
    # Plot the distribution of fractional occupancy (FO) across subjects
    plotting.plot_violin(fo.T, x_label="State", y_label="FO")
    plt.savefig(f'{plot_dir}fo_violin.jpg')
    plt.savefig(f'{plot_dir}fo_violin.pdf')
    plt.close()

    # Calculate mean lifetimes (in seconds)
    mlt = modes.mean_lifetimes(stc, sampling_frequency=1/0.7)
    np.save(f'{save_dir}mlt.npy',mlt)
    # Plot distribution across subjects
    plotting.plot_violin(mlt.T, x_label="State", y_label="Mean Lifetime (s)")
    plt.savefig(f'{plot_dir}mlt_violin.jpg')
    plt.savefig(f'{plot_dir}mlt_violin.pdf')
    plt.close()

    # Calculate mean intervals (in seconds)
    mintv = modes.mean_intervals(stc, sampling_frequency=1/0.7)
    np.save(f'{save_dir}mintv.npy',mintv)
    # Plot distribution across subjects
    plotting.plot_violin(mintv.T, x_label="State", y_label="Mean Interval (s)")
    plt.savefig(f'{plot_dir}mintv_violin.jpg')
    plt.savefig(f'{plot_dir}mintv_violin.pdf')
    plt.close()
    
    # Analyze the transition probability matrix
    # using Louvain community detection algorithm
    tpm = np.load(f'{save_dir}trans_prob.npy')
    G = construct_graph(tpm)
    partition = nx.community.louvain_communities(G)
    print('The final partition is: ',partition)

    with open(f'{save_dir}tpm_partition.pkl', 'wb') as file:
        pickle.dump(partition, file)


def Dynemo_analysis(dataset, save_dir):
    from osl_dynamics.models import load
    model = load(save_dir)
    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))

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