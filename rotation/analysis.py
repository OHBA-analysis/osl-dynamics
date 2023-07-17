import os
import numpy as np
from sklearn.cluster import KMeans
import pickle

def HMM_analysis(dataset, save_dir):
    from osl_dynamics.models import load
    model = load(save_dir)
    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))
        
    

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