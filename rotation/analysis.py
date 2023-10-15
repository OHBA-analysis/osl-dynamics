import os
import random
import json
import warnings
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import networkx as nx
import nibabel as nib

from osl_dynamics.array_ops import cov2corr
import osl_dynamics.data
from osl_dynamics.inference.metrics import pairwise_frobenius_distance,\
    pairwise_matrix_correlations, pairwise_riemannian_distances, pairwise_congruence_coefficient
from osl_dynamics.utils import plotting
from rotation.utils import plot_FO, stdcor2cov,cov2stdcor, first_eigenvector,\
    IC2brain,IC2surface, regularisation, pairwise_fisher_z_correlations,\
    twopair_vector_correlation, twopair_riemannian_distance, twopair_fisher_z_transformed_correlation,\
    hungarian_pair,heatmap_reorder_matrix

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
                 spatial_map_dir:str, spatial_surface_map_dir:str, n_channels:int, n_states:int):
    """
    Post-training analysis of HMM model
    Parameters
    ----------
    dataset: (osl_dynamics.data.Data) dataset to work on
    save_dir: (str) directory to save results
    spatial_map_dir: (str) directory of groupICA spatial maps
    spatial_surface_map_dir: (str) directory of groupICA spatial surface maps
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

    # Calculate the metrics
    if not os.path.isfile(f'{save_dir}metrics_repeat.json'):
        calculate_metrics(model, dataset, save_dir)

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

    # Plot the convergence of loss function
    if not os.path.isfile(f'{plot_dir}loss_history.pdf'):
        plot_loss_history(save_dir,plot_dir)

    # Analyze the transition probability matrix
    # using Louvain community detection algorithm
    if not os.path.isfile(f'{save_dir}tpm_partition.pkl'):
        tpm = np.load(f'{save_dir}trans_prob.npy')
        # Added by swimming 2023-08-09: try to reproduce Diego's results
        # Only work on HMM_ICA_50_state_12
        for i in range(len(tpm)):
            tpm[i,i] = 0
            tpm[i,:] = tpm[i,:] / np.sum(tpm[i,:])
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
    if not os.path.isfile(f'{dist_dir}riemannian_distance_cor.npy'):
        compute_distance(save_dir,dist_dir)

    # Plot the distance between different states/modes
    if not os.path.isfile(f'{plot_dir}/correct_distance_plot_cor.pdf'):
        plot_distance(dist_dir,plot_dir,
                      model='HMM',
                      n_channels=n_channels,
                      n_states=n_states)

    # Plot the influence of regularisation on Riemannian distance
    if not os.path.isfile(f'{plot_dir}/Riemannian_distance_regularisation_plot.pdf'):
        compute_plot_distance_regularisation(save_dir,dist_dir, plot_dir,
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
    if not os.path.isfile(f'{save_dir}mean_activation_surface_map.dscalar.nii'):
        mean_mapping(save_dir,spatial_map_dir,spatial_surface_map_dir)

    # Compute rank-one decomposition of FC map
    if not os.path.isfile(f'{save_dir}FC_sum_of_degree_surface_map.dscalar.nii'):
        FC_mapping(save_dir,spatial_map_dir,spatial_surface_map_dir)

    if not os.path.isfile(f'{plot_dir}mean_FC_relation.pdf'):
        mean_FC_relation(save_dir,plot_dir,'HMM',n_channels,n_states)

    reproduce_analysis_dir = f'{save_dir}reproduce_analysis/'
    if not os.path.exists(reproduce_analysis_dir):
        os.makedirs(reproduce_analysis_dir)
    if not os.path.isfile(f'{reproduce_analysis_dir}FCs_distance_plot_split_4.pdf'):
        reproduce_analysis(save_dir,reproduce_analysis_dir,'HMM',n_channels,n_states,split_strategy='1')
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'HMM',n_channels,n_states, split_strategy='2')
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'HMM',n_channels,n_states, split_strategy='3')
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'HMM', n_channels, n_states, split_strategy='4')



def Dynemo_analysis(dataset:osl_dynamics.data.Data, save_dir:str,
                    spatial_map_dir:str,spatial_surface_map_dir:str, n_channels:int,n_states:int):
    """
    Post-training analysis of Dynemo model
    Parameters
    ----------
    dataset: (osl_dynamics.data.Data) dataset to work on
    save_dir: (str) directory to save results
    spatial_map_dir: (str) directory of spatial maps
    spatial_surface_map_dir: (str) directory of groupICA spatial surface maps
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

    # Plot the convergence of loss function
    if not os.path.isfile(f'{plot_dir}loss_history.pdf'):
        plot_loss_history(save_dir, plot_dir)


    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))

    # Calculate the metrics
    if not os.path.isfile(f'{save_dir}metrics_repeat.json'):
        calculate_metrics(model, dataset, save_dir)


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
    # Fractional occupancy analysis (similar to HMM)
    FO_dir = f'{save_dir}FO_analysis/'
    if not os.path.exists(FO_dir):
        os.makedirs(FO_dir)
    if not os.path.isfile(f'{FO_dir}fo_violin.pdf'):
        FO_dynemo(save_dir,FO_dir,n_channels,n_states)
    # Analyze the distance between different states/modes
    dist_dir = f'{save_dir}distance/'
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    if not os.path.isfile(f'{dist_dir}riemannian_distance_cor.npy'):
        compute_distance(save_dir, dist_dir)

    # Plot the distance between different states/modes
    if not os.path.isfile(f'{plot_dir}/correct_distance_plot_cor.pdf'):
        plot_distance(dist_dir, plot_dir,
                      model='Dynemo',
                      n_channels=n_channels,
                      n_states=n_states)

    # Plot the influence of regularisation on Riemannian distance
    if not os.path.isfile(f'{plot_dir}/Riemannian_distance_regularisation_plot.pdf'):
        compute_plot_distance_regularisation(save_dir, dist_dir, plot_dir,
                                             model='Dynemo',
                                             n_channels=n_channels,
                                             n_states=n_states)

    # Compute the mean activation map
    if not os.path.isfile(f'{save_dir}mean_activation_surface_map.dscalar.nii'):
        mean_mapping(save_dir,spatial_map_dir,spatial_surface_map_dir)

    if not os.path.isfile(f'{save_dir}FC_sum_of_degree_surface_map.dscalar.nii'):
        FC_mapping(save_dir,spatial_map_dir,spatial_surface_map_dir)

    if not os.path.isfile(f'{plot_dir}mean_FC_relation.pdf'):
        mean_FC_relation(save_dir,plot_dir,'Dynemo',n_channels,n_states)

    reproduce_analysis_dir = f'{save_dir}reproduce_analysis/'
    if not os.path.exists(reproduce_analysis_dir):
        os.makedirs(reproduce_analysis_dir)
    if not os.path.isfile(f'{reproduce_analysis_dir}FCs_distance_plot_split_4.pdf'):
        reproduce_analysis(save_dir,reproduce_analysis_dir,'Dynemo',n_channels,n_states,split_strategy='1')
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'Dynemo',n_channels,n_states, split_strategy='2')
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'Dynemo',n_channels,n_states, split_strategy='3')
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'Dynemo', n_channels, n_states, split_strategy='4')

def MAGE_analysis(dataset:osl_dynamics.data.Data, save_dir:str,
                  spatial_map_dir:str, spatial_surface_map_dir:str, n_channels:int, n_states:int):
    """
    Post-training analysis of MAGE
    Parameters
    ----------
    dataset: (osl_dynamics.data.Data) dataset to work on
    save_dir: (str) directory to save results
    spatial_map_dir: (str) directory of groupICA spatial maps
    spatail_surface_map_dir: (str) directory of groupICA spatial surface maps
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

    # Plot the convergence of loss function
    if not os.path.isfile(f'{plot_dir}loss_history.pdf'):
        plot_loss_history(save_dir, plot_dir)

    if not os.path.isfile(f'{save_dir}alpha.pkl'):
        alpha = model.get_alpha(dataset)
        pickle.dump(alpha, open(f'{save_dir}alpha.pkl', "wb"))

    # Calculate the metrics
    if not os.path.isfile(f'{save_dir}metrics_repeat.json'):
        calculate_metrics(model, dataset, save_dir)


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
    if not os.path.isfile(f'{dist_dir}riemannian_distance_cor.npy'):
        compute_distance(save_dir, dist_dir)

    # Plot the distance between different states/modes
    if not os.path.isfile(f'{plot_dir}/correct_distance_plot_cor.pdf'):
        plot_distance(dist_dir, plot_dir,
                      model='MAGE',
                      n_channels=n_channels,
                      n_states=n_states)

    # Plot the influence of regularisation on Riemannian distance
    if not os.path.isfile(f'{plot_dir}/Riemannian_distance_regularisation_plot.pdf'):
        compute_plot_distance_regularisation(save_dir, dist_dir, plot_dir,
                                             model='MAGE',
                                             n_channels=n_channels,
                                             n_states=n_states)


    # Compute the mean activation map
    if not os.path.isfile(f'{save_dir}mean_activation_surface_map.dscalar.nii'):
        mean_mapping(save_dir,spatial_map_dir,spatial_surface_map_dir)

    if not os.path.isfile(f'{save_dir}FC_sum_of_degree_surface_map.dscalar.nii'):
        FC_mapping(save_dir,spatial_map_dir,spatial_surface_map_dir)

    if not os.path.isfile(f'{plot_dir}mean_FC_relation.pdf'):
        mean_FC_relation(save_dir,plot_dir,'MAGE',n_channels,n_states)

    reproduce_analysis_dir = f'{save_dir}reproduce_analysis/'
    if not os.path.exists(reproduce_analysis_dir):
        os.makedirs(reproduce_analysis_dir)
    if not os.path.isfile(f'{reproduce_analysis_dir}FCs_distance_plot_split_4.pdf'):
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'MAGE',n_channels,n_states, split_strategy='1')
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'MAGE',n_channels,n_states, split_strategy='2')
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'MAGE',n_channels,n_states, split_strategy='3')
        reproduce_analysis(save_dir, reproduce_analysis_dir, 'MAGE', n_channels, n_states, split_strategy='4')
def SWC_analysis(save_dir,old_dir,n_channels,n_states):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plot_dir = f'{save_dir}plot/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    measures = ['cor','cov']
    file_name = {'cor':'state_correlations',
                 'cov':'state_covariances'}

    if not os.path.isfile(f'{save_dir}split_1_second_half/state_stds.npy'):
        for measure in measures:
            K_means_clustering(old_dir,save_dir,file_name,n_states,n_channels,measure)

            # Split half and implement K_means again
            random_seed = 42
            random.seed(random_seed)
            swc = np.load(f'{old_dir}cor_swc.npy')
            random_index = random.sample(range(len(swc)), int(len(swc) / 2))

            if not os.path.exists(f'{save_dir}split_1_first_half/'):
                os.makedirs(f'{save_dir}split_1_first_half/')
            K_means_clustering(old_dir,f'{save_dir}split_1_first_half/',file_name,n_states,n_channels,measure,split_index=random_index)

            if not os.path.exists(f'{save_dir}split_1_second_half/'):
                os.makedirs(f'{save_dir}split_1_second_half/')
            random_index_C = list(set(np.arange(len(swc))) - set(random_index))
            K_means_clustering(old_dir,f'{save_dir}split_1_second_half/',file_name,n_states,n_channels,measure,split_index=random_index_C)

    # Reproducibility analysis
    correlations_1 = np.load(f'{save_dir}split_1_first_half/state_correlations.npy')
    correlations_2 = np.load(f'{save_dir}split_1_second_half/state_correlations.npy')
    FCs_fisher_z_transformed_correlation = twopair_fisher_z_transformed_correlation(correlations_1, correlations_2)
    row_column_indices_FCs_Fisher, FCs_correlation_reorder_fisher = hungarian_pair(
            FCs_fisher_z_transformed_correlation, distance=False)

    reproduce_analysis_dir = f'{save_dir}reproduce_analysis/'
    if not os.path.exists(reproduce_analysis_dir):
        os.makedirs(reproduce_analysis_dir)
    split_strategy = '1'
    np.save(f'{reproduce_analysis_dir}FCs_fisher_correlation_split_{split_strategy}.npy',
            FCs_fisher_z_transformed_correlation)
    with open(f'{reproduce_analysis_dir}FCs_row_column_indices_Fisher_split_{split_strategy}.json', 'w') as json_file:
        json.dump(row_column_indices_FCs_Fisher, json_file)
    np.save(f'{reproduce_analysis_dir}FCs_fisher_correlation_reorder_split_{split_strategy}.npy',
            FCs_correlation_reorder_fisher)

    heatmap_reorder_matrix(FCs_correlation_reorder_fisher, reproduce_analysis_dir,
                           'FCs_fisher_z_correlation', row_column_indices_FCs_Fisher,
                           'SWC', n_channels, n_states, split_strategy)

    # Analyze the distance between different states/modes
    dist_dir = f'{save_dir}distance/'
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    if not os.path.isfile(f'{dist_dir}fisher_z_correlation_cor.npy'):
        compute_distance(save_dir, dist_dir)

    # Plot the distance between different states/modes
    if not os.path.isfile(f'{plot_dir}/correct_distance_plot_cor.pdf'):
        plot_distance(dist_dir, plot_dir,
                      model='SWC',
                      n_channels=n_channels,
                      n_states=n_states
                      )

def K_means_clustering(old_dir:str,save_dir:str,file_name:dict,n_states:int,n_channels:int,measure:str,split_index=None):
    """
    Use K_means algorithm to cluster the correlations/covariances from SWC
    Parameters
    ----------
    old_dir: (str) the directory to where the original correlations/covariances are saved
    save_dir: (str) the directory to save the centroids
    file_name: (dict) the file name dictionary
    n_states: (int) number of states
    n_channels: (int) number of channels
    measure: (str) 'cor' or 'cov'
    split_index: (list) the index to split the data, if is None, then do not split
    Returns
    -------

    """

    # Fix the random seed

    swc = np.load(f'{old_dir}/{measure}_swc.npy')

    if split_index is not None:
        swc = swc[split_index,:,:,:]



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
    np.save(f'{save_dir}{file_name[measure]}.npy', kmean_networks)

    if measure == 'cov':
        stds, _ = cov2stdcor(kmean_networks)
        np.save(f'{save_dir}state_stds.npy', stds)
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

def calculate_metrics(model,dataset,save_dir):
    """
    Calculate the metrics: free energy and model evidence of the data
    Parameters
    ----------
    model: the after-training model
    dataset: data to calculate metrics on
    save_dir: where to save the metrics

    Returns
    -------
    """
    from osl_dynamics.models import load

    if not os.path.isfile(f'{save_dir}metrics.json'):
        free_energy = model.free_energy(dataset)
        #evidence = model.evidence(dataset)
        metrics = {'free_energy':free_energy.tolist(),}#'evidence':evidence}
        with open(f'{save_dir}metrics.json', "w") as json_file:
            # Use json.dump to write the data to the file
            json.dump(metrics, json_file)
    if not os.path.isfile(f'{save_dir}metrics_repeat.json'):
        free_energy_list = []
        # We repeat the model for five times
        for i in range(1,6):
            if os.path.exists(f'{save_dir}repeat_{i}'):
                repeat_model = load(f'{save_dir}repeat_{i}/')
                free_energy_list.append(float(repeat_model.free_energy(dataset)))
        with open(f'{save_dir}metrics_repeat.json',"w") as json_file:
            json.dump({'free_energy':free_energy_list},json_file)




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

    # When plotting, change the model name from MAGE to mDynemo
    if model_name == 'MAGE':
        model_name = 'mDynemo'
    means = np.load(f'{save_dir}state_means.npy')
    stds = np.load(f'{save_dir}state_stds.npy')
    corrs = np.load(f'{save_dir}state_correlations.npy')

    # means box plot
    fig, ax = plt.subplots(figsize=(10,6))
    boxplot = ax.boxplot(means.T,vert=True)
    ax.set_xticklabels([f'{i+1}'for i in range(n_states)],fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('State',fontsize=17)
    ax.set_ylabel(r'$\mu$',fontsize=17)
    plt.tight_layout()
    plt.suptitle(f'Mean, {model_name}_ICA_{n_channels}_state_{n_states}',fontsize=22)
    plt.savefig(f'{plot_dir}plot_state_means.jpg')
    plt.savefig(f'{plot_dir}plot_state_means.pdf')
    plt.close()

    # stds box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    boxplot = ax.boxplot(stds.T, vert=True)
    ax.set_xticklabels([f'{i + 1}' for i in range(n_states)],fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('State', fontsize=17)
    ax.set_ylabel(r'$\sigma$',fontsize=17)
    plt.tight_layout()
    plt.suptitle(f'Standard deviation, {model_name}_ICA_{n_channels}_state_{n_states}', fontsize=22)
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
        ax.set_title(f'FC {i + 1}',fontsize=13)
    # Add a colorbar to show the correlation values
    cbar = fig.colorbar(cax, ax=axes, fraction=0.05, pad=0.04)
    cbar.set_label('Correlation',fontsize=15)
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

    # Regularisation for Riemannian distance
    # Please work on correlation matrix only
    eps_values = [0,1e-9,1e-8,1e-7,1e-6,1e-5]

    # Compute four distance/correlation metrics
    np.save(f'{dist_dir}frobenius_distance_cov.npy', pairwise_frobenius_distance(covariances))
    np.save(f'{dist_dir}matrix_correlation_cov.npy', pairwise_matrix_correlations(covariances))
    np.save(f'{dist_dir}congruence_coefficient_cov.npy', pairwise_congruence_coefficient(covariances))
    np.save(f'{dist_dir}fisher_z_correlation_cov.npy', pairwise_fisher_z_correlations(covariances))
    try:
        np.save(f'{dist_dir}riemannian_distance_cov.npy', pairwise_riemannian_distances(covariances))
    except np.linalg.LinAlgError:
        warnings.warn("Riemannian distance is not computed properly for covariances!")

    np.save(f'{dist_dir}frobenius_distance_cor.npy', pairwise_frobenius_distance(correlations))
    np.save(f'{dist_dir}matrix_correlation_cor.npy', pairwise_matrix_correlations(correlations))
    np.save(f'{dist_dir}congruence_coefficient_cor.npy', pairwise_congruence_coefficient(correlations))
    np.save(f'{dist_dir}fisher_z_correlation_cor.npy', pairwise_fisher_z_correlations(correlations))

    for eps in eps_values:
        try:
            np.save(f'{dist_dir}riemannian_distance_cor.npy', pairwise_riemannian_distances(regularisation(correlations,eps)))
            print(f'Riemannian distance calculation succeeds when eps = {eps}')
            break
        except np.linalg.LinAlgError:
            print(f'Riemannian distance is not computed properly when eps = {eps}')

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
    # When plotting, change the model name from MAGE to mDynemo
    if model == 'MAGE':
        model = 'mDynemo'
    measures = ['cor','cov']
    # Remark by swimming 8th Aug 2023
    # Correct implementations are Riemannian distance and Fisher z-transformed correlation
    # So our plot should be only 2 * 2 now.
    # For those Riemannian distance is not available (numerical issues),
    # only plot the histogram of Fisher z-transformed
    for measure in measures:
        fisher = np.load(f'{dist_dir}fisher_z_correlation_{measure}.npy')
        N_states = len(fisher)
        fisher_d = fisher[np.triu_indices(N_states, k=1)]
        try:
            riemannian = np.load(f'{dist_dir}riemannian_distance_{measure}.npy')
            riemannian_d = riemannian[np.triu_indices(N_states, k=1)]
        except  FileNotFoundError:
            fig = plt.figure()
            plt.hist(fisher_d, bins=20, color='blue', alpha=0.7)
            plt.title(f'{measure} distance, {model}_ICA_{n_channels}_states_{n_states}',fontsize=20)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes[0, 0].hist(riemannian_d, bins=20, color='blue', alpha=0.7)
            axes[0, 0].set_xlabel('Riemannian',fontsize=15)
            axes[0, 0].set_ylabel('Frequency',fontsize=15)
            axes[0, 0].set_xticks(axes[0, 0].get_xticks())
            axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(),fontsize=13)
            axes[0, 0].set_yticks(axes[0, 0].get_yticks())
            axes[0, 0].set_yticklabels(axes[0, 0].get_yticklabels(),fontsize=13)

            axes[1, 1].hist(fisher_d,bins=20,color='blue',alpha=0.7)
            axes[1, 1].set_xlabel('Fisher-z',fontsize=15)
            axes[1, 1].set_ylabel('Frequency',fontsize=15)
            axes[1, 1].set_xticks(axes[1, 1].get_xticks())
            axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(),fontsize=13)
            axes[1, 1].set_yticks(axes[1, 1].get_yticks())
            axes[1, 1].set_yticklabels(axes[1, 1].get_yticklabels(),fontsize=13)

            axes[0, 1].scatter(fisher_d, riemannian_d, color='blue', alpha=0.5)
            axes[0, 1].set_xlabel('Fisher-z',fontsize=15)
            axes[0, 1].set_ylabel('Riemannian',fontsize=15)

            axes[0, 1].set_xticks(axes[0, 1].get_xticks())
            axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(),fontsize=13)
            axes[0, 1].set_yticks(axes[0, 1].get_yticks())
            axes[0, 1].set_yticklabels(axes[0, 1].get_yticklabels(),fontsize=13)

            plt.suptitle(f'{measure} distance, {model}_ICA_{n_channels}_states_{n_states}',fontsize=20)

        plt.savefig(f'{plot_dir}correct_distance_plot_{measure}.jpg')
        plt.savefig(f'{plot_dir}correct_distance_plot_{measure}.pdf')
        plt.close()




    '''
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
    '''

def compute_plot_distance_regularisation(save_dir:str,dist_dir:str, plot_dir:str,model:str,n_channels:int,n_states:int):
    """
    This function serves to compute the effect of regularisation on the Riemannian distance.
    The Riemannian distance is numerically unstable because some correlation matrices
    are "almost" positive semi-definite,i.e., scipy.linalg returns small negative eigenvalues.
    One of the solutions is to add a small \epsilon to the diagonal (see rotation.utils.regularisation).
    We need to understand: (1) the appropriate value of \epsilon, (2) the effect of regularisation

    So we add \epsilon=1e-7,...,1e-4 to the correlation matrix, and scatter plot
    the "regularised" Riemannian distance against each other
    Parameters
    ----------
    save_dir: (str) directory to read correlation matrices
    dist_dir: (str) directory to store the distances
    plot_dir: (str) directory to save the plot
    model: (str) The model to use
    n_channels: (int) number of channels
    n_states: (int) number of states

    Returns
    -------

    """

    # When plotting, change the model name from MAGE to mDynemo
    if model == 'MAGE':
        model = 'mDynemo'

    eps_values = [0,1e-8,1e-7,1e-6,1e-5]
    distances = {}
    correlations = np.load(f'{save_dir}state_correlations.npy')

    for eps in eps_values:
        reg_correlations = regularisation(correlations,eps)
        try:
            distances[eps] = pairwise_riemannian_distances(reg_correlations)
        except np.linalg.LinAlgError:
            print(f'Riemannian distance when eps = {eps} failed!')

    np.savez(f'{dist_dir}Riemannian_distance_regularisation.npz',distances)

    # Flatten
    N_states = len(correlations)
    for eps in distances.keys():
        distances[eps] = (distances[eps])[np.triu_indices(N_states, k=1)]

    n_measures = len(eps_values)
    # Start plotting
    fig, axes = plt.subplots(n_measures, n_measures, figsize=(15, 15))

    # Loop through each pair of measures and plot histograms on the diagonal and scatter plots on the off-diagonal
    for i,eps_i in enumerate(eps_values):
        if eps_i in distances.keys():
            for j,eps_j in enumerate(eps_values):
                if i == j:
                    # Plot histogram for diagonal entries
                    axes[i, j].hist(distances[eps_i], bins=20, color='blue', alpha=0.7)
                    axes[i, j].set_xlabel(str(eps_i),fontsize=15)
                    axes[i, i].set_ylabel('Frequency',fontsize=15)
                else:
                    if eps_j in distances.keys():
                        data_1 = distances[eps_j]
                        data_2 = distances[eps_i]
                        # Plot scatter plot for off-diagonal entries (i,j)
                        axes[i, j].scatter(data_1, data_2, color='blue', alpha=0.5)
                        axes[i, j].set_xlabel(str(eps_j),fontsize=15)
                        axes[i, j].set_ylabel(str(eps_i),fontsize=15)


    # Title
    plt.suptitle(f'Riemannian distance regularisation, {model}_ICA_{n_channels}_states_{n_states}', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}Riemannian_distance_regularisation_plot.jpg')
    plt.savefig(f'{plot_dir}Riemannian_distance_regularisation_plot.pdf')
    plt.close()



def mean_mapping(save_dir:str,spatial_map_dir:str,spatial_surface_map_dir:str):
    """
    Obtain mean activation spatial maps of each state/mode in the specified directory
    Parameters
    ----------
    save_dir: (str) directory to work in
    spatial_map_dir: (str) spatial map of independent components
    spatial_surface_map_dir: (str) spatial surface map of independent components
    Returns
    -------
    """
    state_means = np.load(f'{save_dir}state_means.npy')
    spatial_map = nib.load(spatial_map_dir)
    spatial_surface_map = nib.load(spatial_surface_map_dir)

    mean_activation_map = IC2brain(spatial_map, state_means.T)
    mean_activation_surface_map = IC2surface(spatial_surface_map,state_means.T)

    nib.save(mean_activation_map, f'{save_dir}mean_activation_map.nii.gz')
    mean_activation_surface_map.to_filename(f'{save_dir}mean_activation_surface_map.dscalar.nii')



def FC_mapping(save_dir:str,spatial_map_dir:str,spatial_surface_map_dir:str):
    """
    obtain the FC spatial maps of each state/mode in the specified directory
    pipeline: correlation/covariance matrix --> rank-one approximation --> brain map
    Parameters
    ----------
    save_dir: (str) directory to work in
    spatial_map_dir: (str) spatial map of independent components
    spatial_surface_map_dir: (str) spatial surface map of independent components
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
        sum_of_degrees.append(np.sum(correlation,axis=1))
    r1_approxs = np.array(r1_approxs)
    sum_of_degrees = np.array(sum_of_degrees)
    np.save(f'{save_dir}r1_approx_FC.npy', r1_approxs)
    np.save(f'{save_dir}sum_of_degree.npy',sum_of_degrees)

    # Component map to spatial map
    spatial_map = nib.load(spatial_map_dir)
    spatial_surface_map = nib.load(spatial_surface_map_dir)

    FC_degree_map = IC2brain(spatial_map, r1_approxs.T)
    FC_degree_surface_map = IC2surface(spatial_surface_map,r1_approxs.T)

    nib.save(FC_degree_map, f'{save_dir}FC_map.nii.gz')
    FC_degree_surface_map.to_filename(f'{save_dir}FC_surface_map.dscalar.nii')

    sum_of_degree_map = IC2brain(spatial_map, sum_of_degrees.T)
    sum_of_degree_surface_map = IC2surface(spatial_surface_map,sum_of_degrees.T)

    nib.save(sum_of_degree_map, f'{save_dir}FC_sum_of_degree_map.nii.gz')
    sum_of_degree_surface_map.to_filename(f'{save_dir}FC_sum_of_degree_surface_map.dscalar.nii')
def FO_dynemo(save_dir:str,FO_dir:str,n_channels:int,n_states:int):
    """
    Fractional Occupancy Analysis in Dynemo, similar to HMM.
    The aim is to summarize the mixing coefficient time course
    Parameters
    ----------
    save_dir: (str) directory to work in
    FO_dir: (str) directory to save the results
    n_channels: number of channels
    n_states: number of states

    Returns
    -------
    """
    # Step 1: Plot unnormalised alpha
    alpha = pickle.load(open(f"{save_dir}alpha.pkl","rb"))
    plotting.plot_alpha(alpha[0],n_samples=200)
    plt.title('Unnormalised alpha',fontsize=20)
    plt.savefig(f'{FO_dir}example_unnormalised_alpha.jpg')
    plt.savefig(f'{FO_dir}example_unnormalised_alpha.pdf')
    plt.close()

    # Step 2: normalise the alpha
    def normalize_alpha(a:np.ndarray, D:np.ndarray) -> np.ndarray:
        """
        Calculate the weighting for each mode
        measured by the trace of each mode covariance
        Parameters
        ----------
        a: (np.ndarray) N_time_points * N_states
        D: (np.ndarray) covariance matrices N_states * N_channels * N_channels

        Returns
        -------
        wa: (np.ndarray) weighted alpha, having the shape as a
        """
        # Calculate the weighting for each mode
        # We use the trace of each mode covariance
        w = np.trace(D, axis1=1, axis2=2)

        # Weight the alphas
        wa = w[np.newaxis, ...] * a

        # Renormalize so the alphas sum to one
        wa /= np.sum(wa, axis=1, keepdims=True)
        return wa

    # Load the inferred mode covariances
    covs = np.load(f'{save_dir}state_covariances.npy')

    # Renormalize the mixing coefficients
    norm_alpha = [normalize_alpha(a, covs) for a in alpha]

    # Plot the renormalized mixing coefficient time course for the first subject (8 seconds)
    plotting.plot_alpha(norm_alpha[0], n_samples=200)
    plt.title('Normalised alpha',fontsize=20)
    plt.savefig(f'{FO_dir}example_normalised_alpha.jpg')
    plt.savefig(f'{FO_dir}example_normalised_alpha.pdf')
    plt.close()

    # Step 3: Summary statistics
    # Analogous to FO in HMM, we can compute the time average mixing coefficient
    mean_norm_alpha = np.array([np.mean(a, axis=0) for a in norm_alpha])
    np.save(f'{save_dir}fo.npy',mean_norm_alpha)

    plot_FO(mean_norm_alpha, FO_dir)

    std_norm_alpha = np.array([np.std(a, axis=0) for a in norm_alpha])
    np.save(f'{save_dir}normalised_alpha_std.npy',std_norm_alpha)

    # Plot the distribution of fractional occupancy (FO) across subjects
    plotting.plot_violin(mean_norm_alpha.T, x_label="State", y_label="FO")
    plt.savefig(f'{FO_dir}fo_violin.jpg')
    plt.savefig(f'{FO_dir}fo_violin.pdf')
    plt.close()

def FO_MAGE(save_dir:str,FO_dir:str,n_channels:int,n_states:int):
    """
    Fractional Occupancy Analysis in MAGE, similar to HMM.
    The aim is to summarize the mixing coefficient time course
    Parameters
    ----------
    save_dir: (str) directory to work in
    FO_dir: (str) directory to save the results
    n_channels: number of channels
    n_states: number of states

    Returns
    -------

    """
    # Step 1: Plot alpha
    alpha = pickle.load(open(f"{save_dir}alpha.pkl", "rb"))
    alpha_1 = alpha[0]
    alpha_2 = alpha[1]

    plotting.plot_alpha(alpha_1, n_samples=200)
    plt.title('alpha 1', fontsize=20)
    plt.savefig(f'{FO_dir}example_alpha_1.jpg')
    plt.savefig(f'{FO_dir}example_alpha_1.pdf')
    plt.close()

    plotting.plot_alpha(alpha_2, n_samples=200)
    plt.title('alpha 2', fontsize=20)
    plt.savefig(f'{FO_dir}example_alpha_2.jpg')
    plt.savefig(f'{FO_dir}example_alpha_2.pdf')
    plt.close()

    # Step 2: Summary Statistics
    mean_alpha_1 = np.array([np.mean(a, axis=0) for a in alpha_1])
    np.save(f'{save_dir}fo_alpha_1.npy', mean_alpha_1)
    mean_alpha_2 = np.array([np.mean(a, axis=0) for a in alpha_2])
    np.save(f'{save_dir}fo_alpha_2.npy', mean_alpha_2)

    plot_FO(mean_alpha_1, FO_dir,file_name='fo_alpha_1_hist')
    plot_FO(mean_alpha_2,FO_dir,file_name='fo_alpha_2_hist')

    std_alpha_1 = np.array([np.std(a, axis=0) for a in alpha_1])
    np.save(f'{save_dir}alpha_1_std.npy', std_alpha_1)

    std_alpha_2 = np.array([np.std(a, axis=0) for a in alpha_2])
    np.save(f'{save_dir}alpha_2_std.npy', std_alpha_2)

    # Plot the distribution of fractional occupancy (FO) across subjects
    plotting.plot_violin(mean_alpha_1.T, x_label="State", y_label="FO")
    plt.savefig(f'{FO_dir}fo_alpha_1_violin.jpg')
    plt.savefig(f'{FO_dir}fo_alpha_1_violin.pdf')
    plt.close()

    plotting.plot_violin(mean_alpha_2.T, x_label="State", y_label="FO")
    plt.savefig(f'{FO_dir}fo_alpha_2_violin.jpg')
    plt.savefig(f'{FO_dir}fo_alpha_2_violin.pdf')
    plt.close()



def comparison_analysis(models:list,list_channels:list,list_states:list,result_dir:str, save_dir:str):
    """
    Compare results obtained from different models, channel numbers, state numbers.
    Parameters
    ----------
    models: (list) model list.
    list_channels: (list) N_channels list
    list_states: (list) N_states list
    result_dir: (str) where to find the results of different models
    save_dir: (str) where to save comparison results

    Returns
    -------
    """
    # We do not have results for N_channels = 200,300
    list_channels = list_channels[:-2]
    # Create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Compare the reproducibility across all models
    for model in models:
        mean_correlation = {}
        FC_correlation = {}
        if model != 'SWC':
            FC_distance = {}
        for N_channel in list_channels:
            for N_state in list_states:
                data = np.load(f'{result_dir}{model}_ICA_{N_channel}_state_{N_state}/reproduce_analysis/FCs_fisher_correlation_reorder_split_1.npy')
                FC_correlation[f'ICA_{N_channel}_state_{N_state}'] = np.mean(np.diagonal(data))

                if model!= 'SWC':
                    data = np.load(f'{result_dir}{model}_ICA_{N_channel}_state_{N_state}/reproduce_analysis/means_correlation_reorder_split_1.npy')
                    mean_correlation[f'ICA_{N_channel}_state_{N_state}'] = np.mean(np.diagonal(data))
                    data = np.load(
                        f'{result_dir}{model}_ICA_{N_channel}_state_{N_state}/reproduce_analysis/FCs_distance_reorder_split_1.npy')
                    FC_distance[f'ICA_{N_channel}_state_{N_state}'] = np.mean(np.diagonal(data))
        group_comparison_plot(FC_correlation, model, list_channels, list_states, 'Fisher_z_transformed_correlation', save_dir)
        if model != 'SWC':
            group_comparison_plot(FC_distance, model, list_channels, list_states, 'Riemannian_distance',
                                  save_dir)
            group_comparison_plot(mean_correlation, model, list_channels, list_states, 'mean_correlation',
                                  save_dir)

    # Compare the free energy and model evidence across all methods
    for model in models:
        free_energy = {}
        evidence = {}
        for N_channel in list_channels:
            for N_state in list_states:
                with open(f'{result_dir}{model}_ICA_{N_channel}_state_{N_state}/metrics_repeat.json', "r") as json_file:
                    # Use json.load to load the data from the file
                    metrics = json.load(json_file)
                    free_energy[f'ICA_{N_channel}_state_{N_state}'] = metrics['free_energy']
                    #evidence[f'ICA_{N_channel}_state_{N_state}'] = metrics['evidence']

        group_comparison_plot(free_energy, model, list_channels, list_states, 'free_energy', save_dir)
        group_comparison_plot(free_energy,model,list_channels,list_states,'free_energy',save_dir)
        #group_comparison_plot(evidence,model,list_channels,list_states,'model_evidence',save_dir)

def group_comparison_plot(metric:dict,model_name:str,list_channels:list,list_states:list,metric_name:str,save_dir:str):
    """
    Plot the group comparison results.
    The metric should be a dictionary with keys: ICA_{N_channels}_state_{N_states}
    the values should represent some metrics.
    Parameters
    ----------
    metric: (dict) the metrics to plot
    model_name: (str) the model name
    list_channels: (list) the list of N_channels
    list_states: (list) the list of N_states
    metric_name: (str) the name of the metric plotted
    save_dir: (str) where to save the plots
    Returns
    -------
    """

    # Create a bar plot

    # When plotting, change the model name from MAGE to mDynemo
    if model_name == 'MAGE':
        model_name = 'mDynemo'
    plt.figure(figsize=(10, 6))

    bar_width = 0.15
    colors = plt.cm.viridis(np.linspace(0, 1, len(list_states)))

    for i, N_states in enumerate(list_states):
        channel_keys = [f'ICA_{N_channels}_state_{N_states}' for N_channels in list_channels]
        values = [metric[key] for key in channel_keys]
        means =  [np.mean(vals) for vals in values]
        stds = [np.std(vals) for vals in values]
        plt.bar(np.arange(len(channel_keys)) + i * bar_width, means, bar_width, label=f'N_states = {N_states}',
                yerr=stds,color=colors[i])

    plt.xlabel('N_channels', fontsize=15)
    plt.ylabel(metric_name, fontsize=15)
    plt.title(f'Comparison of {metric_name} by N_channels and N_states', fontsize=20)
    plt.xticks(np.arange(len(channel_keys)) + bar_width * (len(list_states) - 1) / 2,
               [str(N_channels) for N_channels in list_channels], fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(prop={'size': 15})
    plt.tight_layout()
    plt.savefig(f'{save_dir}{model_name}_{metric_name}_comparison.jpg')
    plt.savefig(f'{save_dir}{model_name}_{metric_name}_comparison.pdf')
    plt.close()

def mean_FC_relation(save_dir:str,plot_dir:str,model_name:str,n_channels:int,n_states:int):
    """
    Explore the relationship between mean activation and FC map.
    Question: Within each state, does the activated component have strong correlation with other componens.
    Rationale: for HMM, Dynemo, because it's single dynamics, we should expect
    mean activation and FC to be independent, but this is not the case for Dynemo and MAGE.
    Method: compute the sum of FC across rows/columns, but we need to explore whether
    the sign plays a role (both in activation and FC) in that case. So there should be
    four histograms!
    Parameters
    ----------
    save_dir: (str) where state mean FC are saved and where to save the raw data for plotting
    plot_dir: (str) where to plot the results
    model_name: (str) model name
    n_channels: (int) number of channels
    n_states: (int) number of states
    Returns
    -------
    """
    means = np.load(f'{save_dir}state_means.npy')
    means_abs = np.abs(means)
    correlations = np.load(f'{save_dir}state_correlations.npy')
    FC = np.sum(correlations,axis=1)
    FC_abs = np.sum(np.abs(correlations),axis=1)

    result = {'means_FC':np.array([np.corrcoef(means[i, :], FC[i, :])[0, 1] for i in range(n_states)]),
              'means_abs_FC':np.array([np.corrcoef(means_abs[i, :], FC[i, :])[0, 1] for i in range(n_states)]),
              'means_FC_abs':np.array([np.corrcoef(means[i, :], FC_abs[i, :])[0, 1] for i in range(n_states)]),
              'means_abs_FC_abs':np.array([np.corrcoef(means_abs[i, :], FC_abs[i, :])[0, 1] for i in range(n_states)])
              }
    np.savez(f'{save_dir}mean_FC_relation.npz',result)

    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'{model_name}_ICA_{n_channels}_state_{n_states}', fontsize=20)

    # Loop through dictionary keys and plot histograms
    for i, (key, value) in enumerate(result.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ax.hist(value, color='blue', alpha=0.7)
        ax.set_title(key)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}mean_FC_relation.jpg')
    plt.savefig(f'{plot_dir}mean_FC_relation.pdf')
    plt.close()

def reproduce_analysis(save_dir:str, reproduce_analysis_dir:str,model_name:str,n_channels:int,n_states:int,split_strategy:str='1'):
    """
    Analysis the reproducibility of each model
    Parameters
    ----------
    save_dir: (str) the root directory of the model results
    reproduce_analysis_dir: (str) directory to save reproducibility analysis results
    model_name: (str) the model name
    n_channels: (int) number of channels
    n_states: (int) number of states
    split_strategy: (str) split strategy '1','2','3','4'
    Returns
    -------
    """
    extract_state_statistics(f'{save_dir}split_{split_strategy}_first_half/',model_name)
    extract_state_statistics(f'{save_dir}split_{split_strategy}_second_half/', model_name)

    means_1 = np.load(f'{save_dir}split_{split_strategy}_first_half/state_means.npy')
    correlations_1 = np.load(f'{save_dir}split_{split_strategy}_first_half/state_correlations.npy')

    means_2 = np.load(f'{save_dir}split_{split_strategy}_second_half/state_means.npy')
    correlations_2 = np.load(f'{save_dir}split_{split_strategy}_second_half/state_correlations.npy')

    means_correlation = twopair_vector_correlation(means_1,means_2)
    row_column_indices, means_correlation_reorder = hungarian_pair(means_correlation,distance=False)
    np.save(f'{reproduce_analysis_dir}means_correlation_split_{split_strategy}.npy',means_correlation)
    with open(f'{reproduce_analysis_dir}means_row_column_indices_split_{split_strategy}.json', 'w') as json_file:
        json.dump(row_column_indices, json_file)
    np.save(f'{reproduce_analysis_dir}means_correlation_reorder_split_{split_strategy}.npy',means_correlation_reorder)

    heatmap_reorder_matrix(means_correlation_reorder,reproduce_analysis_dir,
                           'means_correlation',row_column_indices,
                           model_name,n_channels,n_states,split_strategy)
    # Work on FCs
    FCs_distance = twopair_riemannian_distance(correlations_1,correlations_2)
    FCs_fisher_z_transformed_correlation = twopair_fisher_z_transformed_correlation(correlations_1,correlations_2)
    row_column_indices_FCs,FCs_distance_reorder = hungarian_pair(FCs_distance,distance=True)
    row_column_indices_FCs_Fisher,FCs_correlation_reorder_fisher =  hungarian_pair(FCs_fisher_z_transformed_correlation,distance=False)

    np.save(f'{reproduce_analysis_dir}FCs_distance_split_{split_strategy}.npy', FCs_distance)
    with open(f'{reproduce_analysis_dir}FCs_row_column_indices_split_{split_strategy}.json', 'w') as json_file:
        json.dump(row_column_indices_FCs, json_file)
    np.save(f'{reproduce_analysis_dir}FCs_distance_reorder_split_{split_strategy}.npy', FCs_distance_reorder)

    np.save(f'{reproduce_analysis_dir}FCs_fisher_correlation_split_{split_strategy}.npy', FCs_fisher_z_transformed_correlation)
    with open(f'{reproduce_analysis_dir}FCs_row_column_indices_Fisher_split_{split_strategy}.json', 'w') as json_file:
        json.dump(row_column_indices_FCs_Fisher, json_file)
    np.save(f'{reproduce_analysis_dir}FCs_fisher_correlation_reorder_split_{split_strategy}.npy', FCs_correlation_reorder_fisher)

    heatmap_reorder_matrix(FCs_distance_reorder,reproduce_analysis_dir,
                           'FCs_distance',row_column_indices_FCs,
                           model_name,n_channels,n_states,split_strategy)

    heatmap_reorder_matrix(FCs_correlation_reorder_fisher, reproduce_analysis_dir,
                           'FCs_fisher_z_correlation', row_column_indices_FCs_Fisher,
                           model_name, n_channels, n_states, split_strategy)

def plot_loss_history(save_dir:str,plot_dir:str):
    """
    Plot the history of loss function.
    The figures should include original training, repeat_1,2,3,4
    Parameters
    ----------
    save_dir: the directory where the trained model is saved.
    plot_dir: where to plot the results

    Returns
    -------

    """
    loss_history = np.load(f'{save_dir}/loss_history.npy')
    epochs = np.arange(1, len(loss_history) + 1)
    plt.plot(epochs, loss_history)
    for i in range(1,5):
        loss_history = np.load(f'{save_dir}/repeat_{i}/loss_history.npy')
        plt.plot(epochs, loss_history)

        # Plotting the loss history

    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('Loss',fontsize=15)
    plt.title('Training Loss Over Epochs',fontsize=20)
    plt.savefig(f'{plot_dir}loss_history.jpg')
    plt.savefig(f'{plot_dir}loss_history.pdf')
    plt.close()