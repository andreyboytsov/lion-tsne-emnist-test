"""
Kernel tSNE (Giesbrecht's algorithm)
As described in:

Andrej Gisbrecht, Alexander Schulz, and Barbara Hammer. Parametric nonlinear dimensionality reduction using kernel
t-SNE. Neurocomputing, 147:71–82, January 2015.

Here we have only general algorithm parts. Cluster attribution experiment and outlier placement experiment will
be done later.
"""
import settings
import pickle
import generate_data
import numpy as np
from scipy.spatial import distance
import os
import logging


def generate_cache_filename(parameters=settings.parameters):
    cache_file_prefix = '../results/kernelized_tsne_parameters_cache'
    return cache_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)


def load_all_kernelized_tsne_embedders(parameters=settings.parameters, regenerate_parameters_cache=False):
    X_mnist = generate_data.load_x_mnist(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)

    # Implementing carefully. Not the fastest, but the most reliable way.
    kernelized_tsne_parameters_cache = dict()
    cache_filename = generate_cache_filename(parameters=parameters)

    if not regenerate_parameters_cache and os.path.isfile(cache_filename):
        with open(cache_filename, 'rb') as f:
            kernelized_tsne_parameters_cache = pickle.load(f)
    else:
        D = distance.squareform(distance.pdist(X_mnist))

        step = 0.01
        choice_K = np.arange(step, 3+step, step) # Let's try those K.

        np.fill_diagonal(D, np.inf)
        closest_neighbor_dist = np.min(D, axis = 1).reshape((1,-1))
        np.fill_diagonal(D, 0)

        # Sigma is a multiply over closest NN distance
        for k in choice_K:
            key = "%.2f"%k
            if k not in kernelized_tsne_parameters_cache or regenerate_parameters_cache:
                kernelized_tsne_parameters_cache[key] = dict()
                # Creating matrix to get coefficients using SLE
                sigma_matrix = k*np.repeat(closest_neighbor_dist, X_mnist.shape[0], axis=0)

                kernel_matrix = np.exp(-D**2/(2*sigma_matrix**2))
                kernel_matrix = kernel_matrix / np.sum(kernel_matrix, axis=1).reshape((-1,1)) # Normalizing  by rows

                coefs = np.linalg.inv(kernel_matrix).dot(Y_mnist)
                kernelized_tsne_parameters_cache[key]['coefs'] = coefs
                kernelized_tsne_parameters_cache[key]['sigma'] = sigma_matrix[0,:]
                logging.info("Got coefs for coefficient %f", k)
        with open(cache_filename, 'wb') as f:
            pickle.dump(kernelized_tsne_parameters_cache, f)
    return kernelized_tsne_parameters_cache


def generate_kernelized_tsne_mapping_function(parameters=settings.parameters, regenerate_parameters_cache=False):
    X_mnist = generate_data.load_x_mnist(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    kernelized_tsne_parameters_cache = load_all_kernelized_tsne_embedders(parameters=parameters,
                                                        regenerate_parameters_cache=regenerate_parameters_cache)

    def kernel_tsne_mapping(x, k=1):
        '''
        Getting kernel tSNE. Starting from scratch, so use all data at once.
        '''
        # Let's go for reliable option.

        cache = kernelized_tsne_parameters_cache["%.2f" % k]

        y = np.zeros((x.shape[0], Y_mnist.shape[1]))
        for i in range(len(x)):
            square_distances = np.sum((X_mnist - x[i, :]) ** 2, axis=1).reshape((1, -1))
            kernel_values = np.exp(-square_distances / (2 * cache['sigma'] ** 2))
            kernel_values = kernel_values / np.sum(kernel_values)
            y[i, :] = kernel_values.dot(cache['coefs']).reshape((-1, Y_mnist.shape[1]))
        return y

    return kernel_tsne_mapping