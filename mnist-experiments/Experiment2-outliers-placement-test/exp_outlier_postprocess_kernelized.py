"""
Post-processing cluster attribution test results for NN:
- Accuracy
- Distance percentile
- KL divergence
"""
import generate_data
import settings
import numpy as np
import exp_outlier_test_kernelized
import pickle
import logging
import os
from scipy.spatial import distance
import lion_tsne
from scipy import stats

distance_matrix_dir_prefix = '../data/UpdatedPMatrices-outliers'


def generate_kernelized_kl_temp_filename(parameters):
    output_file_prefix = '../results/outlier_kernelized_kl_temp_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.outlier_parameter_set, parameters)


def generate_kernelized_postprocess_filename(parameters):
    output_file_prefix = '../results/outlier_kernelized_postprocess_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.outlier_parameter_set, parameters)


def main(parameters = settings.parameters):
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    outlier_samples, _ = generate_data.load_outliers(parameters=parameters)
    X_mnist = generate_data.load_x_mnist(parameters=parameters)

    D_Y = distance.squareform(distance.pdist(Y_mnist))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    nearest_neighbors_y_dist = np.min(D_Y, axis=1)  # Actually, whatever axis

    kernelized_results_file = exp_outlier_test_kernelized.generate_outlier_results_filename(parameters)
    with open(kernelized_results_file, 'rb') as f:
        kernelized_detailed_method_results, kernelized_detailed_method_list = pickle.load(f)
    ind = [4, 24, 49]
    
    kernelized_method_list = [
        kernelized_detailed_method_list[i][:10] + kernelized_detailed_method_list[i][-8:]
        for i in ind]
    kernelized_outliers_results = [kernelized_detailed_method_results[i] for i in ind]

    # =========== DISTANCE PERCENTILES ==========
    kernelized_outliers_percentiles_matrix = np.zeros((len(outlier_samples), len(kernelized_method_list)))
    kernelized_outliers_distance_matrix = np.zeros((len(outlier_samples), len(kernelized_method_list)))
    for i in range(len(outlier_samples)):
        for j in range(len(kernelized_method_list)):
            y = kernelized_outliers_results[j][i,:]
            nn_dist = np.min(np.sqrt(np.sum((Y_mnist-y)**2, axis=1)))
            kernelized_outliers_distance_matrix[i,j] = nn_dist
            kernelized_outliers_percentiles_matrix[i,j] = stats.percentileofscore(nearest_neighbors_y_dist, nn_dist)
    kernelized_outliers_distance_percentiles = np.mean(kernelized_outliers_percentiles_matrix, axis=0)
    kernelized_outliers_distances = np.mean(kernelized_outliers_distance_matrix, axis=0)
    for j in range(len(kernelized_method_list)):
        logging.info("%s: %f, %f", kernelized_method_list[j], kernelized_outliers_distances[j],
              kernelized_outliers_distance_percentiles[j])

    kernelized_outliers_kl = np.zeros((len(kernelized_method_list), len(outlier_samples)))
    processed_indices = list()

    kl_kernelized_tsne_outliers_performance_file = generate_kernelized_kl_temp_filename(parameters)
    if os.path.isfile(kl_kernelized_tsne_outliers_performance_file):
        with open(kl_kernelized_tsne_outliers_performance_file, 'rb') as f:
            kernelized_outliers_kl, processed_indices = pickle.load(f)

    # =========== KL DIVERGENCE ==========
    # KL divergence increase for all 1000 samples is very slow to calculate. Main part of that is calculating P-matrix.
    per_sample_KL = np.zeros((len(outlier_samples),))
    for i in range(len(outlier_samples)):
        if i in processed_indices:
            logging.info("Sample %d already processed. Results loaded.",i)
            continue
        logging.info("Processing sample %d", i)
        distance_matrix_dir = distance_matrix_dir_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.outlier_parameter_set, parameters, os.sep)
        distance_matrix_file = distance_matrix_dir + 'item' + str(j) + '.p'
        # Make sure you can load them one-by-one.
        if os.path.isfile(distance_matrix_file):
            logging.info("\tP-matrix file found. Loading.")
            with open(distance_matrix_file, 'rb') as f:
                new_P, _ = pickle.load(f)
        else:
            logging.info("\tP-matrix file not found. Creating and saving.")
            new_X = np.concatenate((X_mnist, outlier_samples[i, :].reshape((1, -1))), axis=0)
            new_D = distance.squareform(distance.pdist(new_X))
            new_P, new_sigmas = lion_tsne.get_p_and_sigma(distance_matrix=new_D, perplexity=dTSNE_mnist.perplexity)
            with open(distance_matrix_file, 'wb') as f:
                pickle.dump((new_P, new_sigmas), f)
        # For all of methods P-matrix is shared.
        for j in range(len(kernelized_outliers_results)):
            # Single file with p matrix
            new_Y = np.concatenate((Y_mnist, kernelized_outliers_results[j][i, :].reshape((1, -1))), axis=0)
            kernelized_outliers_kl[j, i], _ = lion_tsne.kl_divergence_and_gradient(p_matrix=new_P, y=new_Y)
        processed_indices.append(i)
        with open(kl_kernelized_tsne_outliers_performance_file, 'wb') as f:
            pickle.dump((kernelized_outliers_kl, processed_indices), f)
    # This should be fast
    kernelized_avg_outliers_kl = np.mean(kernelized_outliers_kl, axis=1)

    output_file = generate_kernelized_postprocess_filename(parameters)
    with open(output_file, "wb") as f:
        pickle.dump((kernelized_method_list, kernelized_avg_outliers_kl, kernelized_outliers_distance_percentiles),f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parameters=settings.parameters)
