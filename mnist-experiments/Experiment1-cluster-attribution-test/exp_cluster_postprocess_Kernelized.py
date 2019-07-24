"""
Post-processing cluster attribution test results for kernelized tSNE:
- Accuracy
- Distance percentile
- KL divergence
"""
import generate_data
import settings
import numpy as np
import exp_cluster_attr_test_kernelized
import pickle
import logging
import os
from scipy.spatial import distance
import lion_tsne
from scipy import stats


distance_matrix_dir_prefix = '../data/UpdatedPMatrices'

def generate_kernelized_kl_temp_filename(parameters):
    output_file_prefix = '../results/cluster_attr_kernelized_kl_temp_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)


def generate_kernelized_postprocess_filename(parameters):
    output_file_prefix = '../results/cluster_attr_kernelized_postprocess_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)


def main(parameters = settings.parameters):
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    X_mnist = generate_data.load_y_mnist(parameters=parameters)
    accuracy_nn = parameters.get("accuracy_nn", settings.parameters["accuracy_nn"])
    precision_nn = parameters.get("precision_nn", settings.parameters["precision_nn"])
    picked_neighbor_labels = generate_data.load_picked_neighbors_labels(parameters=parameters)
    labels_mnist = generate_data.load_labels_mnist(parameters=parameters)

    # =================== Some starting stuff
    def get_nearest_neighbors_in_y(y, n=10):
        y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
        return np.argsort(y_distances)[:n]

    kernelized_results_file = exp_cluster_attr_test_kernelized.generate_cluster_results_filename(parameters)
    with open(kernelized_results_file, 'rb') as f:
        kernelized_detailed_tsne_method_results, kernelized_detailed_tsne_accuracy, \
        kernelized_detailed_tsne_precision, kernelized_detailed_tsne_method_list = pickle.load(f)
    ind = [4, 24, 49]
    kernelized_method_list = [
        kernelized_detailed_tsne_method_list[i][:10] + kernelized_detailed_tsne_method_list[i][-8:]
        for i in ind]
    kernelized_method_results = [kernelized_detailed_tsne_method_results[i] for i in ind]

    kernelized_accuracy = np.zeros((len(kernelized_method_list,)))
    kernelized_precision = np.zeros((len(kernelized_method_list,)))

    # ============================== Distance percentiles
    D_Y = distance.squareform(distance.pdist(Y_mnist))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    nearest_neighbors_y_dist = np.min(D_Y, axis=1)  # Actually, whatever axis
    kernelized_nearest_neighbors_percentiles_matrix = np.zeros((len(picked_neighbors), len(kernelized_method_list)))
    for i in range(len(picked_neighbors)):
        for j in range(len(kernelized_method_list)):
            y = kernelized_method_results[j][i, :]
            kernelized_dist = np.min(np.sqrt(np.sum((Y_mnist - y) ** 2, axis=1)))
            kernelized_nearest_neighbors_percentiles_matrix[i, j] = stats.percentileofscore(nearest_neighbors_y_dist,
                                                                                            kernelized_dist)
    kernelized_distance_percentiles = np.mean(kernelized_nearest_neighbors_percentiles_matrix, axis=0)
    for j in range(len(kernelized_method_list)):
        logging.info("%s %f", kernelized_method_list[j], kernelized_distance_percentiles[j])

    # ============================== Accuracy and precision
    for j in range(len(kernelized_method_results)):
        per_sample_accuracy = np.zeros((len(picked_neighbors),))
        per_sample_precision = np.zeros((len(picked_neighbors),))
        for i in range(len(picked_neighbors)):
            expected_label = picked_neighbor_labels[i]
            y = kernelized_method_results[j][i,:]
            x = picked_neighbors[j, :]
            nn_x_indices = get_nearest_neighbors_in_y(x, X_mnist, n=precision_nn)
            nn_y_indices = get_nearest_neighbors_in_y(y, Y_mnist, n=precision_nn)
            matching_indices = len([i for i in nn_x_indices if i in nn_y_indices])
            per_sample_precision[i] = (matching_indices / precision_nn)

            kernelized_indices = get_nearest_neighbors_in_y(kernelized_method_results[j][i,:], n=accuracy_nn)
            obtained_labels = labels_mnist[kernelized_indices]
            per_sample_accuracy[i] = sum(obtained_labels==expected_label) / len(obtained_labels)
        kernelized_accuracy[j] = np.mean(per_sample_accuracy)
        kernelized_precision[j] = np.mean(per_sample_precision)
        logging.info("%s :\t%f\t%f", kernelized_method_list[j], kernelized_precision[j],
                     kernelized_accuracy[j])

    kernelized_kl = np.zeros((len(kernelized_method_list), len(picked_neighbors)))

    processed_indices = list()

    kl_kernelized_performance_file = generate_kernelized_kl_temp_filename(parameters)
    if os.path.isfile(kl_kernelized_performance_file):
        with open(kl_kernelized_performance_file, 'rb') as f:
            kernelized_kl, processed_indices = pickle.load(f)

    # ============================== KL divergence
    # KL divergence increase for all 1000 samples is very slow to calculate. Main part of that is calculating P-matrix.
    per_sample_KL = np.zeros((len(picked_neighbors),))
    for i in range(len(picked_neighbors)):
        if i in processed_indices:
            logging.info("Sample %d already processed. Results loaded.", i)
            continue
        logging.info("Processing sample %d", i)
        distance_matrix_dir = distance_matrix_dir_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters, os.sep)
        distance_matrix_file = distance_matrix_dir + 'item' + str(j) + '.p'
        # Make sure you can load them one-by-one.
        if os.path.isfile(distance_matrix_file):
            logging.info("\tP-matrix file found. Loading.")
            with open(distance_matrix_file, 'rb') as f:
                new_P, _ = pickle.load(f)
        else:
            logging.info("\tP-matrix file not found. Creating and saving.")
            new_X = np.concatenate((X_mnist, picked_neighbors[i, :].reshape((1, -1))), axis=0)
            new_D = distance.squareform(distance.pdist(new_X))
            new_P, new_sigmas = lion_tsne.get_p_and_sigma(distance_matrix=new_D, perplexity=dTSNE_mnist.perplexity)
            with open(distance_matrix_file, 'wb') as f:
                pickle.dump((new_P, new_sigmas), f)
        # For all of methods P-matrix is shared.
        for j in range(len(kernelized_method_results)):
            # Single file with p matrix
            new_Y = np.concatenate((Y_mnist, kernelized_method_results[j][i, :].reshape((1, -1))), axis=0)
            kernelized_kl[j, i], _ = lion_tsne.kl_divergence_and_gradient(p_matrix=new_P, y=new_Y)
        processed_indices.append(i)
        with open(kl_kernelized_performance_file, 'wb') as f:
            pickle.dump((kernelized_kl, processed_indices), f)
    # This should be fast
    kernelized_avg_kl = np.mean(kernelized_kl, axis=1)

    output_file = generate_kernelized_postprocess_filename(parameters)
    with open(output_file, "wb") as f:
        pickle.dump((kernelized_method_list, kernelized_accuracy, kernelized_precision,
                     kernelized_avg_kl, kernelized_distance_percentiles),f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parameters=settings.parameters)