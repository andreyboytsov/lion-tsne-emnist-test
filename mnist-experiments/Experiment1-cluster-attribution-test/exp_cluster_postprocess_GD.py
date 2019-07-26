"""
Post-processing cluster attribution test results for GD:
- Accuracy
- Distance percentile
- KL divergence
"""
import generate_data
import settings
import numpy as np
import exp_cluster_attr_test_GD
import pickle
import logging
import os
from scipy.spatial import distance
import lion_tsne
from scipy import stats


distance_matrix_dir_prefix = '../data/UpdatedPMatrices'

def generate_gd_kl_temp_filename(parameters):
    output_file_prefix = '../results/cluster_attr_gd_kl_temp_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)


def generate_gd_postprocess_filename(parameters):
    output_file_prefix = '../results/cluster_attr_gd_postprocess_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)


def main(parameters = settings.parameters):
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    X_mnist = generate_data.load_x_mnist(parameters=parameters)
    accuracy_nn = parameters.get("accuracy_nn", settings.parameters["accuracy_nn"])
    precision_nn = parameters.get("precision_nn", settings.parameters["precision_nn"])
    picked_neighbor_labels = generate_data.load_picked_neighbors_labels(parameters=parameters)
    labels_mnist = generate_data.load_labels_mnist(parameters=parameters)

    # =================== ACCURACY
    def get_nearest_neighbors_in_y(y, Y_mnist, n=10):
        y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
        return np.argsort(y_distances)[:n]


    gd_method_list = [r'Closest $Y_{init}$',
                  r'Random $Y_{init}$',
                  r'Closest $Y_{init}$; new $\sigma$',
                  r'Random $Y_{init}$; new $\sigma$',
                  r'Closest $Y_{init}$; EE',
                  r'Random $Y_{init}$; EE',
                  r'Closest $Y_{init}$; new $\sigma$; EE',
                  r'Random $Y_{init}$; new $\sigma$; EE']

    gd_results_file = exp_cluster_attr_test_GD.generate_cluster_results_filename(parameters=parameters)
    with open(gd_results_file, 'rb') as f:
        (picked_neighbors_y_gd_transformed, picked_neighbors_y_gd_variance_recalc_transformed,
         picked_neighbors_y_gd_transformed_random, picked_neighbors_y_gd_variance_recalc_transformed_random,
         picked_neighbors_y_gd_early_exagg_transformed_random,
         picked_neighbors_y_gd_early_exagg_transformed,
         picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random,
         picked_random_starting_positions,
         picked_neighbors_y_gd_variance_recalc_early_exagg_transformed, covered_samples) = pickle.load(f)

    gd_method_results = [
        picked_neighbors_y_gd_transformed,
        picked_neighbors_y_gd_transformed_random,
        picked_neighbors_y_gd_variance_recalc_transformed,
        picked_neighbors_y_gd_variance_recalc_transformed_random,
        picked_neighbors_y_gd_early_exagg_transformed,
        picked_neighbors_y_gd_early_exagg_transformed_random,
        picked_neighbors_y_gd_variance_recalc_early_exagg_transformed,
        picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random,
    ]

    input_time_file = exp_cluster_attr_test_GD.generate_time_results_filename(parameters)
    with open(input_time_file, 'rb') as f:
        picked_neighbors_y_time_gd_transformed, picked_neighbors_y_time_gd_variance_recalc_transformed, \
        picked_neighbors_y_time_gd_transformed_random, \
        picked_neighbors_y_time_gd_variance_recalc_transformed_random, \
        picked_neighbors_y_time_gd_early_exagg_transformed_random, \
        picked_neighbors_y_time_gd_early_exagg_transformed, \
        picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed_random, \
        picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed, covered_samples = pickle.load(f)


    gd_time = [
        np.mean(picked_neighbors_y_time_gd_transformed),
        np.mean(picked_neighbors_y_time_gd_transformed_random),
        np.mean(picked_neighbors_y_time_gd_variance_recalc_transformed),
        np.mean(picked_neighbors_y_time_gd_variance_recalc_transformed_random),
        np.mean(picked_neighbors_y_time_gd_early_exagg_transformed),
        np.mean(picked_neighbors_y_time_gd_early_exagg_transformed_random),
        np.mean(picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed),
        np.mean(picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed_random),
    ]

    gd_accuracy = np.zeros((len(gd_method_list,)))
    gd_precision = np.zeros((len(gd_method_list, )))

    # ============================== Distance percentiles
    D_Y = distance.squareform(distance.pdist(Y_mnist))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    nearest_neighbors_y_dist = np.min(D_Y, axis=1)  # Actually, whatever axis
    gd_nearest_neighbors_percentiles_matrix = np.zeros((len(picked_neighbors), len(gd_method_list)))
    for i in range(len(picked_neighbors)):
        for j in range(len(gd_method_list)):
            y = gd_method_results[j][i, :]
            nn_dist = np.min(np.sqrt(np.sum((Y_mnist - y) ** 2, axis=1)))
            gd_nearest_neighbors_percentiles_matrix[i, j] = stats.percentileofscore(nearest_neighbors_y_dist, nn_dist)
    gd_distance_percentiles = np.mean(gd_nearest_neighbors_percentiles_matrix, axis=0)
    for j in range(len(gd_method_list)):
        logging.info("%s :\t%f", gd_method_list[j], gd_distance_percentiles[j])

    # ============================== KL divergence
    for j in range(len(gd_method_results)):
        per_sample_accuracy = np.zeros((len(picked_neighbors),))
        per_sample_precision = np.zeros((len(picked_neighbors),))
        for i in range(len(picked_neighbors)):
            expected_label = picked_neighbor_labels[i]
            nn_indices = get_nearest_neighbors_in_y(gd_method_results[j][i,:], Y_mnist, n=accuracy_nn)
            obtained_labels = labels_mnist[nn_indices]
            per_sample_accuracy[i] = sum(obtained_labels==expected_label) / len(obtained_labels)

            x = picked_neighbors[i, :]
            y = gd_method_results[j][i, :]
            nn_x_indices = get_nearest_neighbors_in_y(x, X_mnist, n=precision_nn)
            nn_y_indices = get_nearest_neighbors_in_y(y, Y_mnist, n=precision_nn)
            matching_indices = len([i for i in nn_x_indices if i in nn_y_indices])
            per_sample_precision[i] = (matching_indices / precision_nn)

        gd_accuracy[j] = np.mean(per_sample_accuracy)
        gd_precision[j] = np.mean(per_sample_precision)
        logging.info("%s :\t%f\t%f", gd_method_list[j], gd_precision[j], gd_accuracy[j])

    gd_kl = np.zeros((len(gd_method_list), len(picked_neighbors)))

    processed_indices = list()

    kl_gd_performance_file = generate_gd_kl_temp_filename(parameters)
    if os.path.isfile(kl_gd_performance_file):
        with open(kl_gd_performance_file, 'rb') as f:
            gd_kl, processed_indices = pickle.load(f)

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
        for j in range(len(gd_method_results)):
            # Single file with p matrix
            new_Y = np.concatenate((Y_mnist, gd_method_results[j][i, :].reshape((1, -1))), axis=0)
            gd_kl[j, i], _ = lion_tsne.kl_divergence_and_gradient(p_matrix=new_P, y=new_Y)
        processed_indices.append(i)
        with open(kl_gd_performance_file, 'wb') as f:
            pickle.dump((gd_kl, processed_indices), f)
    # This should be fast
    gd_avg_kl = np.mean(gd_kl, axis=1)

    output_file = generate_gd_postprocess_filename(parameters)
    with open(output_file, "wb") as f:
        pickle.dump((gd_method_list, gd_accuracy, gd_precision, gd_time, gd_avg_kl, gd_distance_percentiles), f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parameters=settings.parameters)