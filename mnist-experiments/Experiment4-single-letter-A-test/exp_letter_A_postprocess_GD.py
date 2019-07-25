"""
Post-processing letter placement test results for GD:
- Distance percentile
- KL divergence
"""
import generate_data
import settings
import numpy as np
import exp_letter_A_test_GD
import pickle
import logging
import os
from scipy.spatial import distance
import lion_tsne
from scipy import stats

distance_matrix_dir_prefix = '../data/UpdatedPMatrices-letters-A'


def generate_gd_kl_temp_filename(parameters):
    output_file_prefix = '../results/letter_A_gd_kl_temp_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_A_parameter_set, parameters)


def generate_gd_postprocess_filename(parameters):
    output_file_prefix = '../results/letter_A_gd_postprocess_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_A_parameter_set, parameters)


def main(parameters = settings.parameters):
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    letter_A_samples, _ = generate_data.load_A_letters(parameters=parameters)
    X_mnist = generate_data.load_x_mnist(parameters=parameters)

    D_Y = distance.squareform(distance.pdist(Y_mnist))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    nearest_neighbors_y_dist = np.min(D_Y, axis=1)  # Actually, whatever axis

    # ============== KL Divergence
    gd_method_list = [r'Closest $Y_{init}$',
                  r'Random $Y_{init}$',
                  r'Closest $Y_{init}$; new $\sigma$',
                  r'Random $Y_{init}$; new $\sigma$',
                  r'Closest $Y_{init}$; EE',
                  r'Random $Y_{init}$; EE',
                  r'Closest $Y_{init}$; new $\sigma$; EE',
                  r'Random $Y_{init}$; new $\sigma$; EE']

    gd_results_file = exp_letter_A_test_GD.generate_letter_A_results_filename(parameters=parameters)
    with open(gd_results_file, 'rb') as f:
        (letters_A_y_gd_transformed, letters_A_y_gd_variance_recalc_transformed,
         letters_A_y_gd_transformed_random, letters_A_y_gd_variance_recalc_transformed_random,
         letters_A_y_gd_early_exagg_transformed_random,
         letters_A_y_gd_early_exagg_transformed,
         letters_A_y_gd_variance_recalc_early_exagg_transformed_random,
         picked_random_starting_positions,
         letters_A_y_gd_variance_recalc_early_exagg_transformed, covered_samples) = pickle.load(f)

    gd_letters_A_results = [
        letters_A_y_gd_transformed,
        letters_A_y_gd_transformed_random,
        letters_A_y_gd_variance_recalc_transformed,
        letters_A_y_gd_variance_recalc_transformed_random,
        letters_A_y_gd_early_exagg_transformed,
        letters_A_y_gd_early_exagg_transformed_random,
        letters_A_y_gd_variance_recalc_early_exagg_transformed,
        letters_A_y_gd_variance_recalc_early_exagg_transformed_random,
    ]

    gd_letters_A_kl = np.zeros((len(gd_method_list), len(letter_A_samples)))

    processed_indices = list()

    kl_gd_letters_A_performance_file = generate_gd_kl_temp_filename(parameters)
    if os.path.isfile(kl_gd_letters_A_performance_file):
        with open(kl_gd_letters_A_performance_file, 'rb') as f:
            gd_letters_A_kl, processed_indices = pickle.load(f)

    # KL divergence increase for all 1000 samples is very slow to calculate. Main part of that is calculating P-matrix.
    per_sample_KL = np.zeros((len(letter_A_samples),))
    for i in range(len(letter_A_samples)):
        if i in processed_indices:
            logging.info("Sample %d already processed. Results loaded.", i)
            continue
        logging.info("Processing sample %d", i)
        distance_matrix_dir = distance_matrix_dir_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.letter_A_parameter_set, parameters, os.sep)
        distance_matrix_file = distance_matrix_dir + 'item' + str(i) + '.p'
        # Make sure you can load them one-by-one.
        if os.path.isfile(distance_matrix_file):
            logging.info("\tP-matrix file found. Loading.")
            with open(distance_matrix_file, 'rb') as f:
                new_P, _ = pickle.load(f)
        else:
            logging.info("\tP-matrix file not found. Creating and saving.")
            new_X = np.concatenate((X_mnist, letter_A_samples[i, :].reshape((1, -1))), axis=0)
            new_D = distance.squareform(distance.pdist(new_X))
            new_P, new_sigmas = lion_tsne.get_p_and_sigma(distance_matrix=new_D, perplexity=dTSNE_mnist.perplexity)
            with open(distance_matrix_file, 'wb') as f:
                pickle.dump((new_P, new_sigmas), f)
        # For all of methods P-matrix is shared.
        for j in range(len(gd_letters_A_results)):
            # Single file with p matrix
            new_Y = np.concatenate((Y_mnist, gd_letters_A_results[j][i, :].reshape((1, -1))), axis=0)
            gd_letters_A_kl[j, i], _ = lion_tsne.kl_divergence_and_gradient(p_matrix=new_P, y=new_Y)
        processed_indices.append(i)
        with open(kl_gd_letters_A_performance_file, 'wb') as f:
            pickle.dump((gd_letters_A_kl, processed_indices), f)
    # This should be fast
    gd_avg_letters_A_kl = np.mean(gd_letters_A_kl, axis=1)

    # ============== Distance percentiles
    gd_letters_A_percentiles_matrix = np.zeros((len(letter_A_samples), len(gd_method_list)))
    gd_letters_A_distance_matrix = np.zeros((len(letter_A_samples), len(gd_method_list)))
    for i in range(len(letter_A_samples)):
        for j in range(len(gd_method_list)):
            y = gd_letters_A_results[j][i, :]
            nn_dist = np.min(np.sqrt(np.sum((Y_mnist - y) ** 2, axis=1)))
            gd_letters_A_distance_matrix[i, j] = nn_dist
            gd_letters_A_percentiles_matrix[i, j] = stats.percentileofscore(nearest_neighbors_y_dist, nn_dist)
    gd_letters_A_distance_percentiles = np.mean(gd_letters_A_percentiles_matrix, axis=0)
    gd_letters_A_distances = np.mean(gd_letters_A_distance_matrix, axis=0)
    for j in range(len(gd_method_list)):
        logging.info("%s: %f, %f", gd_method_list[j], gd_letters_A_distances[j], gd_letters_A_distance_percentiles[j])

    output_file = generate_gd_postprocess_filename(parameters)
    with open(output_file, "wb") as f:
        pickle.dump((gd_method_list, gd_avg_letters_A_kl, gd_letters_A_distance_percentiles),f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parameters=settings.parameters)
