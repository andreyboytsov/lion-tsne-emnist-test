"""
Post-processing cluster attribution test results for NN:
- Accuracy
- Distance percentile
- KL divergence
"""
import generate_data
import settings
import numpy as np
import exp_letter_test_NN
import pickle
import logging
import os
from scipy.spatial import distance
import lion_tsne
from scipy import stats
import neural_network_commons


distance_matrix_dir_prefix = '../data/UpdatedPMatrices-letters'


def generate_nn_kl_temp_filename(parameters):
    output_file_prefix = '../results/letter_nn_kl_temp_'
    return output_file_prefix + generate_data.combine_prefixes(
        neural_network_commons.nn_model_prefixes | settings.letter_parameter_set, parameters)


def generate_nn_postprocess_filename(parameters):
    output_file_prefix = '../results/letter_nn_postprocess_'
    return output_file_prefix + generate_data.combine_prefixes(
        neural_network_commons.nn_model_prefixes | settings.letter_parameter_set, parameters)


def main(parameters = settings.parameters):
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    X_mnist = generate_data.load_y_mnist(parameters=parameters)

    letter_samples, _, _ = generate_data.load_letters(parameters=parameters)

    nn_results_file = exp_letter_test_NN.generate_letter_results_filename(parameters)
    with open(nn_results_file, 'rb') as f:
        nn_letters_results, nn_models_orig, nn_method_list = pickle.load(f)

    D_Y = distance.squareform(distance.pdist(Y_mnist))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    nearest_neighbors_y_dist = np.min(D_Y, axis=1)  # Actually, whatever axis

    # ================ KL DIVERGENCE ===================
    nn_letters_kl = np.zeros((len(nn_method_list), len(letter_samples)))

    processed_indices = list()

    kl_nn_letters_performance_file = generate_nn_kl_temp_filename(parameters)

    # KL divergence increase for all 1000 samples is very slow to calculate. Main part of that is calculating P-matrix.
    per_sample_KL = np.zeros((len(letter_samples),))
    for i in range(len(letter_samples)):
        if i in processed_indices:
            logging.info("Sample %d already processed. Results loaded.", i)
            continue
        logging.info("Processing sample %d", i)
        distance_matrix_dir = distance_matrix_dir_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.letter_parameter_set, parameters, os.sep)
        distance_matrix_file = distance_matrix_dir + 'item' + str(i) + '.p'
        # Make sure you can load them one-by-one.
        if os.path.isfile(distance_matrix_file):
            logging.info("\tP-matrix file found. Loading.")
            with open(distance_matrix_file, 'rb') as f:
                new_P, _ = pickle.load(f)
        else:
            logging.info("\tP-matrix file not found. Creating and saving.")
            new_X = np.concatenate((X_mnist, letter_samples[i, :].reshape((1, -1))), axis=0)
            new_D = distance.squareform(distance.pdist(new_X))
            new_P, new_sigmas = lion_tsne.get_p_and_sigma(distance_matrix=new_D, perplexity=dTSNE_mnist.perplexity)
            with open(distance_matrix_file, 'wb') as f:
                pickle.dump((new_P, new_sigmas), f)
        # For all of methods P-matrix is shared.
        for j in range(len(nn_letters_results)):
            # Single file with p matrix
            new_Y = np.concatenate((nn_models_orig[j], nn_letters_results[j][i, :].reshape((1, -1))), axis=0)
            nn_letters_kl[j, i], _ = lion_tsne.kl_divergence_and_gradient(p_matrix=new_P, y=new_Y)
        processed_indices.append(i)
        with open(kl_nn_letters_performance_file, 'wb') as f:
            pickle.dump((nn_letters_kl, processed_indices), f)
    # This should be fast
    nn_avg_letters_kl = np.mean(nn_letters_kl, axis=1)

    # ================ DISTANCE MATRICES ===================
    nn_letters_percentiles_matrix = np.zeros((len(letter_samples), len(nn_method_list)))
    nn_letters_distance_matrix = np.zeros((len(letter_samples), len(nn_method_list)))
    for i in range(len(letter_samples)):
        for j in range(len(nn_method_list)):
            y = nn_letters_results[j][i, :]
            nn_dist = np.min(np.sqrt(np.sum((nn_models_orig[j] - y) ** 2, axis=1)))
            nn_letters_distance_matrix[i, j] = nn_dist
            nn_letters_percentiles_matrix[i, j] = stats.percentileofscore(nearest_neighbors_y_dist, nn_dist)
    nn_letters_distance_percentiles = np.mean(nn_letters_percentiles_matrix, axis=0)
    nn_letters_distances = np.mean(nn_letters_distance_matrix, axis=0)
    for j in range(len(nn_method_list)):
        print(nn_method_list[j], nn_letters_distances[j], nn_letters_distance_percentiles[j])


    output_file = generate_nn_postprocess_filename(parameters)
    with open(output_file, "wb") as f:
        pickle.dump((nn_method_list, nn_avg_letters_kl, nn_letters_distance_percentiles), f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parameters=settings.parameters)