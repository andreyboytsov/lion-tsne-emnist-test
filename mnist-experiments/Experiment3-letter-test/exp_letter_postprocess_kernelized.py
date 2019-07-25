"""
Post-processing cluster attribution test results for NN:
- Accuracy
- Distance percentile
- KL divergence
"""
import generate_data
import settings
import numpy as np
import exp_letter_test_kernelized
import pickle
import logging
import os
from scipy.spatial import distance
import lion_tsne
from scipy import stats

distance_matrix_dir_prefix = '../data/UpdatedPMatrices-letters'


def generate_kernelized_kl_temp_filename(parameters):
    output_file_prefix = '../results/letter_kernelized_kl_temp_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_parameter_set, parameters)


def generate_kernelized_postprocess_filename(parameters):
    output_file_prefix = '../results/letter_kernelized_postprocess_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_parameter_set, parameters)


def main(parameters = settings.parameters, regenerate = False):
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    letter_samples, _, _ = generate_data.load_letters(parameters=parameters)
    X_mnist = generate_data.load_x_mnist(parameters=parameters)

    D_Y = distance.squareform(distance.pdist(Y_mnist))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    nearest_neighbors_y_dist = np.min(D_Y, axis=1)  # Actually, whatever axis

    kernelized_results_file = exp_letter_test_kernelized.generate_letter_results_filename(parameters)
    with open(kernelized_results_file, 'rb') as f:
        kernelized_detailed_method_results, kernelized_detailed_tsne_time, kernelized_detailed_method_list = pickle.load(f)
    ind = [4, 24, 49]
    
    kernelized_method_list = [
        kernelized_detailed_method_list[i][:10] + kernelized_detailed_method_list[i][-8:]
        for i in ind]
    kernelized_letters_results = [kernelized_detailed_method_results[i] for i in ind]

    # =========== DISTANCE PERCENTILES ==========
    kernelized_letters_percentiles_matrix = np.zeros((len(letter_samples), len(kernelized_method_list)))
    kernelized_letters_distance_matrix = np.zeros((len(letter_samples), len(kernelized_method_list)))
    for i in range(len(letter_samples)):
        for j in range(len(kernelized_method_list)):
            y = kernelized_letters_results[j][i,:]
            nn_dist = np.min(np.sqrt(np.sum((Y_mnist-y)**2, axis=1)))
            kernelized_letters_distance_matrix[i,j] = nn_dist
            kernelized_letters_percentiles_matrix[i,j] = stats.percentileofscore(nearest_neighbors_y_dist, nn_dist)
    kernelized_letters_distance_percentiles = np.mean(kernelized_letters_percentiles_matrix, axis=0)
    kernelized_letters_distances = np.mean(kernelized_letters_distance_matrix, axis=0)
    kernelized_per_item_time = kernelized_detailed_tsne_time / len(letter_samples)
    for j in range(len(kernelized_method_list)):
        logging.info("%s: %f, %f", kernelized_method_list[j], kernelized_letters_distances[j],
              kernelized_letters_distance_percentiles[j])

    kernelized_letters_kl = np.zeros((len(kernelized_method_list), len(letter_samples)))
    processed_indices = list()

    kl_kernelized_tsne_letters_performance_file = generate_kernelized_kl_temp_filename(parameters)
    if os.path.isfile(kl_kernelized_tsne_letters_performance_file) and not regenerate:
        with open(kl_kernelized_tsne_letters_performance_file, 'rb') as f:
            kernelized_letters_kl, processed_indices = pickle.load(f)

    # =========== KL DIVERGENCE ==========
    # KL divergence increase for all 1000 samples is very slow to calculate. Main part of that is calculating P-matrix.
    per_sample_KL = np.zeros((len(letter_samples),))
    for i in range(len(letter_samples)):
        if i in processed_indices:
            logging.info("Sample %d already processed. Results loaded.",i)
            continue
        logging.info("Processing sample %d", i)
        distance_matrix_dir = distance_matrix_dir_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.letter_parameter_set, parameters, os.sep)
        distance_matrix_file = distance_matrix_dir + 'item' + str(j) + '.p'
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
        for j in range(len(kernelized_letters_results)):
            # Single file with p matrix
            new_Y = np.concatenate((Y_mnist, kernelized_letters_results[j][i, :].reshape((1, -1))), axis=0)
            kernelized_letters_kl[j, i], _ = lion_tsne.kl_divergence_and_gradient(p_matrix=new_P, y=new_Y)
        processed_indices.append(i)
        with open(kl_kernelized_tsne_letters_performance_file, 'wb') as f:
            pickle.dump((kernelized_letters_kl, processed_indices), f)
    # This should be fast
    kernelized_avg_letters_kl = np.mean(kernelized_letters_kl, axis=1)

    output_file = generate_kernelized_postprocess_filename(parameters)
    with open(output_file, "wb") as f:
        pickle.dump((kernelized_method_list, kernelized_avg_letters_kl,
                     kernelized_per_item_time, kernelized_letters_distance_percentiles),f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parameters=settings.parameters, regenerate = True)
