"""
EXPERIMENT:

Cluster attribution test, RBF interpolation.
"""
import logging
import pickle
import datetime
import settings
import numpy as np
from scipy import stats
from scipy.spatial import distance

import generate_data
import exp_idw_power_performance
import exp_lion_power_performance
import os
import lion_tsne

cluster_results_file_prefix = '../results/cluster_attr_'

distance_matrix_dir_prefix = '../data/UpdatedPMatrices'

rbf_functions = ('multiquadric', 'linear', 'cubic', 'quintic', 'gaussian',
                 'inverse', 'thin-plate')
idw_powers = (1, 10, 20, 40)
lion_percentiles = (90, 95, 99, 100)
n_digits = 1

# TODO make parallelizeable?


def get_nearest_neighbors_in_y(y, Y_mnist, n=10):
    y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
    return np.argsort(y_distances)[:n]


def calc_accuracy(*, common_info, embedded_neighbors, parameters):
    picked_neighbors = common_info['picked_neighbors']
    accuracy_nn = common_info["accuracy_nn"]
    picked_neighbors_labels = common_info["picked_neighbors_labels"]
    labels_mnist = common_info["labels_mnist"]
    Y_mnist = common_info["Y_mnist"]
    per_sample_accuracy = list()
    for j in range(len(picked_neighbors)):
        y = embedded_neighbors[j, :]
        expected_label = picked_neighbors_labels[j]
        nn_indices = get_nearest_neighbors_in_y(y, Y_mnist, n=accuracy_nn)
        obtained_labels = labels_mnist[nn_indices]
        per_sample_accuracy.append(sum(obtained_labels == expected_label) / len(obtained_labels))
    return np.mean(per_sample_accuracy)


def calc_distance_perc(*, common_info, embedded_neighbors, parameters):
    picked_neighbors = common_info['picked_neighbors']
    Y_mnist = common_info["Y_mnist"]
    nearest_neighbors_y_dist = common_info["nearest_neighbors_y_dist"]
    per_sample_nearest_neighbors_percentiles = list()
    for j in range(len(picked_neighbors)):
        y = embedded_neighbors[j, :]
        nn_dist = np.min(np.sqrt(np.sum((Y_mnist - y) ** 2, axis=1)))
        per_sample_nearest_neighbors_percentiles.append(stats.percentileofscore(nearest_neighbors_y_dist, nn_dist))
    return np.mean(per_sample_nearest_neighbors_percentiles)


def calc_kl(*, common_info, embedded_neighbors, parameters):
    dTSNE_mnist = common_info["dTSNE_mnist"]
    X_mnist = common_info["X_mnist"]
    Y_mnist = common_info["Y_mnist"]
    picked_neighbors = common_info["picked_neighbors"]
    per_sample_kl_divergences = list()
    for j in range(len(picked_neighbors)):
        distance_matrix_dir = distance_matrix_dir_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters, os.sep)
        distance_matrix_file = distance_matrix_dir + 'item' + str(j) + '.p'
        # Don't store those matrices in a single file. Way too large.

        # Make sure you can load them one-by-one.
        if os.path.isfile(distance_matrix_file):
            if j%50==0:
                logging.info("\t%d P-matrix file found. Loading.", j)
            with open(distance_matrix_file, 'rb') as f:
                new_P, _ = pickle.load(f)
        else:
            if j%50==0:
                logging.info("\t%d P-matrix file not found. Creating and saving.", j)
            new_X = np.concatenate((X_mnist, picked_neighbors[j, :].reshape((1, -1))), axis=0)
            new_D = distance.squareform(distance.pdist(new_X))
            new_P, new_sigmas = lion_tsne.get_p_and_sigma \
                (distance_matrix=new_D, perplexity=dTSNE_mnist.perplexity)
            if not os.path.isdir(distance_matrix_dir):
                logging.info('Creating directory: %s', distance_matrix_dir)
                os.mkdir(distance_matrix_dir)
            with open(distance_matrix_file, 'wb') as f:
                pickle.dump((new_P, new_sigmas), f)
        # Single file with p matrix.
        # Now use it to calculate KL divergence.
        new_Y = np.concatenate((Y_mnist, embedded_neighbors[j, :].reshape((1, -1))), axis=0)
        kl, _ = lion_tsne.kl_divergence_and_gradient(p_matrix=new_P, y=new_Y)
        per_sample_kl_divergences.append(kl)
    return np.mean(per_sample_kl_divergences)


features_to_functions = {
    "KL-Divergence": calc_kl,
    "Accuracy": calc_accuracy,
    "DistancePercentile": calc_distance_perc,
}


def generate_all_embedders(dTSNE_mnist):
    _, _, idw_optimal_power = exp_idw_power_performance.load_idw_power_plot()
    _, _, lion_optimal_powers = exp_lion_power_performance.load_lion_power_plot()

    embedders = dict()
    for p in lion_percentiles:
        embedders["LION-"+str(p)+"-"+str(round(lion_optimal_powers[p], n_digits))] = \
            dTSNE_mnist.generate_embedding_function(
                    function_kwargs={'radius_x_percentile':p, 'power': lion_optimal_powers[p]})
        logging.info("Generated embedder LION-%d (%f)",p, lion_optimal_powers[p])
    for i in rbf_functions:
        embedders["RBF-"+i] = dTSNE_mnist.generate_embedding_function(embedding_function_type='rbf',
                                                                     function_kwargs={'function': i})
        logging.info("Generated embedder RBF-%s",i)
    for p in idw_powers + (idw_optimal_power, ):
        embedders["IDW-"+str(round(p, n_digits))] = dTSNE_mnist.generate_embedding_function(
            embedding_function_type='weighted-inverse-distance', function_kwargs={'power': p})
        logging.info("Generated embedder IDW-%f",round(p, n_digits))
    return embedders


def get_common_info(parameters):
    res = {}
    res['dTSNE_mnist'] = generate_data.load_dtsne_mnist(parameters=parameters)
    res['Y_mnist'] = generate_data.load_y_mnist(parameters=parameters)
    res['X_mnist'] = generate_data.load_x_mnist(parameters=parameters)
    res['labels_mnist'] = generate_data.load_labels_mnist(parameters=parameters)
    res['picked_neighbors'] = generate_data.load_picked_neighbors(parameters=parameters)
    res['picked_neighbors_labels'] = generate_data.load_picked_neighbors_labels(parameters=parameters)
    res['accuracy_nn'] = parameters.get("accuracy_nn", parameters["accuracy_nn"])
    D_Y = distance.squareform(distance.pdist(res['Y_mnist']))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    res['nearest_neighbors_y_dist'] = np.min(D_Y, axis=1)  # Actually, whatever axis
    return res


def process_single_embedder(*, embedder, embedder_name, results, regenerate, common_info,
                            cluster_results_file, parameters):
    if embedder_name not in results:
        results[embedder_name] = {}

    logging.info("Trying embedder %s", embedder_name)

    need_embedding = ("TimePerPoint" not in results[embedder_name]) or\
                     ("EmbeddedPoints" not in results[embedder_name]) or regenerate
    save_embedding = ("EmbeddedPoints" not in results[embedder_name]) or regenerate
    save_time = ("TimePerPoint" not in results[embedder_name]) or regenerate
    logging.info("Embedding is%srequired", " " if need_embedding else " NOT ")

    embedder_start_time = datetime.datetime.now()
    embedded_neighbors = embedder(common_info['picked_neighbors'])\
        if need_embedding else results[embedder_name]["EmbeddedPoints"]
    embedder_end_time = datetime.datetime.now()

    results[embedder_name]["TimePerPoint"] = (embedder_end_time - embedder_start_time) / len(embedded_neighbors) if save_time \
        else results[embedder_name]["TimePerPoint"]
    results[embedder_name]["EmbeddedPoints"] = embedded_neighbors
    logging.info("Time %s", "SAVED" if save_time else "KEPT")
    logging.info("Embedding %s", "SAVED" if save_embedding else "KEPT")

    for feat in features_to_functions:
        if (feat not in results[embedder_name]) or regenerate:
            logging.info("Calculating %s...", feat)
            results[embedder_name][feat] = features_to_functions[feat](common_info=common_info,
                                                           embedded_neighbors=results[embedder_name]["EmbeddedPoints"],
                                                           parameters=parameters)
            logging.info("Finished calculating %s: %s", feat, results[embedder_name][feat])
        else:
            logging.info("%s loaded: %s", feat, results[embedder_name][feat])
    logging.info("Time to embed a single point: %s", results[embedder_name]["TimePerPoint"])

    with open(cluster_results_file, 'wb') as f:
        pickle.dump(results, f)


def main(*, regenerate=False, parameters=settings.parameters):
    start_time = datetime.datetime.now()
    logging.info("IDW/RBF cluster attribution experiment started: %s", start_time)
    cluster_results_file = cluster_results_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)

    common_info = get_common_info(parameters)
    results = dict()
    embedders = generate_all_embedders(common_info['dTSNE_mnist'])

    if os.path.isfile(cluster_results_file):
        with open(cluster_results_file, 'rb') as f:
            results = pickle.load(f)

    for embedder_name in embedders.keys():
        process_single_embedder(embedder=embedders[embedder_name], embedder_name=embedder_name, results=results,
                regenerate=regenerate, common_info=common_info, cluster_results_file=cluster_results_file,
                                parameters=parameters)

    end_time = datetime.datetime.now()
    logging.info("Cluster attribution experiment ended: %s", end_time)
    logging.info("Cluster attribution experiment duration: %s", end_time-start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(regenerate=False)
# # Doing it from scratch takes REALLY long time. If possible, save results & pre-load
#
# # These are consequences of parallelization
# input_files = ['gd_results' + str(100 * i) + '_' + str(100 * i + 100) + '.p' for i in range(10)]
# output_file = 'gd_results.p'
# covered_samples = list()
#
# first_sample_inc = 0  # Change only if it is one of "Other notebooks just for parallelization"
# last_sample_exclusive = len(picked_neighbors)
#
# # Let's build all possible combinations. Later we'll decide what to plot
# picked_neighbors_y_gd_transformed = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
# picked_neighbors_y_gd_variance_recalc_transformed = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
# picked_neighbors_y_gd_transformed_random = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
# picked_neighbors_y_gd_variance_recalc_transformed_random = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
#
# picked_neighbors_y_gd_early_exagg_transformed_random = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
# picked_neighbors_y_gd_early_exagg_transformed = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
# picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random = np.zeros(
#     (len(picked_neighbors), Y_mnist.shape[1]))
# picked_neighbors_y_gd_variance_recalc_early_exagg_transformed = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
#
# picked_random_starting_positions = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
#
# for fname in input_files:
#     if not os.path.isfile(fname):
#         print("File not found: ", fname)
#     else:
#         with open(fname, 'rb') as f:
#             cur_gd_transformed, cur_gd_variance_recalc_transformed, \
#             cur_gd_transformed_random, cur_gd_variance_recalc_transformed_random, \
#             cur_gd_early_exagg_transformed_random, cur_gd_early_exagg_transformed, \
#             cur_gd_variance_recalc_early_exagg_transformed_random, cur_random_starting_positions, \
#             cur_gd_variance_recalc_early_exagg_transformed, cur_covered_samples = pickle.load(f)
#         print(cur_covered_samples)
#         # Each files covers only part of samples
#         for i in cur_covered_samples:
#             print("\tLoaded sample ", i)
#             picked_neighbors_y_gd_transformed[i, :] = cur_gd_transformed[i, :]
#             picked_neighbors_y_gd_variance_recalc_transformed[i, :] = cur_gd_variance_recalc_transformed[i, :]
#             picked_random_starting_positions[i, :] = cur_random_starting_positions[i, :]
#             picked_neighbors_y_gd_transformed_random[i, :] = cur_gd_transformed_random[i, :]
#             picked_neighbors_y_gd_variance_recalc_transformed_random[i, :] = cur_gd_variance_recalc_transformed_random[
#                                                                              i, :]
#             picked_neighbors_y_gd_early_exagg_transformed_random[i, :] = cur_gd_early_exagg_transformed_random[i, :]
#             picked_neighbors_y_gd_early_exagg_transformed[i, :] = picked_neighbors_y_gd_early_exagg_transformed[i, :]
#             picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random[i, :] = \
#                 cur_gd_variance_recalc_early_exagg_transformed_random[i, :]
#             picked_neighbors_y_gd_variance_recalc_early_exagg_transformed[i, :] = \
#                 cur_gd_variance_recalc_early_exagg_transformed[i, :]
#             covered_samples.append(i)
#
# for i in range(first_sample_inc, last_sample_exclusive):
#     np.random.seed(i)  # We reset random seed every time. Otherwise, if you load partial results from file, everything
#     # will depend on which parts were loaded, random sequence will "shift" depend on that, and reproducibility will be lost.
#     # I.e. if put seed(0) before the loop and start from scratch, then you will have some random sequence [abc] for sample 0,
#     # other (continuation of that sequence) [def] for sample 1, etc. But if you already loaded sample 0 from file, you will
#     # have [abc] for sample 1, [def] for sample 2, etc. Reproducibility should not depend on what parts are loaded.
#     # Hence, random seed every time, and it depends on ABSOLUTE sample number.
#     print(" ====================== Sample ", i, "\n\n")
#     if i in covered_samples:
#         print("Already loaded.")
#     else:
#         neighbor = picked_neighbors[i].reshape((1, -1))
#         picked_neighbors_y_gd_transformed[i, :] = dTSNE_mnist.transform(neighbor, y='closest',
#                                                                         verbose=2,
#                                                                         optimizer_kwargs={'early_exaggeration': None})
#         picked_neighbors_y_gd_variance_recalc_transformed[i, :] = dTSNE_mnist.transform(neighbor, keep_sigmas=False,
#                                                                                         y='closest',
#                                                                                         verbose=2, optimizer_kwargs={
#                 'early_exaggeration': None})
#
#         # Let's pick random starts at any point. not necessary near the center.
#         y_start = np.array([[
#             np.random.uniform(np.min(Y_mnist[:, 0]), np.max(Y_mnist[:, 0])),
#             np.random.uniform(np.min(Y_mnist[:, 1]), np.max(Y_mnist[:, 1]))
#         ]])
#
#         picked_random_starting_positions[i, :] = y_start
#
#         picked_neighbors_y_gd_transformed_random[i, :] = dTSNE_mnist.transform(neighbor, y=y_start,  # y='random',
#                                                                                verbose=2, optimizer_kwargs={
#                 'early_exaggeration': None})
#         picked_neighbors_y_gd_variance_recalc_transformed_random[i, :] = dTSNE_mnist.transform(neighbor,
#                                                                                                keep_sigmas=False,
#                                                                                                y=y_start,  # y='random',
#                                                                                                verbose=2,
#                                                                                                optimizer_kwargs={
#                                                                                                    'early_exaggeration': None})
#
#         picked_neighbors_y_gd_early_exagg_transformed_random[i, :] = dTSNE_mnist.transform(neighbor, y=y_start,
#                                                                                            # y='random',
#                                                                                            verbose=2)
#         picked_neighbors_y_gd_early_exagg_transformed[i, :] = dTSNE_mnist.transform(neighbor, y='closest', verbose=2)
#
#         picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random[i, :] = dTSNE_mnist.transform(neighbor,
#                                                                                                            y=y_start,
#                                                                                                            keep_sigmas=False,
#                                                                                                            verbose=2)
#         picked_neighbors_y_gd_variance_recalc_early_exagg_transformed[i, :] = dTSNE_mnist.transform(neighbor,
#                                                                                                     keep_sigmas=False,
#                                                                                                     y='closest',
#                                                                                                     verbose=2)
#
#         covered_samples.append(i)
#     # Re-saving even if it is a loaded sample
#     print("Saving...")
#     # Gradient descent results take a long while. Let's cache.
#     with open(output_file, 'wb') as f:
#         pickle.dump((picked_neighbors_y_gd_transformed, picked_neighbors_y_gd_variance_recalc_transformed,
#                      picked_neighbors_y_gd_transformed_random, picked_neighbors_y_gd_variance_recalc_transformed_random,
#                      picked_neighbors_y_gd_early_exagg_transformed_random,
#                      picked_neighbors_y_gd_early_exagg_transformed,
#                      picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random,
#                      picked_random_starting_positions,
#                      picked_neighbors_y_gd_variance_recalc_early_exagg_transformed, covered_samples), f)
