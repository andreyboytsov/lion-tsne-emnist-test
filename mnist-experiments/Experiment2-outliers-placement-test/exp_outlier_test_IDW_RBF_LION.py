"""
EXPERIMENT:

Outlier placement test: RBF interpolation, IDW interpolation, LION interpolation.
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

outlier_results_file_prefix = '../results/outlier_'

distance_matrix_dir_prefix = '../data/UpdatedPMatrices-outliers'

rbf_functions = ('multiquadric', 'linear', 'cubic', 'quintic', 'gaussian',
                 'inverse', 'thin-plate')
idw_powers = (1, 10, 20, 40)
lion_percentiles = (90, 95, 99, 100)
n_digits = 1

# TODO make parallelizeable?


def get_nearest_neighbors_in_y(y, Y_mnist, n=10):
    y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
    return np.argsort(y_distances)[:n]


def calc_distance_perc(*, common_info, embedded_neighbors, parameters):
    outlier_samples = common_info["outlier_samples"]
    Y_mnist = common_info["Y_mnist"]
    nearest_neighbors_y_dist = common_info["nearest_neighbors_y_dist"]
    per_sample_nearest_neighbors_percentiles = list()
    for j in range(len(outlier_samples)):
        y = embedded_neighbors[j, :]
        nn_dist = np.min(np.sqrt(np.sum((Y_mnist - y) ** 2, axis=1)))
        per_sample_nearest_neighbors_percentiles.append(stats.percentileofscore(nearest_neighbors_y_dist, nn_dist))
    return np.mean(per_sample_nearest_neighbors_percentiles)


def calc_kl(*, common_info, embedded_neighbors, parameters):
    dTSNE_mnist = common_info["dTSNE_mnist"]
    X_mnist = common_info["X_mnist"]
    Y_mnist = common_info["Y_mnist"]
    outlier_samples = common_info["outlier_samples"]
    per_sample_kl_divergences = list()
    for j in range(len(outlier_samples)):
        distance_matrix_dir = distance_matrix_dir_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.outlier_parameter_set, parameters, os.sep)
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
            new_X = np.concatenate((X_mnist, outlier_samples[j, :].reshape((1, -1))), axis=0)
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
    res['X_mnist'] = generate_data.load_x_mnist(parameters=parameters)
    res['Y_mnist'] = generate_data.load_y_mnist(parameters=parameters)
    outlier_samples, _ = generate_data.load_outliers(parameters=settings.parameters)
    res['outlier_samples'] = outlier_samples
    D_Y = distance.squareform(distance.pdist(res['Y_mnist']))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    res['nearest_neighbors_y_dist'] = np.min(D_Y, axis=1)  # Actually, whatever axis
    return res


def process_single_embedder(*, embedder, embedder_name, results, regenerate, common_info,
                            outlier_results_file, parameters):
    if embedder_name not in results:
        results[embedder_name] = {}

    logging.info("Trying embedder %s", embedder_name)

    need_embedding = ("TimePerPoint" not in results[embedder_name]) or\
                     ("EmbeddedPoints" not in results[embedder_name]) or regenerate
    save_embedding = ("EmbeddedPoints" not in results[embedder_name]) or regenerate
    save_time = ("TimePerPoint" not in results[embedder_name]) or regenerate
    logging.info("Embedding is%srequired", " " if need_embedding else " NOT ")

    embedder_start_time = datetime.datetime.now()
    embedded_outliers = embedder(common_info['outlier_samples'])\
        if need_embedding else results[embedder_name]["EmbeddedPoints"]
    embedder_end_time = datetime.datetime.now()

    results[embedder_name]["TimePerPoint"] = (embedder_end_time - embedder_start_time) / len(embedded_outliers) if save_time \
        else results[embedder_name]["TimePerPoint"]
    results[embedder_name]["EmbeddedPoints"] = embedded_outliers
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

    with open(outlier_results_file, 'wb') as f:
        pickle.dump(results, f)


def generate_outlier_results_filename(parameters=settings.parameters):
    return outlier_results_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.outlier_parameter_set, parameters)


def main(*, regenerate=False, parameters=settings.parameters):
    start_time = datetime.datetime.now()
    logging.info("IDW/RBF/LION outlier experiment started: %s", start_time)
    outlier_results_file = generate_outlier_results_filename(parameters)

    common_info = get_common_info(parameters)
    results = dict()
    embedders = generate_all_embedders(common_info['dTSNE_mnist'])

    if os.path.isfile(outlier_results_file) and not regenerate:
        with open(outlier_results_file, 'rb') as f:
            results = pickle.load(f)

    for embedder_name in embedders.keys():
        process_single_embedder(embedder=embedders[embedder_name], embedder_name=embedder_name, results=results,
                regenerate=regenerate, common_info=common_info, outlier_results_file=outlier_results_file,
                                parameters=parameters)

    end_time = datetime.datetime.now()
    logging.info("Outlier experiment ended: %s", end_time)
    logging.info("Outlier experiment duration: %s", end_time-start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(regenerate=False)

