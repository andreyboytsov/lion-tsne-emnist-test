"""
Adding precision calculations.
"""

import numpy as np
import cluster_lion_RBF_IDW_commons
import generate_data
import settings
import pickle
import logging
import os

original_files_prefixes = [
    '../results/cluster_attr_RBF_IDW',
    '../results/cluster_attr_IDW_higher',
    '../results/cluster_attr_LION'
]

output_prefix = '../results/cluster_attr_precision_RBF_IDW_LION'


def get_nearest_neighbors(x, X_mnist, n=10):
    x_distances = np.sum((X_mnist - x) ** 2, axis=1)
    return np.argsort(x_distances)[:n]


def calc_precision(embedded_neighbors, X_mnist, Y_mnist, picked_neighbors, precision_nn):
    per_sample_precision = list()

    for j in range(len(picked_neighbors)):
        if j%200==0:
            logging.info(j)
        y = embedded_neighbors[j, :]
        x = picked_neighbors[j, :]
        nn_x_indices = get_nearest_neighbors(x, X_mnist, n=precision_nn)
        nn_y_indices = get_nearest_neighbors(y, Y_mnist, n=precision_nn)
        matching_indices = len([i for i in nn_x_indices if i in nn_y_indices])
        per_sample_precision.append(matching_indices / precision_nn)
    return np.mean(per_sample_precision)


def main(parameters=settings.parameters, regenerate=False):
    picked_neighbors = generate_data.load_picked_neighbors(parameters=parameters)
    precision_nn = parameters.get("precision_nn", settings.parameters["precision_nn"])
    X_mnist = generate_data.load_x_mnist(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)

    result = dict()

    output_file = \
        cluster_lion_RBF_IDW_commons.generate_cluster_results_filename(output_prefix, parameters)

    if os.path.isfile(output_file) and not regenerate:
        with open(output_file, "rb") as f:
            result = pickle.load(f)
            logging.info("Previous result loaded")
    else:
        logging.info("No previous result or regeneration requested")

    for fname_prefix in original_files_prefixes:
        cluster_results_file = \
            cluster_lion_RBF_IDW_commons.generate_cluster_results_filename(fname_prefix, parameters)
        logging.info("Processing file: %s", cluster_results_file)
        with open(cluster_results_file, 'rb') as f:
            res = pickle.load(f)
            for i in res.keys():
                logging.info("Processing method: %s", i)
                if i not in result or regenerate:

                    precision = calc_precision(res[i]["EmbeddedPoints"], X_mnist, Y_mnist, picked_neighbors,
                                               precision_nn)
                    logging.info("%s precision: %f (accuracy was %f)", i, precision, res[i]["Accuracy"])
                    result[i] = precision

                    with open(output_file, "wb") as f:
                        pickle.dump(result, f)


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main(parameters=settings.parameters, regenerate=True)
