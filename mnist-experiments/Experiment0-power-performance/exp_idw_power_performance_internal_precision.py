import settings
import generate_data
import numpy as np
import datetime
import logging
import pickle
from scipy.spatial import distance
import os

idw_power_options = np.arange(0.1, 200.1, step=0.1)
idw_percentile_options = [90, 95, 99, 100]
idw_power_performance_file_prefix = '../results/idw_power_internal_precision_'
idw_power_plot_file_prefix = '../results/idw_power_plot_internal_precision_'


def generate_idw_power_filename(parameters=settings.parameters):
    return idw_power_performance_file_prefix +\
                    generate_data.combine_prefixes(settings.x_neighbors_selection_parameter_set, parameters)


def generate_idw_power_plot_filename(parameters=settings.parameters):
    return idw_power_plot_file_prefix + generate_data.combine_prefixes(settings.nn_accuracy_parameter_set, parameters)


def load_idw_power_performance(*, regenerate=False, recursive_regenerate=False, parameters=settings.parameters):
    idw_power_plot_data_file = generate_idw_power_filename(parameters)
    if not os.path.isfile(idw_power_plot_data_file) or regenerate:
        generate_idw_power_performance(regenerate=True,
                                        recursive_regenerate=recursive_regenerate, parameters=parameters)
    with open(idw_power_plot_data_file, 'rb') as f:
        return pickle.load(f)


def load_idw_power_plot(*, regenerate=False, recursive_regenerate=False, parameters=settings.parameters):
    idw_power_plot_data_file = generate_idw_power_plot_filename(parameters)
    if not os.path.isfile(idw_power_plot_data_file) or regenerate:
        generate_idw_power_performance(regenerate=True,
                                        recursive_regenerate=recursive_regenerate, parameters=parameters)
    with open(idw_power_plot_data_file, 'rb') as f:
        return pickle.load(f)


def get_nearest_neighbors(y, Y_mnist, exclude_index, n):
    y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
    y_distances[exclude_index] = np.inf
    return np.argsort(y_distances)[:n]

def generate_idw_power_performance(*, regenerate=False, recursive_regenerate=False, parameters=settings.parameters):
    global_idw_precision_by_y = dict()
    global_idw_precision_by_x = dict()

    start_time = datetime.datetime.now()
    logging.info("IDW internal precision power experiment started: %s", start_time)
    idw_power_performance_file = generate_idw_power_filename(parameters)
    idw_power_plot_file = generate_idw_power_plot_filename(parameters)

    X_mnist = generate_data.load_x_mnist(parameters=parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)
    precision_nn = parameters.get("precision_nn", settings.parameters["precision_nn"])

    distance_matrix = distance.squareform(distance.pdist(X_mnist))
    np.fill_diagonal(distance_matrix, np.inf)  # We are not interested in distance to itself
    nn_x_distance = np.min(distance_matrix, axis=1)  # Any axis will do
    radius_x = dict()
    for p in idw_percentile_options:
        radius_x[p] = np.percentile(nn_x_distance, p)

    if os.path.isfile(idw_power_performance_file) and not regenerate:
        with open(idw_power_performance_file, 'rb') as f:
            global_idw_precision_by_x, global_idw_precision_by_y = pickle.load(f)
    else:
        logging.info("Regeneration requested")

    for p in idw_power_options:
        if p in global_idw_precision_by_x:
            logging.info("Loaded p %f %f %f", p, global_idw_precision_by_x[p], global_idw_precision_by_y[p])
            continue

        logging.info("Processing p %f", p)

        per_sample_precision_x = list()
        per_sample_precision_y = list()
        for i in range(len(X_mnist)):
            distances = distance_matrix[i, :].copy()
            # distances[i] = np.inf #Not interested in distance to itself
            # Step 1. Find nearest neighbors in the neighborhood.
            neighbor_indices = list(range(X_mnist.shape[0]))
            neighbor_indices.remove(i)
            weights = 1 / distances[neighbor_indices] ** p
            weights = weights / np.sum(weights)
            cur_y_result = weights.dot(Y_mnist[neighbor_indices, :])

            nn_xreal_indices = get_nearest_neighbors(X_mnist[i,:], X_mnist, n=precision_nn, exclude_index=i)
            nn_yreal_indices = get_nearest_neighbors(Y_mnist[i,:], Y_mnist, n=precision_nn, exclude_index=i)
            nn_yembedded_indices = get_nearest_neighbors(cur_y_result, Y_mnist, n=precision_nn, exclude_index=i)
            matching_indices_xreal_yembedded = len([j for j in nn_xreal_indices if j in nn_yembedded_indices])
            matching_indices_yreal_yembedded = len([j for j in nn_yreal_indices if j in nn_yembedded_indices])
            per_sample_precision_x.append(matching_indices_xreal_yembedded / precision_nn)
            per_sample_precision_y.append(matching_indices_yreal_yembedded / precision_nn)

        global_idw_precision_by_x[p] = np.mean(per_sample_precision_x)
        global_idw_precision_by_y[p] = np.mean(per_sample_precision_y)

        # Just in case it will become unstable due to too few neighbors
        # lion_power_plot_data[(p, perc)]['PowerSquareDistSum'] = y_sum_square_dist
        # lion_power_plot_data[(p, perc)]['PowerSquareDistCount'] = y_count

        with open(idw_power_performance_file, 'wb') as f:
            pickle.dump((global_idw_precision_by_x, global_idw_precision_by_y), f)

    EPS = 1e-5
    y = list()
    x_global = list()
    for cur_power in idw_power_options:
        closest_power = [i for i in global_idw_precision_by_x if np.abs(i - cur_power) < EPS]
        if len(closest_power) > 0:
            x_global.append(cur_power)
            y.append(global_idw_precision_by_x[closest_power[0]])
    idw_optimal_power_precision_by_x = x_global[np.argmax(y)]
    precision_plot_by_x = y

    EPS = 1e-5
    y = list()
    x_global = list()
    for cur_power in idw_power_options:
        closest_power = [i for i in global_idw_precision_by_y if np.abs(i - cur_power) < EPS]
        if len(closest_power) > 0:
            x_global.append(cur_power)
            y.append(global_idw_precision_by_y[closest_power[0]])
    idw_optimal_power_precision_by_y = x_global[np.argmax(y)]
    precision_plot_by_y = y

    with open(idw_power_plot_file, 'wb') as f:
        pickle.dump((x_global, precision_plot_by_x, precision_plot_by_y, idw_optimal_power_precision_by_x, idw_optimal_power_precision_by_y), f)
    logging.info("IDW optimal power (precision by X): %f", idw_optimal_power_precision_by_x)
    logging.info("IDW optimal power (precision by Y): %f", idw_optimal_power_precision_by_y)

    end_time = datetime.datetime.now()
    logging.info("IDW internal precision power experiment ended: %s", end_time)
    logging.info("IDW internal precision power experiment duration: %s", end_time - start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_idw_power_performance(regenerate=False)
