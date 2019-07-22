import settings
import generate_data
import numpy as np
import datetime
import logging
import pickle
from scipy.spatial import distance
import os

idw_power_options = np.arange(0.1, 50.1, step=0.1)
idw_percentile_options = [90, 95, 99, 100]
idw_power_performance_file_prefix = '../results/idw_power'
idw_power_plot_file_prefix = '../results/idw_power_plot'


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


def generate_idw_power_performance(*, regenerate=False, recursive_regenerate=False, parameters=settings.parameters):
    global_idw_power_performance = dict()  # Start from scratch
    global_idw_power_performance_abs = dict()  # Start from scratch
    global_idw_accuracy = dict()

    start_time = datetime.datetime.now()
    logging.info("IDW power experiment started: %s", start_time)
    idw_power_performance_file = generate_idw_power_filename(parameters)
    idw_power_plot_file = generate_idw_power_plot_filename(parameters)

    X_mnist = generate_data.load_x_mnist(parameters=parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=parameters)
    picked_neighbor_labels = generate_data.load_picked_neighbors_labels(parameters=parameters)
    labels_mnist = generate_data.load_labels_mnist(parameters=settings.parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)
    accuracy_nn = parameters.get("accuracy_nn", settings.parameters["accuracy_nn"])

    def get_nearest_neighbors_in_y(y, n=10):
        y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
        return np.argsort(y_distances)[:n]

    distance_matrix = distance.squareform(distance.pdist(X_mnist))
    np.fill_diagonal(distance_matrix, np.inf)  # We are not interested in distance to itself
    nn_x_distance = np.min(distance_matrix, axis=1)  # Any axis will do
    radius_x = dict()
    for p in idw_percentile_options:
        radius_x[p] = np.percentile(nn_x_distance, p)

    if os.path.isfile(idw_power_performance_file) and not regenerate:
        with open(idw_power_performance_file, 'rb') as f:
            global_idw_power_performance, global_idw_power_performance_abs, global_idw_accuracy = pickle.load(f)
    else:
        logging.info("Regeneration requested")

    for p in idw_power_options:

        interpolator = dTSNE_mnist.generate_embedding_function(
            embedding_function_type='weighted-inverse-distance',
            function_kwargs={'power': p})

        per_sample_accuracy = np.zeros((len(picked_neighbors),))
        for i in range(len(picked_neighbors)):
            expected_label = picked_neighbor_labels[i]
            result = interpolator(picked_neighbors[i], verbose=0)
            nn_indices = get_nearest_neighbors_in_y(result, n=accuracy_nn)
            obtained_labels = labels_mnist[nn_indices]
            per_sample_accuracy[i] = sum(obtained_labels == expected_label) / len(obtained_labels)
        cur_acc = np.mean(per_sample_accuracy)

        if p in global_idw_power_performance:
            logging.info("Loaded p %f", p)
            continue
        logging.info("Processing p %f", p)
        y_sum_square_dist = 0.0
        y_sum_abs_dist = 0.0
        y_abs_dist = 0.0
        y_count = 0.0
        for i in range(len(X_mnist)):
            distances = distance_matrix[i, :].copy()
            # distances[i] = np.inf #Not interested in distance to itself
            # Step 1. Find nearest neighbors in the neighborhood.
            neighbor_indices = list(range(X_mnist.shape[0]))
            neighbor_indices.remove(i)
            num_neighbors = len(neighbor_indices)
            weights = 1 / distances[neighbor_indices] ** p
            weights = weights / np.sum(weights)
            cur_y_result = weights.dot(Y_mnist[neighbor_indices, :])
            y_sum_square_dist += np.sum(cur_y_result - Y_mnist[i, :]) ** 2
            y_sum_abs_dist += np.sqrt(np.sum(cur_y_result - Y_mnist[i, :]) ** 2)
            y_count += 1.0

        global_idw_power_performance[p] = y_sum_square_dist / y_count
        global_idw_power_performance_abs[p] = y_sum_abs_dist / y_count
        global_idw_accuracy[p] = cur_acc

        # Just in case it will become unstable due to too few neighbors
        # lion_power_plot_data[(p, perc)]['PowerSquareDistSum'] = y_sum_square_dist
        # lion_power_plot_data[(p, perc)]['PowerSquareDistCount'] = y_count

        with open(idw_power_performance_file, 'wb') as f:
            pickle.dump((global_idw_power_performance, global_idw_power_performance_abs, global_idw_accuracy), f)

    EPS = 1e-5
    y = list()
    x_global = list()
    for cur_power in idw_power_options:
        closest_power = [i for i in global_idw_power_performance_abs if np.abs(i - cur_power) < EPS]
        if len(closest_power) > 0:
            x_global.append(cur_power)
            y.append(global_idw_power_performance[closest_power[0]])
    idw_optimal_power = x_global[np.argmin(y)]

    with open(idw_power_plot_file, 'wb') as f:
        pickle.dump((x_global, y, idw_optimal_power), f)
    logging.info("IDW optial power: %f", idw_optimal_power)

    end_time = datetime.datetime.now()
    logging.info("IDW power experiment ended: %s", end_time)
    logging.info("IDW power experiment duration: %s", end_time - start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_idw_power_performance(regenerate=True)
