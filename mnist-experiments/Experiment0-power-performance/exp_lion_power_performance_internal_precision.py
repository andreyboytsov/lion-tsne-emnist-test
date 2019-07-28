import settings
from scipy.spatial import distance
import logging
import datetime
import numpy as np
import generate_data
import pickle
import os

lion_power_options = np.arange(0.1, 40.1, step=0.1)
lion_percentile_options = [90, 95, 99, 100]
lion_power_performance_prefix = '../results/lion_power_internal_precision'
lion_power_plot_prefix = '../results/lion_power_plot_internal_precision'

regenerate_lion_accuracy = True


def generate_lion_power_performance_filename(parameters=settings.parameters):
    return lion_power_performance_prefix + generate_data.combine_prefixes(settings.nn_accuracy_parameter_set, parameters)


def generate_lion_power_plot_filename(parameters=settings.parameters):
    return lion_power_plot_prefix + generate_data.combine_prefixes(settings.nn_accuracy_parameter_set, parameters)


def load_lion_power_performance(*, regenerate=False, recursive_regenerate=False, parameters=settings.parameters):
    lion_power_plot_data_file = generate_lion_power_performance_filename(parameters)
    if not os.path.isfile(lion_power_plot_data_file) or regenerate:
        generate_lion_power_performance(regenerate=True,
                                        recursive_regenerate=recursive_regenerate, parameters=parameters)
    with open(lion_power_plot_data_file, 'rb') as f:
        return pickle.load(f)


def load_lion_power_plot(*, regenerate=False, recursive_regenerate=False, parameters=settings.parameters):
    lion_power_plot_data_file = generate_lion_power_plot_filename(parameters)
    if not os.path.isfile(lion_power_plot_data_file) or regenerate:
        generate_lion_power_performance(regenerate=True,
                                        recursive_regenerate=recursive_regenerate, parameters=parameters)
    with open(lion_power_plot_data_file, 'rb') as f:
        return pickle.load(f)


def generate_lion_power_performance(*, regenerate=False, recursive_regenerate=False, parameters=settings.parameters):
    start_time = datetime.datetime.now()
    logging.info("LION power internal precision experiment started: %s", start_time)

    precision_nn = parameters.get("precision_nn", settings.parameters["precision_nn"])

    lion_power_performance_data_file = generate_lion_power_performance_filename(parameters)
    lion_power_plot_data_file = generate_lion_power_plot_filename(parameters)

    lion_power_performance_data = dict()  # Start from scratch

    X_mnist = generate_data.load_x_mnist(parameters=settings.parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)
    Y_mnist = generate_data.load_y_mnist(parameters=settings.parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)

    def get_nearest_neighbors(y, Y_mnist, n, exclude_index):
        y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
        y_distances[exclude_index] = np.inf
        return np.argsort(y_distances)[:n]

    distance_matrix = distance.squareform(distance.pdist(X_mnist))
    np.fill_diagonal(distance_matrix, np.inf)  # We are not interested in distance to itself
    nn_x_distance = np.min(distance_matrix, axis=1)  # Any axis will do
    radius_x = dict()
    for p in lion_percentile_options:
        radius_x[p] = np.percentile(nn_x_distance, p)
    logging.info("Radius X: %s", radius_x)

    if os.path.isfile(lion_power_performance_data_file) and not regenerate:
        with open(lion_power_performance_data_file, 'rb') as f:
            lion_power_performance_data = pickle.load(f)

    for perc in lion_percentile_options:
        for p in lion_power_options:
            logging.info("Processing percentile and power: %f, %d", p, perc)
            key = str(perc) + ";" + "%.3f" % (p)
            logging.info("Key: %s", key)
            if key not in lion_power_performance_data:
                lion_power_performance_data[key] = dict()

            if 'InternalPrecisionByX' not in lion_power_performance_data[key] or regenerate:
                logging.info("Power performance not found for power %f percentile %d.\tCalculating...", p, perc)

                per_sample_precision_x = list()
                per_sample_precision_y = list()
                for i in range(len(X_mnist)):
                    distances = distance_matrix[i, :].copy()
                    distances[i] = np.inf  # Not interested in distance to itself
                    # Step 1. Find nearest neighbors in the neighborhood.
                    neighbor_indices = np.where(distances <= radius_x[perc])[0]
                    num_neighbors = len(neighbor_indices)
                    if num_neighbors >= 2:  # Below 2? Cannot interpolate
                        # We are good
                        weights = 1 / distances[neighbor_indices] ** p
                        weights = weights / np.sum(weights)
                        cur_y_result = weights.dot(Y_mnist[neighbor_indices, :])

                        nn_xreal_indices = get_nearest_neighbors(X_mnist[i, :], X_mnist, n=precision_nn,
                                                                 exclude_index=i)
                        nn_yreal_indices = get_nearest_neighbors(Y_mnist[i, :], Y_mnist, n=precision_nn,
                                                                 exclude_index=i)
                        nn_yembedded_indices = get_nearest_neighbors(cur_y_result, Y_mnist, n=precision_nn,
                                                                     exclude_index=i)
                        matching_indices_xreal_yembedded = len(
                            [j for j in nn_xreal_indices if j in nn_yembedded_indices])
                        matching_indices_yreal_yembedded = len(
                            [j for j in nn_yreal_indices if j in nn_yembedded_indices])
                        per_sample_precision_x.append(matching_indices_xreal_yembedded / precision_nn)
                        per_sample_precision_y.append(matching_indices_yreal_yembedded / precision_nn)

                new_dict = dict()
                new_dict['InternalPrecisionByX'] = np.mean(per_sample_precision_x)
                new_dict['InternalPrecisionByY'] = np.mean(per_sample_precision_y)

                for ndk in new_dict.keys():
                    lion_power_performance_data[key][ndk] = new_dict[ndk]

                with open(lion_power_performance_data_file, 'wb') as f:
                    pickle.dump(lion_power_performance_data, f)
            else:
                logging.info("Power FOUND for power %f percentile %d. Using loaded.", p, perc)

            logging.info("%s %s", key, lion_power_performance_data[key])

    lion_optimal_power_x = dict()
    lion_power_plot_x = dict()
    for perc in lion_percentile_options:
        y = list()
        for cur_power in lion_power_options:
            key = str(perc) + ";%.3f" % (cur_power)
            # print(cur_power, perc, lion_power_plot_data[key])
            y.append(lion_power_performance_data[key]['InternalPrecisionByX'])
        lion_power_plot_x[perc] = y
        lion_optimal_power_x[perc] = lion_power_options[np.argmin(y)]

    lion_optimal_power_y = dict()
    lion_power_plot_y = dict()
    for perc in lion_percentile_options:
        y = list()
        for cur_power in lion_power_options:
            key = str(perc) + ";%.3f" % (cur_power)
            # print(cur_power, perc, lion_power_plot_data[key])
            y.append(lion_power_performance_data[key]['InternalPrecisionByY'])
        lion_power_plot_y[perc] = y
        lion_optimal_power_y[perc] = lion_power_options[np.argmin(y)]

    with open(lion_power_plot_data_file, 'wb') as f:
        pickle.dump((lion_power_options, lion_power_plot_y, lion_optimal_power_y, lion_power_plot_x, lion_optimal_power_x), f)
    logging.info("LION optimal power X: %s", lion_optimal_power_x)
    logging.info("LION optimal power Y: %s", lion_optimal_power_y)


    end_time = datetime.datetime.now()
    logging.info("LION power experiment ended: %s", end_time)
    logging.info("LION power experiment duration: %s", end_time-start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_lion_power_performance(parameters=settings.parameters, regenerate=False, recursive_regenerate=False)

