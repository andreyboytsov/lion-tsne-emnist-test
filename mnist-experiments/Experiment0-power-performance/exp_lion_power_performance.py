import settings
from scipy.spatial import distance
import logging
import datetime
import numpy as np
import generate_data
import pickle
import os

lion_power_options = np.arange(0.1, 120.1, step=0.1)
lion_percentile_options = [90, 95, 99, 100]
lion_power_performance_prefix = '../results/lion_power'
lion_power_plot_prefix = '../results/lion_power_plot'

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
    logging.info("LION power experiment started: %s", start_time)

    accuracy_nn = parameters.get("accuracy_nn", settings.parameters["accuracy_nn"])

    lion_power_performance_data_file = generate_lion_power_performance_filename(parameters)
    lion_power_plot_data_file = generate_lion_power_plot_filename(parameters)

    lion_power_performance_data = dict()  # Start from scratch

    X_mnist = generate_data.load_x_mnist(parameters=settings.parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)
    Y_mnist = generate_data.load_y_mnist(parameters=settings.parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)
    labels_mnist = generate_data.load_labels_mnist(parameters=settings.parameters, regenerate=recursive_regenerate,
                                         recursive_regenerate=recursive_regenerate)

    def get_nearest_neighbors_in_y(y, n=10):
        y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
        return np.argsort(y_distances)[:n]

    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=settings.parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=settings.parameters)
    picked_neighbor_labels = generate_data.load_picked_neighbors_labels(parameters=settings.parameters)

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

            if 'Accuracy' not in lion_power_performance_data[key]:
                logging.info("Accuracy not found for power %f percentile %d. \tCalculating...", p, perc)
                interpolator = dTSNE_mnist.generate_lion_tsne_embedder(verbose=0, random_state=0, function_kwargs={
                    'radius_x_percentile': perc,
                    'power': p})

                per_sample_accuracy = np.zeros((len(picked_neighbors),))
                for i in range(len(picked_neighbors)):
                    # if i%100==0:
                    #    print("\tPower: ",p,"Processing:",i)
                    expected_label = picked_neighbor_labels[i]
                    result = interpolator(picked_neighbors[i], verbose=0)
                    nn_indices = get_nearest_neighbors_in_y(result, n=accuracy_nn)
                    obtained_labels = labels_mnist[nn_indices]
                    per_sample_accuracy[i] = sum(obtained_labels == expected_label) / len(obtained_labels)
                cur_acc = np.mean(per_sample_accuracy)
                # print('================= ',p,perc, cur_acc)
                lion_power_performance_data[key]['Accuracy'] = cur_acc
                with open(lion_power_performance_data_file, 'wb') as f:
                    pickle.dump(lion_power_performance_data, f)
            else:
                logging.info("Accuracy FOUND for power %f percentile %d. Using loaded.", p, perc)

            if 'PowerSquareDist' not in lion_power_performance_data[key] or regenerate:
                logging.info("Power performance not found for power %f percentile %d.\tCalculating...", p, perc)

                y_sum_square_dist = 0.0
                y_sum_abs_dist = 0.0
                y_count = 0.0
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
                        y_sum_square_dist += np.sum(cur_y_result - Y_mnist[i, :]) ** 2
                        y_sum_abs_dist += np.sqrt(np.sum(cur_y_result - Y_mnist[i, :]) ** 2)
                        y_count += 1.0
                new_dict = dict()
                new_dict['PowerSquareDist'] = y_sum_square_dist / y_count
                new_dict['PowerAbsDist'] = y_sum_abs_dist / y_count
                # Just in case it will become unstable due to too few neighbors
                new_dict['PowerSquareDistSum'] = y_sum_square_dist
                new_dict['PowerSquareDistCount'] = y_count
                for ndk in new_dict.keys():
                    lion_power_performance_data[key][ndk] = new_dict[ndk]

                with open(lion_power_performance_data_file, 'wb') as f:
                    pickle.dump(lion_power_performance_data, f)
            else:
                logging.info("Power FOUND for power %f percentile %d. Using loaded.", p, perc)

            logging.info("%s %s", key, lion_power_performance_data[key])

    lion_optimal_power = dict()
    lion_power_plot_y = dict()
    for perc in lion_percentile_options:
        y = list()
        for cur_power in lion_power_options:
            key = str(perc) + ";%.3f" % (cur_power)
            # print(cur_power, perc, lion_power_plot_data[key])
            y.append(lion_power_performance_data[key]['PowerSquareDist'])
        lion_power_plot_y[perc] = y
        lion_optimal_power[perc] = lion_power_options[np.argmin(y)]

    with open(lion_power_plot_data_file, 'wb') as f:
        pickle.dump((lion_power_options, lion_power_plot_y, lion_optimal_power), f)
    logging.info("LION optimal power: %s", lion_optimal_power)


    end_time = datetime.datetime.now()
    logging.info("LION power experiment ended: %s", end_time)
    logging.info("LION power experiment duration: %s", end_time-start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_lion_power_performance(parameters=settings.parameters, regenerate=False, recursive_regenerate=False)

