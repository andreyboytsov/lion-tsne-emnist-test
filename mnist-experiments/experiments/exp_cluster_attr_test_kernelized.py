# TODO rough estimation. Make better code later.

from scipy.spatial import distance
import pickle
import numpy as np
import os
import generate_data
import settings
import matplotlib.pyplot as plt
from scipy import stats
import kernelized_tsne
import logging


def generate_cluster_results_filename(parameters=settings.parameters):
    cluster_results_file_prefix = '../results/cluster_attr_kernelized_'
    return cluster_results_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)


def main(parameters=settings.parameters,regenerate_parameters_cache=False):
    step = 0.01
    choice_K = np.arange(step, 3 + step, step)  # Let's try those K.

    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=parameters)
    picked_neighbor_labels = generate_data.load_picked_neighbors_labels(parameters=parameters)
    accuracy_nn = parameters.get("accuracy_nn", settings.parameters["accuracy_nn"])
    labels_mnist = generate_data.load_labels_mnist(parameters=parameters)
    baseline_accuracy = generate_data.get_baseline_accuracy(parameters=parameters)

    D_Y = distance.squareform(distance.pdist(Y_mnist))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    nearest_neighbors_y_dist = np.min(D_Y, axis=1)  # Actually, whatever axis

    def get_nearest_neighbors_in_y(y, n=10):
        y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
        return np.argsort(y_distances)[:n]

    # Implementing carefully. Not the fastest, but the most reliable way.

    kernel_tsne_mapping = kernelized_tsne.generate_kernelized_tsne_mapping_function(
        parameters=parameters,
        regenerate_parameters_cache=regenerate_parameters_cache
    )

    kernelized_detailed_tsne_method_list = ["Kernelized tSNE; K=%.2f" % (k) for k in choice_K]
    kernelized_detailed_tsne_method_results = [kernel_tsne_mapping(picked_neighbors, k=k) for k in choice_K]

    kernelized_detailed_tsne_accuracy = np.zeros((len(kernelized_detailed_tsne_method_list),))

    for j in range(len(kernelized_detailed_tsne_method_results)):
        per_sample_accuracy = np.zeros((len(picked_neighbors),))
        for i in range(len(picked_neighbors)):
            expected_label = picked_neighbor_labels[i]
            nn_indices = get_nearest_neighbors_in_y(kernelized_detailed_tsne_method_results[j][i, :], n=accuracy_nn)
            obtained_labels = labels_mnist[nn_indices]
            per_sample_accuracy[i] = sum(obtained_labels == expected_label) / len(obtained_labels)
        kernelized_detailed_tsne_accuracy[j] = np.mean(per_sample_accuracy)
        logging.info(kernelized_detailed_tsne_method_list[j], kernelized_detailed_tsne_accuracy[j])

    # Accuracy-vs-power plot
    legend_list = list()
    f, ax = plt.subplots()
    f.set_size_inches(6, 3)
    x = [k for k in choice_K]  # Ensuring order
    y = kernelized_detailed_tsne_accuracy
    # plt.title("IDW - Accuracy vs Power") # We'd better use figure caption
    # ax.legend([h1,h2,h3,h4,h5,h6], ["Closest Training Set Image"]+idw_method_list)
    plt.plot(x, y, c='blue')
    h = plt.axhline(y=baseline_accuracy, c='black', linestyle='--')
    plt.legend([h], ["Baseline Accuracy (%.4f)" % baseline_accuracy])
    plt.xlabel("Kernelized tSNE: K parameter")
    plt.ylabel("10-NN Accuracy")
    plt.ylim([0, 1])
    plt.xlim([0, 2])
    f.tight_layout()
    plt.savefig("../figures/kernelized-tsne-K-vs-accuracy.png")

    ind = [4, 24, 49]
    kernelized_tsne_method_list = [
        kernelized_detailed_tsne_method_list[i][:10] + kernelized_detailed_tsne_method_list[i][-8:]
        for i in ind]
    kernelized_tsne_method_results = [kernelized_detailed_tsne_method_results[i] for i in ind]

    kernelized_tsne_nearest_neighbors_percentiles_matrix = np.zeros((len(picked_neighbors), len(kernelized_tsne_method_list)))
    for i in range(len(picked_neighbors)):
        for j in range(len(kernelized_tsne_method_list)):
            y = kernelized_tsne_method_results[j][i,:]
            nn_dist = np.min(np.sqrt(np.sum((Y_mnist-y)**2, axis=1)))
            kernelized_tsne_nearest_neighbors_percentiles_matrix[i,j] = stats.percentileofscore(nearest_neighbors_y_dist, nn_dist)
    kernelized_tsne_distance_percentiles = np.mean(kernelized_tsne_nearest_neighbors_percentiles_matrix, axis=0)
    for j in range(len(kernelized_tsne_method_list)):
        print(kernelized_tsne_method_list[j], kernelized_tsne_distance_percentiles[j])

    output_file = generate_cluster_results_filename(parameters)
    with open(output_file, 'wb') as f:
        pickle.dump((kernelized_detailed_tsne_method_results, kernelized_detailed_tsne_accuracy,
                     kernelized_detailed_tsne_method_list), f)


if __name__ == "__main__":
    main(parameters=settings.parameters, regenerate_parameters_cache=False)