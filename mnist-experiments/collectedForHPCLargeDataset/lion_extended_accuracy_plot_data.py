import pickle
import numpy as np
from scipy.io import loadmat
import os
import sys

i = int(sys.argv[1]) if len(sys.argv)>1 else 0

step=1
lion_extended_power_options = np.arange(1, 200+step, step)
lion_extended_power_options = sorted(list(lion_extended_power_options) + [13.8, 16.5, 21.1, 26.6])
#Those are found by cross-validation below. Just want to estimate accuracy at those spots

lion_extended_percentile_options = [90,95,99,100]

ptsne_train_mat = loadmat('mnist_train.mat')
ptsne_test_mat = loadmat('mnist_test.mat')

accuracy_nn = 10

ptsne_mapped = loadmat('mnist_mapped.mat') # We need Y to take as ground truth
ptsne_Y_train = ptsne_mapped['mapped_train_X']
ptsne_Y_test = ptsne_mapped['mapped_test_X']

ptsne_X_train = ptsne_train_mat['train_X']
ptsne_labels_train = ptsne_train_mat['train_labels'].reshape(-1)-1
ptsne_X_test = ptsne_test_mat['test_X']
ptsne_labels_test = ptsne_test_mat['test_labels'].reshape(-1)-1

lion_extended_accuracy_plot_data_file = './lion-processed-samples/lion_extended_accuracy_plot_data_'+str(i)+'.p'

with open('cell_centers.p', 'rb') as f:
    cell_centers = pickle.load(f)

# This cell calculates test set accuracy for each combination of power and radius_x percentile

np.random.seed(111222)

# Leaving it extensible to get smth like "KL vs power" etc. if we need to.
regenerate_extended_lion_accuracy = False

lion_extended_accuracy_plot_data = dict()
lion_extended_precision_30_plot_data = dict()
lion_extended_precision_50_plot_data = dict()
already_tested_keys = dict()

with open('extended_nn_dist.p', "rb") as f:
    nn_x_extended_distance, nn_y_extended_distance = pickle.load(f)

radius_y = np.percentile(nn_y_extended_distance, 100)
radius_y_close = np.percentile(nn_y_extended_distance, 10)
radius_y += radius_y_close

radius_extended_x = dict()
for p in lion_extended_percentile_options:
    radius_extended_x[p] = np.percentile(nn_x_extended_distance, p)

# One at a time, so no such things as outliers overlap, etc.

sample_accuracy = dict()
sample_precision_30 = dict()
sample_precision_50 = dict()


def ptsne_get_nearest_neighbors_in_y(y, ptsne_Y_train, n,i=-1):
    y_distances = np.sum((ptsne_Y_train - y)**2, axis=1)
    if i>=0:
        y_distances[i] = np.inf
    return np.argsort(y_distances)[:n]


print("Processing:", i)

with open('./ExtendedDistanceVectors/test_' + str(i) + '.p', 'rb') as f:
    x_distances_to_all_training_samples = pickle.load(f)

y_result = dict()  # Embeddings in all cases

# If we got exact match, power and percentile does not matter, response is the same
exact_match = np.where(x_distances_to_all_training_samples == 0)[0]
if len(exact_match) > 0:
    print("Exact match!")
    # If we got exact match, power and percentile does not matter, response is the same
    for perc in lion_extended_percentile_options:
        for p in lion_extended_power_options:
            key = str(perc) + ";" + "%.3f" % (p)
            if key not in already_tested_keys:
                y_result[key] = ptsne_Y_train[exact_match[0], :]
else:
    # Not an exact match. Need to go further.
    for perc in lion_extended_percentile_options:
        # print(perc)
        # This part depends only on radius_x
        cur_neighbor_indices = np.where(x_distances_to_all_training_samples <= radius_extended_x[perc])[0]
        local_interpolation_distances = x_distances_to_all_training_samples[cur_neighbor_indices]
        # print(cur_neighbor_indices)
        # print(local_interpolation_distances)
        if len(local_interpolation_distances) == 0:  # Outlier handling does not depend on power.
            # print(perc, 0)
            for p in lion_extended_power_options:
                key = str(perc) + ";" + "%.3f" % (p)
                # No need to pop. Each sample is independent anyway
                if key not in already_tested_keys:
                    y_result[key] = cell_centers[np.random.randint(0, len(cell_centers)), :]
        elif len(local_interpolation_distances) == 1:  # Single-neighbor handling does not depend on power
            # print(perc, 1)
            # Now a single neighbor
            single_neighbor_index = cur_neighbor_indices[0]
            single_neighbor_nn_dist = nn_x_extended_distance[single_neighbor_index]
            if single_neighbor_nn_dist <= radius_extended_x[perc]:
                # Treat as outliers
                for p in lion_extended_power_options:
                    key = str(perc) + ";" + "%.3f" % (p)
                    # No need to pop. Each sample is independent anyway
                    if key not in already_tested_keys:
                        y_result[key] = cell_centers[np.random.randint(0, len(cell_centers)), :]
            else:
                for p in lion_extended_power_options:
                    key = str(perc) + ";" + "%.3f" % (p)
                    if key not in already_tested_keys:
                        random_dist = np.random.uniform(low=0, high=radius_y_close)
                        random_angle = np.random.uniform(low=0, high=2 * np.pi)
                        y_result[key] = ptsne_Y_train[single_neighbor_index, :].copy()
                        y_result[key][0] += random_dist * np.cos(
                            random_angle)  # Considering 0 is X. DOes not matter, really
                        y_result[key][1] += random_dist * np.sin(random_angle)
        else:
            # print(perc, 2)
            for p in lion_extended_power_options:
                key = str(perc) + ";" + "%.3f" % (p)
                # weights = 1 / np.array([decimal.Decimal(i) for i in local_interpolation_distances]) ** p
                if key not in already_tested_keys:
                    weights = 1 / local_interpolation_distances ** p
                    weights = weights / np.sum(weights)
                    y_result[key] = weights.dot(ptsne_Y_train[cur_neighbor_indices, :])

print("Got results")

# TODO Got results for all cases
for perc in lion_extended_percentile_options:
    for p in lion_extended_power_options:
        key = str(perc) + ";" + "%.3f" % (p)
        print(perc, p, key)

        expected_label = ptsne_labels_test[i]
        result = y_result[key]
        nn_indices = ptsne_get_nearest_neighbors_in_y(result, ptsne_Y_train, n=accuracy_nn)
        obtained_labels = ptsne_labels_train[nn_indices]
        sample_accuracy[key] = sum(obtained_labels == expected_label) / len(obtained_labels)

        nn_x_indices = ptsne_get_nearest_neighbors_in_y(ptsne_X_test[i,:], ptsne_X_train, n=30)
        nn_y_indices = ptsne_get_nearest_neighbors_in_y(result, ptsne_Y_train, n=30)
        matching_indices = len([k for k in nn_x_indices if k in nn_y_indices])
        sample_precision_30[key] = (matching_indices / 30)

        nn_x_indices = ptsne_get_nearest_neighbors_in_y(ptsne_X_test[i,:], ptsne_X_train, n=50)
        nn_y_indices = ptsne_get_nearest_neighbors_in_y(result, ptsne_Y_train, n=50)
        matching_indices = len([k for k in nn_x_indices if k in nn_y_indices])
        sample_precision_50[key] = (matching_indices / 50)

        print("\t", sample_accuracy[key], sample_precision_30[key], sample_precision_50[key])


with open(lion_extended_accuracy_plot_data_file, 'wb') as f:
    pickle.dump((y_result, sample_accuracy, sample_precision_30,
                 sample_precision_50), f)