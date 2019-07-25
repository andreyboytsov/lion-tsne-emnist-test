import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

ptsne_train_mat = loadmat('mnist_train.mat')
ptsne_test_mat = loadmat('mnist_test.mat')

accuracy_nn = 10

# TODO Need that file
ptsne_mapped = loadmat('mnist_mapped.mat') # We need Y to take as ground truth
ptsne_Y_train = ptsne_mapped['mapped_train_X']
ptsne_Y_test = ptsne_mapped['mapped_test_X']
ptsne_Y_outliers = ptsne_mapped['mapped_outliers_X']

ptsne_X_train = ptsne_train_mat['train_X']
ptsne_labels_train = ptsne_train_mat['train_labels'].reshape(-1)-1
ptsne_X_test = ptsne_test_mat['test_X']
ptsne_labels_test = ptsne_test_mat['test_labels'].reshape(-1)-1

import itertools

extended_nn_dist_file = 'extended_nn_dist.p'
with open(extended_nn_dist_file, "rb") as f:
    nn_x_extended_distance, nn_y_extended_distance = pickle.load(f)

radius_y = np.percentile(nn_y_extended_distance, 100)
radius_y_close = np.percentile(nn_y_extended_distance, 10)
radius_y += radius_y_close

# Pre-generate cells for outlier placement
# Each point is an idependent test, so only one layer should be fine
available_cells = list()  # Contain number of cells counting on each axis

y_min = np.min(ptsne_Y_train, axis=0).reshape(-1)  # Let's precompute
y_max = np.max(ptsne_Y_train, axis=0).reshape(-1)  # Let's precompute

print('Minimums: ', y_min)
print('Maximums: ', y_max)

# Number of cells per dimension.
original_cell_nums = [int(np.floor((y_max[i] - y_min[i]) / (2 * radius_y))) for i in range(ptsne_Y_train.shape[1])]
# Within y_min to y_max cell can be slighlty larger to divide exactly
adjusted_cell_sizes = [(y_max[i] - y_min[i]) / original_cell_nums[i] for i in range(ptsne_Y_train.shape[1])]
# How many outer layers did we have to add. For now - none.
# added_outer_layers = 0 #We do it locally, cause runs are independent
cell_list = list(itertools.product(*[list(range(i)) for i in original_cell_nums]))
for cell in cell_list:
    # Bounds for each dimension
    cell_bounds_min = [y_min[i] + cell[i] * adjusted_cell_sizes[i] for i in range(ptsne_Y_train.shape[1])]
    cell_bounds_max = [y_min[i] + (cell[i] + 1) * adjusted_cell_sizes[i] for i in range(ptsne_Y_train.shape[1])]
    samples_in_cell = np.array([True] * ptsne_Y_train.shape[0])  #
    for i in range(len(cell_bounds_min)):
        samples_in_cell = samples_in_cell & \
                          (cell_bounds_min[i] <= ptsne_Y_train[:, i]) & (cell_bounds_max[i] >= ptsne_Y_train[:, i])
    if not samples_in_cell.any():
        available_cells.append(cell)

cell_centers = np.zeros((len(available_cells), ptsne_Y_train.shape[1]))
for i in range(len(available_cells)):
    # Cells can only be from first layer
    cell = available_cells[i]
    for j in range(ptsne_Y_train.shape[1]):
        cell_centers[i, j] = y_min[j] + cell[j] * adjusted_cell_sizes[j] + adjusted_cell_sizes[j] / 2


with open('cell_centers.p', 'wb') as f:
    pickle.dump(cell_centers, f)