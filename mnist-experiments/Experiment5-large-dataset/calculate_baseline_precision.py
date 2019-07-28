import pickle
import numpy as np
from scipy.io import loadmat

ptsne_train_mat = loadmat('mnist_train.mat')

accuracy_nn = 10
#precision_nn = 50

ptsne_mapped = loadmat('mnist_mapped.mat') # We need Y to take as ground truth
ptsne_Y_train = ptsne_mapped['mapped_train_X']
ptsne_X_train = ptsne_train_mat['train_X']

precision_nn_filename = 'baseline_precision_by_nn.p'
precision_by_nn = dict()

def get_nearest_neighbors(y, Y_mnist, n=10):
    y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
    return np.argsort(y_distances)[:n]

for precision_nn in [30, 50]:
    per_sample_precision = list()
    for i in range(len(ptsne_X_train)):
        if i % 100 == 0:
            print("Processing for baseline precision: %d", i)
        x = ptsne_X_train[i, :]
        y = ptsne_Y_train[i, :]
        nn_x_indices = get_nearest_neighbors(x, ptsne_X_train, n=precision_nn+1) # +1 to account for "itself"
        nn_y_indices = get_nearest_neighbors(y, ptsne_Y_train, n=precision_nn+1) # +1 to account for "itself"
        matching_indices = len([j for j in nn_x_indices if j in nn_y_indices and j != i])
        per_sample_precision.append(matching_indices / precision_nn)
    precision_by_nn[precision_nn] = np.mean(per_sample_precision)
    print('GOT IT: ', precision_nn, precision_by_nn[precision_nn])
    with open(precision_nn_filename, 'wb') as f:
        pickle.dump(precision_by_nn, f)