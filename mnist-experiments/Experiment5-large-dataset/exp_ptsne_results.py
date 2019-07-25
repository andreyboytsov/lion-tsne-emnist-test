# TODO Rework with precision father than accuracy

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


# That plot is just for my own reference 
plt.gcf().set_size_inches(8,8)
#chosen_ptsne = 2
legend_list = list()
for l in sorted(set(ptsne_labels_train)):
    plt.scatter(ptsne_Y_train[ptsne_labels_train == l,0], ptsne_Y_train[ptsne_labels_train == l,1], marker = '.', alpha=1.0)
    legend_list.append(str(l))
#plt.title("MNIST Dataset - TSNE visualization")
#plt.tight_layout()
plt.legend(legend_list)
plt.show()
#plt.scatter(res1['mapped_train_X'][:,0], res1['mapped_train_X'][:,1], c='gray',marker='.')


def ptsne_get_nearest_neighbors_in_y(y, ptsne_Y_train, n,i=-1):
    y_distances = np.sum((ptsne_Y_train - y)**2, axis=1)
    if i>=0:
        y_distances[i] = np.inf
    return np.argsort(y_distances)[:n]


per_sample_accuracy = np.zeros((len(ptsne_X_test),))
per_sample_precision_30 = np.zeros((len(ptsne_X_test),))
per_sample_precision_50 = np.zeros((len(ptsne_X_test),))

for i in range(len(ptsne_X_test)):
    if  i%100 == 0:
        print("Processing",i, " of ", len(ptsne_X_test))
    expected_label = ptsne_labels_test[i]
    nn_indices = ptsne_get_nearest_neighbors_in_y(ptsne_Y_test[i,:], ptsne_Y_train, n=accuracy_nn)
    obtained_labels = ptsne_labels_train[nn_indices]
    per_sample_accuracy[i] = sum(obtained_labels==expected_label) / len(obtained_labels)

    y = ptsne_Y_test[i, :]
    x = ptsne_X_test[i, :]
    nn_x_indices = ptsne_get_nearest_neighbors_in_y(x, ptsne_X_train, n=30)
    nn_y_indices = ptsne_get_nearest_neighbors_in_y(y, ptsne_Y_train, n=30)
    matching_indices = len([k for k in nn_x_indices if k in nn_y_indices])
    per_sample_precision_30[i] = (matching_indices / 30)

    nn_x_indices = ptsne_get_nearest_neighbors_in_y(x, ptsne_X_train, n=50)
    nn_y_indices = ptsne_get_nearest_neighbors_in_y(y, ptsne_Y_train, n=50)
    matching_indices = len([k for k in nn_x_indices if k in nn_y_indices])
    per_sample_precision_50[i] = (matching_indices / 50)

ptsne_accuracy = np.mean(per_sample_accuracy)
ptsne_precision_30 = np.mean(per_sample_precision_30)
ptsne_precision_50 = np.mean(per_sample_precision_50)

with open('ptsne_results.p', 'wb') as f:
    pickle.dump( (ptsne_accuracy, ptsne_precision_30, ptsne_precision_50,
                  per_sample_accuracy, per_sample_precision_30, per_sample_precision_50) , f)


print("PTSNE accuracy: ", ptsne_accuracy)
print("PTSNE precision 30: ", ptsne_precision_30)
print("PTSNE precision 50: ", ptsne_precision_50)