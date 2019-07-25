# TODO Rework with precision father than accuracy

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

ptsne_train_mat = loadmat('mnist_train.mat')
ptsne_test_mat = loadmat('mnist_test.mat')

accuracy_nn = 10
precision_nn = 50

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


def ptsne_get_nearest_neighbors_in_y(y,n,i=-1):
    y_distances = np.sum((ptsne_Y_train - y)**2, axis=1)
    if i>=0:
        y_distances[i] = np.inf
    return np.argsort(y_distances)[:n]

per_sample_accuracy = np.zeros((len(ptsne_X_test),))
for i in range(len(ptsne_X_test)):
    if  i%1000==0:
        print("Processing",i)
    expected_label = ptsne_labels_test[i]
    nn_indices = ptsne_get_nearest_neighbors_in_y(ptsne_Y_test[i,:], n=accuracy_nn)
    obtained_labels = ptsne_labels_train[nn_indices]
    per_sample_accuracy[i] = sum(obtained_labels==expected_label) / len(obtained_labels)
ptsne_accuracy = np.mean(per_sample_accuracy) 
print("PTSNE accuracy: ", ptsne_accuracy)





lion_extended_percentile_options = [90, 95, 99, 100]

extended_nn_dist_file = 'extended_nn_dist.p'
regenerate_extended_distances = False

if os.path.isfile(extended_nn_dist_file) and not regenerate_extended_distances:
    with open(extended_nn_dist_file, "rb") as f:
        nn_x_extended_distance, nn_y_extended_distance = pickle.load(f)
else:
    lim = len(ptsne_X_train)
    nn_x_extended_distance = np.zeros(len(ptsne_X_train))
    nn_y_extended_distance = np.zeros(len(ptsne_X_train))
    for i in range(lim):
        if i % 1000 == 0:
            print("Regenerating extended nn distance", i)
        with open('ExtendedDistanceVectors/train_' + str(i) + '.p', 'rb') as f:
            dist = pickle.load(f)
        nn_x_extended_distance[i] = np.min(dist)
        # print(nn_x_extended_distance[i])
        dist_y = np.sqrt(np.sum((ptsne_Y_train - ptsne_Y_train[i, :]) ** 2, axis=1))
        dist_y[i] = np.inf
        nn_y_extended_distance[i] = np.min(dist_y)

    with open(extended_nn_dist_file, "wb") as f:
        pickle.dump((nn_x_extended_distance, nn_y_extended_distance), f)