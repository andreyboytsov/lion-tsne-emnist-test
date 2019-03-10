"""
It contains some methods common for many experiments for LION-tSNE on MNIST dataset
"""
import numpy as np
import lion_tsne
import pickle
from scipy.spatial import distance

"""
MNIST temp file location
"""
mnist_file = '../experiment-temp-files/mnist.p'

"""
Minimum distance between samples. Can be used to match and detect randomization problems.
"""
expected_min_dist = 0.451927052496


def load_mnist_data(prepare_cluster_attribution_test=True):
    """
    Loads a subset of MNIST dataset and a result of its tSNE processing.

    :param prepare_cluster_attribution_test: Will prepare additional data for cluster attribution test

    :return: a tuple with several values
      [0] - X_mnist, 2500x30, first index - MNIST image, second index - first 30 PCA components
      [1] - Y_mnist, 2500x2, first index - MNIST image (same sa above), second index - tSNE results
      [2] - X_mnist_raw, 2500x784, first index - MNIST image, second index - unrolled 28x28 grayscale image (values
            between 0 and 1)
      [3] - labels_mnist, 2500-long array of labels for images
      [4] - dTSNE_mnist, LION-tSNE incorporation of the data. Actually, you can generate other embeddings from that
            object as well.
      [5] - mnist_chosen_indices, 2500-long array, which elements of MNIST dataset correspond to 2500-long subset
    If prepare_cluster_attribution_test:
      [6] - X_mnist_unchosen_raw
      [7] - X_mnist_unchosen_pca
      [8] - labels_mnist_unchosen
    """

    print("Loading from file...")

    with open(mnist_file, 'rb') as f:
        X_mnist_raw, P_mnist, sigma_mnist, Y_mnist, labels_mnist_onehot, mnist_pca, \
        all_mnist_trained_images, all_mnist_labels, mnist_chosen_indices = pickle.load(f)

    labels_mnist = np.argmax(labels_mnist_onehot, axis=1)

    temp = np.ascontiguousarray(X_mnist_raw).view(
        np.dtype((np.void, X_mnist_raw.dtype.itemsize * X_mnist_raw.shape[1])))
    _, un_idx = np.unique(temp, return_index=True)
    X_mnist_raw = X_mnist_raw[un_idx, :]
    labels_mnist = labels_mnist[un_idx]
    mnist_chosen_indices = mnist_chosen_indices[un_idx]

    X_mnist = mnist_pca.transform(X_mnist_raw)
    dTSNE_mnist = lion_tsne.LionTSNE(perplexity=30)
    dTSNE_mnist.incorporate(x=X_mnist, y=Y_mnist, p_matrix=P_mnist, sigma=sigma_mnist)

    D = distance.pdist(X_mnist)
    min_dist = np.min(D)
    print("After PCA - minimum distance between samples is ", min_dist,
          "\nExpected: ", expected_min_dist,
          "\nDifference: ", min_dist-expected_min_dist)

    return_tuple = X_mnist, Y_mnist, X_mnist_raw, labels_mnist, dTSNE_mnist, mnist_chosen_indices

    if prepare_cluster_attribution_test:
        print("Generating data for cluster attribution test")
        ind_unchosen = [i for i in range(len(all_mnist_labels)) if i not in mnist_chosen_indices]
        np.random.seed(10)  # Any seed, just don't use it again for selecting indices from same dataset

        X_mnist_unchosen_raw = all_mnist_trained_images[ind_unchosen]
        labels_mnist_unchosen = all_mnist_trained_images[ind_unchosen]
        labels_mnist_unchosen = np.argmax(labels_mnist_unchosen, axis=1)
        X_mnist_unchosen_pca = mnist_pca.transform(X_mnist_unchosen_raw)
        return_tuple += X_mnist_unchosen_raw, X_mnist_unchosen_pca, labels_mnist_unchosen

    return return_tuple
########################################################################################################################



"""
def something():
    cluster_results_file = 'clustering_results.p'
    regenerate_cluster_results = False

    picked_indices_y_mnist = Y_mnist[picked_indices, :]
    picked_closest_training_indices_y_mnist = Y_mnist[nearest_training_indices, :]

    if os.path.isfile(cluster_results_file) and not regenerate_cluster_results:
        print("Results file found. Loading...")
        with open(cluster_results_file, 'rb') as f:
            picked_neighbors_y_multiquadric, picked_neighbors_y_cubic, picked_neighbors_y_linear, \
            picked_neighbors_y_quintic, picked_neighbors_y_gaussian, picked_neighbors_y_inverse, \
            picked_neighbors_y_thin_plate, picked_neighbors_y_idw1, picked_neighbors_y_idw10, \
            picked_neighbors_y_idw20, picked_neighbors_y_idw_optimal, picked_neighbors_y_idw40 = pickle.load(f)
    else:
"""

"""
        picked_neighbors_y_idw1 = emb_mnist_idw1(picked_neighbors)
        print("Got IDW1")
        picked_neighbors_y_idw20 = emb_mnist_idw20(picked_neighbors)
        print("Got IDW20")
        picked_neighbors_y_idw_optimal = emb_mnist_idw_optimal(picked_neighbors)
        print("Got IDW optimal")
        picked_neighbors_y_idw40 = emb_mnist_idw40(picked_neighbors)
        print("Got IDW 40")
        picked_neighbors_y_idw10 = emb_mnist_idw10(picked_neighbors)
        print("Got IDW 10")
        with open(cluster_results_file, 'wb') as f:
            pickle.dump((picked_neighbors_y_multiquadric, picked_neighbors_y_cubic, picked_neighbors_y_linear, \
                         picked_neighbors_y_quintic, picked_neighbors_y_gaussian, picked_neighbors_y_inverse, \
                         picked_neighbors_y_thin_plate, picked_neighbors_y_idw1, picked_neighbors_y_idw10, \
                         picked_neighbors_y_idw20, picked_neighbors_y_idw_optimal, picked_neighbors_y_idw40), f)
        print("Saved")
"""
