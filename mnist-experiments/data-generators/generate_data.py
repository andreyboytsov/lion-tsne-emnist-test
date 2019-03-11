"""
Generates different intermediate information one-by-one.

WARNING: It is not designed for concurrency.
"""
import os
import pickle
import numpy as np
import logging
import settings

data_dir_prefix = '..'+os.sep+'data'+os.sep
keras_mnist_file = data_dir_prefix+'keras_mnist.p'
x_mnist_raw_prefix = data_dir_prefix+'X_mnist_raw'
labels_mnist_prefix = data_dir_prefix+'labels_mnist'
mnist_chosen_indices_prefix = data_dir_prefix+'mnist_chosen_indices'
mnist_pca_prefix = data_dir_prefix+'mnist_pca'
x_mnist_prefix = data_dir_prefix+'x_mnist'
y_mnist_prefix = data_dir_prefix+'y_mnist'
dtsne_mnist_prefix = data_dir_prefix+'dstne_mnist'
picked_neighbors_prefix = data_dir_prefix+'picked_neighbors'
picked_neighbors_labels_prefix = data_dir_prefix+'picked_neighbor_labels'
picked_neighbors_raw_prefix = data_dir_prefix+'picked_neighbors_raw'
nearest_training_indices_prefix = data_dir_prefix+'nearest_training_indices'
chosen_labels_prefix = data_dir_prefix+'chosen_labels'
suffix = '.p'

def combine_prefixes(prefixes, parameters):
    res = ""
    for i in sorted(prefixes):
        prefix_value = parameters.get(i, settings.parameters[i])
        res += "_"+str(prefix_value)
    res += suffix
    return res


def save_and_report(generate_filename, parameters, data):
    fname = generate_filename(parameters)
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
        logging.info("%s saved. Size: %d", fname, os.stat(fname).st_size)


def load_or_remake(get_filename_function, generator_function, parameters, regenerate, recursive_regenerate):
    fname = get_filename_function(parameters=parameters)
    logging.info("%s - regeneration %s requested, recursive regeneration %s requested", fname,
                 "IS" if regenerate else "IS NOT", "IS" if recursive_regenerate else "IS NOT")
    if not os.path.isfile(fname) or regenerate:
        logging.info("Regeneration of %s", fname)
        generator_function(parameters=parameters, recursive_regenerate=recursive_regenerate)
    with open(fname, 'rb') as f:
        logging.info("Loading %s", fname)
        return pickle.load(f)


def get_x_mnist_raw_filename(parameters=settings.parameters):
    return x_mnist_raw_prefix + combine_prefixes(settings.raw_parameter_set, parameters)


def get_labels_mnist_filename(parameters=settings.parameters):
    return labels_mnist_prefix + combine_prefixes(settings.raw_parameter_set, parameters)


def get_mnist_chosen_indices_filename(parameters=settings.parameters):
    return mnist_chosen_indices_prefix + combine_prefixes(settings.raw_parameter_set, parameters)


def get_mnist_pca_filename(parameters=settings.parameters):
    return mnist_pca_prefix + combine_prefixes(settings.pca_parameter_set, parameters)


def get_x_mnist_filename(parameters=settings.parameters):
    return x_mnist_prefix + combine_prefixes(settings.pca_parameter_set, parameters)


def get_y_mnist_filename(parameters=settings.parameters):
    return y_mnist_prefix + combine_prefixes(settings.tsne_parameter_set, parameters)


def get_dtsne_mnist_filename(parameters=settings.parameters):
    return dtsne_mnist_prefix + combine_prefixes(settings.tsne_parameter_set, parameters)


def get_picked_neighbors_filename(parameters=settings.parameters):
    return picked_neighbors_prefix + combine_prefixes(settings.x_neighbors_selection_parameter_set, parameters)


def get_picked_neighbors_labels_filename(parameters=settings.parameters):
    return picked_neighbors_labels_prefix + combine_prefixes(settings.x_neighbors_selection_parameter_set, parameters)


def get_picked_neighbors_raw_filename(parameters=settings.parameters):
    return picked_neighbors_raw_prefix + combine_prefixes(settings.x_neighbors_selection_parameter_set, parameters)


def get_nearest_training_indices_filename(parameters=settings.parameters):
    return nearest_training_indices_prefix + combine_prefixes(settings.x_neighbors_selection_parameter_set, parameters)


def get_chosen_labels_filename(parameters=settings.parameters):
    return chosen_labels_prefix + combine_prefixes(settings.x_neighbors_selection_parameter_set, parameters)


def generate_keras_mnist(*, parameters=settings.parameters, recursive_regenerate=False):
    """
    Imports MNIST data from Tensorflow-Keras example and re-saves it to file.
    Use to lock the dataset in place.

    PREDECESSOR: None

    Saves a tuple:
    [0] all_mnist_trained_images - (Nx28x28), grayscale, each pixel between 0 and 1
    [1] all_mnist_labels - Vector of labels
    """
    logging.info("STARTED Caching KERAS-MNIST")
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    (all_mnist_trained_images, all_mnist_labels),(_, _) = mnist.load_data()
    all_mnist_trained_images = all_mnist_trained_images / 255.0
    all_mnist_trained_images = all_mnist_trained_images.reshape((all_mnist_trained_images.shape[0], -1))
    save_and_report(lambda parameters: keras_mnist_file, {}, (all_mnist_trained_images, all_mnist_labels))
    logging.info("COMPLETED Caching KERAS-MNIST")
    return all_mnist_trained_images, all_mnist_labels


def load_keras_mnist(*, parameters=settings.parameters, regenerate=False):
    """
    Loads previously saved data from Tensorflow-Keras. If necessary, regenerates.

    PREDECESSOR: None

    :parameter: Recalculate information and re-save the data file.

    :return: a tuple
    [0] all_mnist_trained_images - (Nx28x28), grayscale, each pixel between 0 and 1
    [1] all_mnist_labels - Vector of labels
    """
    return load_or_remake(lambda parameters: keras_mnist_file, generate_keras_mnist, parameters, regenerate, False)


def generate_raw_and_labels(*, parameters=settings.parameters, recursive_regenerate=False):
    """
    Generates 2500x784 raw MNIST data, coresponding 2500 labels, and a 2500-long list of indices for correspondence
    between entire MNIST and a sample.

    PREDECESSOR: load_keras_mnist

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
    Remaining parameter values will be ignored. See settings.parameters for defaults.
    :param recursive_regenerate: Regenerate predecessors
    """
    logging.info("STARTED Generating raw MNIST dataset")
    all_mnist_trained_images, all_mnist_labels = load_keras_mnist(parameters=parameters,
                                                                  regenerate=recursive_regenerate)
    selection_random_seed = parameters.get("selection_random_seed", settings.parameters["selection_random_seed"])
    num_images_raw = parameters.get("num_images_raw", settings.parameters["num_images_raw"])

    np.random.seed(selection_random_seed)
    ind = np.random.choice(np.arange(len(all_mnist_trained_images)), size=num_images_raw)
    mnist_chosen_indices = ind
    X_mnist_raw = all_mnist_trained_images[ind]
    labels_mnist = all_mnist_labels[ind]

    temp = np.ascontiguousarray(X_mnist_raw).view(
        np.dtype((np.void, X_mnist_raw.dtype.itemsize * X_mnist_raw.shape[1])))
    _, un_idx = np.unique(temp, return_index=True)
    X_mnist_raw = X_mnist_raw[un_idx, :]
    labels_mnist = labels_mnist[un_idx]
    mnist_chosen_indices = mnist_chosen_indices[un_idx]
    save_and_report(get_x_mnist_raw_filename, parameters, X_mnist_raw)
    save_and_report(get_labels_mnist_filename, parameters, labels_mnist)
    save_and_report(get_mnist_chosen_indices_filename, parameters, mnist_chosen_indices)
    logging.info("COMPLETED Generating raw MNIST dataset")


def load_x_mnist_raw(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    """
    Loads 2500x784 raw MNIST data. Generates those data, if the file does not exist.

    GENERATED WITH: mnist_labels, mnist_chosen_indices
    Regenerating this data will also trigger the regeneration of the data from GENERATED_WITH list.
    PREDECESSOR: load_keras_mnist

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
    Remaining parameter values will be ignored. See default_num_images_raw and default_selection_random_seed for
    defaults.
    :param regenerate: Forces to regenerate, even if the file exists
    :param recursive_regenerate: Regenerate predecessors as well (takes effect only if regeneration is requested or
           required).
    :return: 2500x784 raw MNIST data
    """
    return load_or_remake(get_x_mnist_raw_filename, generate_raw_and_labels, parameters, regenerate,
                          recursive_regenerate)


def load_labels_mnist(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    """
    Loads 2500-long list of labels. Generates those data, if the file does not exist.

    GENERATED WITH: mnist_labels, mnist_chosen_indices
    Regenerating this data will also trigger the regeneration of the data from GENERATED_WITH list.
    PREDECESSOR: load_keras_mnist

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
    Remaining parameter values will be ignored. See default_num_images_raw and default_selection_random_seed for
    defaults.
    :param regenerate: Forces to regenerate, even if the file exists
    :param recursive_regenerate: Regenerate predecessors as well (takes effect only if regeneration is requested or
           required).
    :return: 2500-long list of labels
    """
    return load_or_remake(get_labels_mnist_filename, generate_raw_and_labels, parameters, regenerate,
                          recursive_regenerate)


def load_mnist_chosen_indices(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    """
    Loads 2500-long list of indices. Each value is the index of i-th sampled value in the entire MNISt dataset.
    Generates those data, if the file does not exist.

    GENERATED WITH: mnist_labels, mnist_chosen_indices
    Regenerating this data will also trigger the regeneration of the data from GENERATED_WITH list.
    PREDECESSOR: load_keras_mnist

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
    Remaining parameter values will be ignored. See default_num_images_raw and default_selection_random_seed for
    defaults.
    :param regenerate: Forces to regenerate, even if the file exists
    :param recursive_regenerate: Regenerate predecessors as well (takes effect only if regeneration is requested or
           required).
    :return: 2500-long list of labels
    """
    return load_or_remake(get_mnist_chosen_indices_filename, generate_raw_and_labels, parameters, regenerate,
                   recursive_regenerate)


def generate_pca_mnist(*, parameters=settings.parameters, recursive_regenerate=False):
    """
    Generates PCA processor form MNIST data and its effect on X_mnist_raw.

    PREDECESSOR: X_mnist_raw

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
        "num_pca_dimensions": number of kept dimesions after PCA decomposition.
        "random_pca_seed": random seed for PCA calculation.
    Remaining parameter values will be ignored. See default_num_images_raw and default_selection_random_seed for
    defaults.
    :param regenerate: Forces to regenerate, even if the file exists
    :param recursive_regenerate: Regenerate predecessors as well (takes effect only if regeneration is requested or
           required).
    """
    from sklearn.decomposition import PCA
    from scipy.spatial import distance

    num_pca_dimensions = parameters.get("num_pca_dimensions", settings.parameters["num_pca_dimensions"])
    pca_random_seed = parameters.get("pca_random_seed", settings.parameters["pca_random_seed"])
    X_mnist_raw = load_x_mnist_raw(parameters=parameters, regenerate=recursive_regenerate,
                                   recursive_regenerate=recursive_regenerate)
    mnist_pca = PCA(n_components=num_pca_dimensions, random_state=pca_random_seed)
    X_mnist = mnist_pca.fit_transform(X_mnist_raw)

    save_and_report(get_mnist_pca_filename, parameters, mnist_pca)
    save_and_report(get_x_mnist_filename, parameters, X_mnist)

    D = distance.pdist(X_mnist)
    min_dist = np.min(D)
    logging.info("After PCA - minimum distance between samples is %f", min_dist)


def load_pca_mnist(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    """
    PCA transformer for MNIST data.

    GENERATED WITH: X_mnist
    Regenerating this data will also trigger the regeneration of the data from GENERATED_WITH list.
    PREDECESSOR: X_mnist_raw

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
        "num_pca_dimensions": number of kept dimesions after PCA decomposition.
        "random_pca_seed": random seed for PCA calculation.
    Remaining parameter values will be ignored. See default_num_images_raw and default_selection_random_seed for
    defaults.
    :param regenerate: Forces to regenerate, even if the file exists
    :param recursive_regenerate: Regenerate predecessors as well (takes effect only if regeneration is requested or
           required).
    :return: PCA transformer
    """
    return load_or_remake(get_mnist_pca_filename, generate_pca_mnist, parameters, regenerate, recursive_regenerate)


def load_x_mnist(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    """
    PCA transformer for MNIST data.

    GENERATED WITH: mnist_pca
    Regenerating this data will also trigger the regeneration of the data from GENERATED_WITH list.
    PREDECESSOR: X_mnist_raw

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
        "num_pca_dimensions": number of kept dimesions after PCA decomposition.
        "pca_random_seed": random seed for PCA calculation.
    Remaining parameter values will be ignored. See default_num_images_raw and default_selection_random_seed for
    defaults.
    :param regenerate: Forces to regenerate, even if the file exists
    :param recursive_regenerate: Regenerate predecessors as well (takes effect only if regeneration is requested or
           required).
    :return: PCA transformer
    """
    return load_or_remake(get_x_mnist_filename, generate_pca_mnist, parameters, regenerate, recursive_regenerate)


def generate_y_mnist(*, parameters=settings.parameters, recursive_regenerate=False):
    """

    PREDECESSOR: X_mnist

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
        "num_pca_dimensions": number of kept dimesions after PCA decomposition.
        "pca_random_seed": random seed for PCA calculation.
        "tsne_random_seed": random seed for tSNE algorithm
        "tsne_perpelxity": perplexity for tSNE algorithm
    :param recursive_regenerate: Regenerate predecessors as well
    """
    import lion_tsne
    tsne_random_seed = parameters.get("tsne_random_seed", settings.parameters["tsne_random_seed"])
    tsne_perplexity = parameters.get("tsne_perplexity", settings.parameters["tsne_perplexity"])
    tsne_momentum = parameters.get("tsne_momentum", settings.parameters["tsne_momentum"])
    tsne_n_iters = parameters.get("tsne_n_iters", settings.parameters["tsne_n_iters"])
    tsne_early_exaggeration_iters = parameters.get("tsne_early_exaggeration_iters",
                                                   settings.parameters["tsne_early_exaggeration_iters"])

    X_mnist=load_x_mnist(parameters=parameters,regenerate=recursive_regenerate,
                         recursive_regenerate=recursive_regenerate)

    dTSNE_mnist = lion_tsne.LionTSNE(perplexity=tsne_perplexity)
    Y_mnist = dTSNE_mnist.fit(X_mnist, optimizer_kwargs={'momentum': tsne_momentum,
                                                         'n_iter': tsne_n_iters,
                                                         'early_exaggeration_iters' : tsne_early_exaggeration_iters},
                              random_seed=tsne_random_seed, verbose=2)
    save_and_report(get_y_mnist_filename, parameters, Y_mnist)
    save_and_report(get_dtsne_mnist_filename, parameters, dTSNE_mnist)


def load_y_mnist(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    """
    tSNE-transformed MNIST data

    PREDECESSOR: X_mnist
    GENERATED WITH: dTSNE_mnist - LION-tSNE embedder.
    Regenerating this data will also trigger the regeneration of the data from GENERATED_WITH list.

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
        "num_pca_dimensions": number of kept dimesions after PCA decomposition.
        "pca_random_seed": random seed for PCA calculation.
        "tsne_random_seed": random seed for tSNE algorithm
        "tsne_perpelxity": perplexity for tSNE algorithm
    Remaining parameter values will be ignored. See default_* for defaults.
    :param regenerate: Forces to regenerate, even if the file exists
    :param recursive_regenerate: Regenerate predecessors as well (takes effect only if regeneration is requested or
           required).
    :return: tSNE-transformed MNIST data
    """
    return load_or_remake(get_y_mnist_filename, generate_y_mnist, parameters, regenerate, recursive_regenerate)


def load_dtsne_mnist(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    """
    LION-tSNE transformer for MNIST data.

    PREDECESSOR: X_mnist
    GENERATED WITH: dTSNE_mnist - LION-tSNE embedder.
    Regenerating this data will also trigger the regeneration of the data from GENERATED_WITH list.

    :param parameters: dictionary. Can contain those values:
        "num_images_raw": Number of the images to select from MNIST. Later only non-similar images will
        be kept.
        "selection_random_seed": Random seed for selecting random images from entire MNIST.
        "num_pca_dimensions": number of kept dimesions after PCA decomposition.
        "pca_random_seed": random seed for PCA calculation.
        "tsne_random_seed": random seed for tSNE algorithm
        "tsne_perpelxity": perplexity for tSNE algorithm
    Remaining parameter values will be ignored. See default_* for defaults.
    :param regenerate: Forces to regenerate, even if the file exists
    :param recursive_regenerate: Regenerate predecessors as well (takes effect only if regeneration is requested or
           required).
    :return: LION-tSNE transformer for MNIST data.
    """
    return load_or_remake(get_dtsne_mnist_filename, generate_y_mnist, parameters, regenerate, recursive_regenerate)


def generate_picked_neighbors(parameters=settings.parameters, recursive_regenerate=False):
    """
    Generates temp file with

    :param parameters: Algorithm parameters. See default_ for default values
    :param recursive_regenerate: Regenerate all dependencies.
    """
    # We pick 10 random training points for which there exists a neighbor in not chosen incides, and that neighbor is
    # closer to
    X_mnist = load_x_mnist(parameters=parameters, regenerate=recursive_regenerate,
                           recursive_regenerate=recursive_regenerate)
    # Upper line will regenerate everything, if needed.
    # Lower loads can be non-recursive.
    X_mnist_raw = load_x_mnist_raw(parameters=parameters)
    mnist_pca = load_pca_mnist(parameters=parameters)
    mnist_chosen_indices = load_mnist_chosen_indices(parameters=parameters)
    labels_mnist = load_labels_mnist(parameters=parameters)
    (all_mnist_images, all_mnist_labels) = load_keras_mnist(parameters=parameters)

    ind_to_pick = parameters.get("neighbor_indices_to_pick", settings.parameters["neighbor_indices_to_pick"])
    neighbor_picking_random_seed = parameters.get("neighbor_picking_random_seed",
                                                  settings.parameters["neighbor_picking_random_seed"])

    ind_unchosen = [i for i in range(len(all_mnist_labels)) if i not in mnist_chosen_indices]
    np.random.seed(neighbor_picking_random_seed)  # Any seed, just don't use it again for selecting indices from same dataset

    X_mnist_unchosen_raw = all_mnist_images[ind_unchosen]
    labels_mnist_unchosen = all_mnist_labels[ind_unchosen]

    X_mnist_unchosen_pca = mnist_pca.transform(X_mnist_unchosen_raw)

    picked_indices = list()
    chosen_labels = list()
    nearest_training_indices = list()
    picked_neighbors = np.zeros((ind_to_pick, X_mnist.shape[1]))
    picked_neighbor_labels = list()
    picked_neighbors_raw = np.zeros((ind_to_pick, X_mnist_raw.shape[1]))

    remaining_indices = list(np.arange(len(mnist_chosen_indices)))
    while (len(picked_indices) < ind_to_pick) and (len(remaining_indices) > 0):
        chosen_index = np.random.choice(remaining_indices, 1)
        remaining_indices.remove(chosen_index)

        # Without PCA
        # chosen_raw_value = X_mnist_raw[chosen_index, :]
        # distances_in_training_samples = np.sum((X_mnist_raw - chosen_raw_value)**2, axis=1)
        # With PCA
        chosen_value = X_mnist[chosen_index, :]
        distances_in_training_samples = np.sum((X_mnist - chosen_value) ** 2, axis=1)

        distances_in_training_samples[
            distances_in_training_samples == 0] = np.infty  # We don't need training sample itself
        closest_in_training_samples = np.argmin(distances_in_training_samples)
        closest_distance_in_training_samples = np.min(distances_in_training_samples)

        # Without PCA
        # distances_to_unchosen = np.sum((X_mnist_unchosen_raw - chosen_value)**2, axis=1)
        # With PCA
        distances_to_unchosen = np.sum((X_mnist_unchosen_pca - chosen_value) ** 2, axis=1)

        distances_to_unchosen[distances_to_unchosen == 0] = np.infty  # We don't need exactly the same samples
        if np.min(distances_to_unchosen) < closest_distance_in_training_samples:
            picked_neighbors_raw[len(picked_indices), :] = X_mnist_unchosen_raw[np.argmin(distances_to_unchosen), :]
            picked_neighbors[len(picked_indices), :] = X_mnist_unchosen_pca[np.argmin(distances_to_unchosen), :]
            picked_indices.append(chosen_index[0])
            chosen_labels.append(labels_mnist[chosen_index[0]])
            picked_neighbor_labels.append(labels_mnist_unchosen[np.argmin(distances_to_unchosen)])
            nearest_training_indices.append(closest_in_training_samples)
            logging.info("\tFound neighbor")
        else:
            logging.info("\tNothing closer than one of training samples")
            pass

    save_and_report(get_picked_neighbors_filename, parameters, picked_neighbors)
    save_and_report(get_picked_neighbors_labels_filename, parameters, picked_neighbor_labels)
    save_and_report(get_picked_neighbors_raw_filename, parameters, picked_neighbors_raw)
    save_and_report(get_nearest_training_indices_filename, parameters, nearest_training_indices)
    save_and_report(get_chosen_labels_filename, parameters, chosen_labels)


def load_picked_neighbors(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    return load_or_remake(get_picked_neighbors_filename, generate_picked_neighbors, parameters, regenerate,
                          recursive_regenerate)


def load_picked_neighbors_labels(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    return load_or_remake(get_picked_neighbors_labels_filename, generate_picked_neighbors, parameters, regenerate,
                          recursive_regenerate)


def load_picked_neighbors_raw(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    return load_or_remake(get_picked_neighbors_raw_filename, generate_picked_neighbors, parameters, regenerate,
                          recursive_regenerate)


def load_nearest_training_indices(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    return load_or_remake(get_nearest_training_indices_filename, generate_picked_neighbors, parameters, regenerate,
                          recursive_regenerate)


def load_chosen_labels(*, parameters=settings.parameters, regenerate=False, recursive_regenerate=False):
    return load_or_remake(get_chosen_labels_filename, generate_picked_neighbors, parameters, regenerate,
                          recursive_regenerate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #load_y_mnist(regenerate=True, recursive_regenerate=True)
    #generate_picked_neighbors(recursive_regenerate=True)
    pass
