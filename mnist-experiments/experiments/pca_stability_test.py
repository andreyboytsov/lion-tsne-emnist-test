from sklearn.decomposition import PCA
from scipy.spatial import distance
import generate_data
import settings
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

X_mnist_raw = generate_data.load_x_mnist_raw()

num_pca_dimensions = settings.parameters.get("num_pca_dimensions", settings.parameters["num_pca_dimensions"])
pca_random_seed = settings.parameters.get("pca_random_seed", settings.parameters["pca_random_seed"])

X_mnist_old = np.zeros((X_mnist_raw.shape[0], num_pca_dimensions))

for i in range(1000):
    mnist_pca = PCA(n_components=num_pca_dimensions, svd_solver='full', random_state=i)
    X_mnist = mnist_pca.fit_transform(X_mnist_raw)

    D = distance.pdist(X_mnist)
    min_dist = np.min(D)
    logging.info("After PCA - minimum distance between samples is %f, dist to old %f", min_dist,
                 np.max(np.abs(X_mnist_old-X_mnist)))
    X_mnist_old = X_mnist
