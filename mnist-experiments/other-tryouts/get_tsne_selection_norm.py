# At some point I lost reproducibility because PCA required random state, and I did not lock it
# (turns out it used some solvers use approximation with randomization needed).
# That is an attempt to reproduce old results by trying different random seeds and comparing the picture.
# It is unlikely to work, but why not try.

import matplotlib.pyplot as plt
import generate_data
from matplotlib.font_manager import FontProperties
import settings
import os
import logging
import numpy as np

regenerate = False
logging.basicConfig(level=logging.INFO)

X_mnist_old = generate_data.load_x_mnist()

for i in range(0,10):
    parameters = settings.parameters.copy()
    parameters["pca_random_seed"] = i
    X_mnist_new = generate_data.load_x_mnist(parameters=parameters)
    print(i, np.max(np.abs(X_mnist_old-X_mnist_new)))
