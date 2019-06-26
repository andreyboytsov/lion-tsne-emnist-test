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

regenerate = False
logging.basicConfig(level=logging.INFO)

for i in ['full']:
    fname = '../figures/PCA_and_tSNE/mnist_tsne_original'+str(i)+'.png'
    if os.path.isfile(fname):
        logging.info("%s exists", fname)
        continue

    parameters = settings.parameters.copy()
    parameters["pca_random_seed"] = i
    labels_mnist = generate_data.load_labels_mnist(parameters=parameters)
    Y_mnist= generate_data.load_y_mnist(parameters=parameters)

    plt.figure(dpi=300)
    font_properties = FontProperties()
    font_properties.set_family('serif')
    font_properties.set_name('Times New Roman')
    font_properties.set_size(9)

    plt.gcf().set_size_inches(3.3,3.3) #Let's set the plot sizes that just fit paper margins
    legend_list = list()
    for l in set(sorted(labels_mnist)):
        plt.scatter(Y_mnist[labels_mnist == l, 0], Y_mnist[labels_mnist == l, 1], marker = '.', s=5)
        legend_list.append(str(l))
    #plt.title("MNIST Dataset - TSNE visualization")
    #plt.tight_layout()
    l = plt.legend(legend_list, bbox_to_anchor=(0.99, 1.025), markerscale=8, prop=font_properties)
    plt.tight_layout(rect=[-0.17, -0.1, 1.03, 1.03])

    plt.tick_params(axis='both', which='both',bottom=False,top=False,labelbottom=False,
                                              left=False,right=False,labelleft=False)

    plt.savefig(fname)