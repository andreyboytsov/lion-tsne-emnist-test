"""
EXPERIMENT:

Cluster attribution test, repeated gradient descent application
"""
import logging
import pickle
import settings
import numpy as np

import generate_data

parameters = settings.parameters

def main(regenerate):
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    Y_mnist= generate_data.load_y_mnist(parameters=parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=parameters)

    # Doing it from scratch takes REALLY long time. If possible, save results & pre-load

    # These are consequences of parallelization
    # input_files = ['gd_results' + str(100 * i) + '_' + str(100 * i + 100) + '.p' for i in range(10)]
    output_file_prefix = '../results/cluster_attr_gd_'
    output_file = output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)
    covered_samples = list()

    first_sample_inc = 0  # Change only if it is one of "Other notebooks just for parallelization"
    last_sample_exclusive = len(picked_neighbors)

    # Let's build all possible combinations. Later we'll decide what to plot
    picked_neighbors_y_gd_transformed = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
    picked_neighbors_y_gd_variance_recalc_transformed = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
    picked_neighbors_y_gd_transformed_random = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
    picked_neighbors_y_gd_variance_recalc_transformed_random = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))

    picked_neighbors_y_gd_early_exagg_transformed_random = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
    picked_neighbors_y_gd_early_exagg_transformed = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))
    picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random = np.zeros(
         (len(picked_neighbors), Y_mnist.shape[1]))
    picked_neighbors_y_gd_variance_recalc_early_exagg_transformed = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))

    picked_random_starting_positions = np.zeros((len(picked_neighbors), Y_mnist.shape[1]))

    for i in range(first_sample_inc, last_sample_exclusive):
         np.random.seed(i)  # We reset random seed every time. Otherwise, if you load partial results from file, everything
         # will depend on which parts were loaded, random sequence will "shift" depend on that, and reproducibility will be lost.
         # I.e. if put seed(0) before the loop and start from scratch, then you will have some random sequence [abc] for sample 0,
         # other (continuation of that sequence) [def] for sample 1, etc. But if you already loaded sample 0 from file, you will
         # have [abc] for sample 1, [def] for sample 2, etc. Reproducibility should not depend on what parts are loaded.
         # Hence, random seed every time, and it depends on ABSOLUTE sample number.
         logging.info(" ====================== Sample %d\n\n", i)
         if i in covered_samples or regenerate:
             logging.info("Already loaded.")
         else:
             neighbor = picked_neighbors[i].reshape((1, -1))
             picked_neighbors_y_gd_transformed[i, :] = dTSNE_mnist.transform(neighbor, y='closest',
                                                                             verbose=2,
                                                                             optimizer_kwargs={'early_exaggeration': None})
             picked_neighbors_y_gd_variance_recalc_transformed[i, :] = dTSNE_mnist.transform(neighbor, keep_sigmas=False,
                                                                                             y='closest',
                                                                                             verbose=2, optimizer_kwargs={
                     'early_exaggeration': None})

             # Let's pick random starts at any point. not necessary near the center.
             y_start = np.array([[
                 np.random.uniform(np.min(Y_mnist[:, 0]), np.max(Y_mnist[:, 0])),
                 np.random.uniform(np.min(Y_mnist[:, 1]), np.max(Y_mnist[:, 1]))
             ]])

             picked_random_starting_positions[i, :] = y_start

             picked_neighbors_y_gd_transformed_random[i, :] = dTSNE_mnist.transform(neighbor, y=y_start,  # y='random',
                    verbose=2, optimizer_kwargs={
                     'early_exaggeration': None})
             picked_neighbors_y_gd_variance_recalc_transformed_random[i, :] = dTSNE_mnist.transform(neighbor,
                    keep_sigmas=False,
                    y=y_start,  # y='random',
                    verbose=2,
                    optimizer_kwargs={
                        'early_exaggeration': None})

             picked_neighbors_y_gd_early_exagg_transformed_random[i, :] = dTSNE_mnist.transform(neighbor, y=y_start,
                                                                                                # y='random',
                                                                                                verbose=2)
             picked_neighbors_y_gd_early_exagg_transformed[i, :] = dTSNE_mnist.transform(neighbor, y='closest', verbose=2)

             picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random[i, :] = \
                 dTSNE_mnist.transform(neighbor,
                                        y=y_start,
                                        keep_sigmas=False,
                                        verbose=2)
             picked_neighbors_y_gd_variance_recalc_early_exagg_transformed[i, :] = dTSNE_mnist.transform(neighbor,
                                                                                                         keep_sigmas=False,
                                                                                                         y='closest',
                                                                                                         verbose=2)

             covered_samples.append(i)
         # Re-saving even if it is a loaded sample
         logging.info("Saving...")
         # Gradient descent results take a long while. Let's cache.
         with open(output_file, 'wb') as f:
             pickle.dump((picked_neighbors_y_gd_transformed, picked_neighbors_y_gd_variance_recalc_transformed,
                          picked_neighbors_y_gd_transformed_random, picked_neighbors_y_gd_variance_recalc_transformed_random,
                          picked_neighbors_y_gd_early_exagg_transformed_random,
                          picked_neighbors_y_gd_early_exagg_transformed,
                          picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random,
                          picked_random_starting_positions,
                          picked_neighbors_y_gd_variance_recalc_early_exagg_transformed, covered_samples), f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(regenerate=False)
