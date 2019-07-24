"""
EXPERIMENT:

Cluster attribution test, repeated gradient descent application.
This is the modification used for Uni.lu HPC cluster.
The goal is to split the jobs.
"""
import pickle
import numpy as np
import datetime
import sys


def generate_cluster_results_filename(i):
    return './cluster-results/cluster_attr_gd_'+str(i)+'.p'


def main(i):
    with open('dtsne_mnist.p', "rb") as f:
        dTSNE_mnist = pickle.load(f)
    with open('y_mnist.p', "rb") as f:
        Y_mnist = pickle.load(f)
    with open('picked_neighbors.p', "rb") as f:
        picked_neighbors = pickle.load(f)

    # Doing it from scratch takes REALLY long time. If possible, save results & pre-load

    # These are consequences of parallelization
    output_file = generate_cluster_results_filename(i)

    np.random.seed(i)  # We reset random seed every time. Otherwise, if you load partial results from file, everything
    # will depend on which parts were loaded, random sequence will "shift" depend on that, and reproducibility will be lost.
    # I.e. if put seed(0) before the loop and start from scratch, then you will have some random sequence [abc] for sample 0,
    # other (continuation of that sequence) [def] for sample 1, etc. But if you already loaded sample 0 from file, you will
    # have [abc] for sample 1, [def] for sample 2, etc. Reproducibility should not depend on what parts are loaded.
    # Hence, random seed every time, and it depends on ABSOLUTE sample number.
    print(" ====================== Sample ",i,"\n\n")

    neighbor = picked_neighbors[i].reshape((1, -1))

    embedder_start_time = datetime.datetime.now()
    picked_neighbors_y_gd_transformed = dTSNE_mnist.transform(neighbor, y='closest',
                                                                 verbose=2,
                                                                 optimizer_kwargs={'early_exaggeration': None})
    embedder_end_time = datetime.datetime.now()
    picked_neighbors_y_time_gd_transformed = (embedder_end_time - embedder_start_time).total_seconds()
    print("Time: ", picked_neighbors_y_time_gd_transformed)


    embedder_start_time = datetime.datetime.now()
    picked_neighbors_y_gd_variance_recalc_transformed = dTSNE_mnist.transform(neighbor, keep_sigmas=False,
          y='closest',
          verbose=2, optimizer_kwargs={'early_exaggeration': None})
    embedder_end_time = datetime.datetime.now()
    picked_neighbors_y_time_gd_variance_recalc_transformed = \
                (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (VR): ", picked_neighbors_y_time_gd_variance_recalc_transformed)

    # Let's pick random starts at any point. not necessary near the center.
    y_start = np.array([[
     np.random.uniform(np.min(Y_mnist[:, 0]), np.max(Y_mnist[:, 0])),
     np.random.uniform(np.min(Y_mnist[:, 1]), np.max(Y_mnist[:, 1]))
    ]])

    picked_random_starting_positions = y_start

    embedder_start_time = datetime.datetime.now()
    picked_neighbors_y_gd_transformed_random = dTSNE_mnist.transform(neighbor, y=y_start,  # y='random',
        verbose=2, optimizer_kwargs={
         'early_exaggeration': None})
    embedder_end_time = datetime.datetime.now()
    picked_neighbors_y_time_gd_transformed_random = (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (random): ", picked_neighbors_y_time_gd_transformed_random)

    embedder_start_time = datetime.datetime.now()
    picked_neighbors_y_gd_variance_recalc_transformed_random = dTSNE_mnist.transform(neighbor,
        keep_sigmas=False,
        y=y_start,  # y='random',
        verbose=2,
        optimizer_kwargs={
            'early_exaggeration': None})
    embedder_end_time = datetime.datetime.now()
    picked_neighbors_y_time_gd_variance_recalc_transformed_random = \
                        (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (VR,random): ", picked_neighbors_y_time_gd_variance_recalc_transformed_random)

    embedder_start_time = datetime.datetime.now()
    picked_neighbors_y_gd_early_exagg_transformed_random = dTSNE_mnist.transform(neighbor, y=y_start,
                                                                                    # y='random',
                                                                                    verbose=2)
    embedder_end_time = datetime.datetime.now()
    picked_neighbors_y_time_gd_early_exagg_transformed_random = \
        (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (EE,random): ", picked_neighbors_y_time_gd_early_exagg_transformed_random)

    embedder_start_time = datetime.datetime.now()
    picked_neighbors_y_gd_early_exagg_transformed = dTSNE_mnist.transform(neighbor, y='closest', verbose=2)
    embedder_end_time = datetime.datetime.now()
    picked_neighbors_y_time_gd_early_exagg_transformed = (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (EE): ", picked_neighbors_y_time_gd_early_exagg_transformed)

    embedder_start_time = datetime.datetime.now()
    picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random = \
     dTSNE_mnist.transform(neighbor,
                            y=y_start,
                            keep_sigmas=False,
                            verbose=2)
    embedder_end_time = datetime.datetime.now()
    picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed_random = \
            (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (VR, EE, random): ", picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed_random)

    embedder_start_time = datetime.datetime.now()
    picked_neighbors_y_gd_variance_recalc_early_exagg_transformed = dTSNE_mnist.transform(neighbor,
       keep_sigmas=False,y='closest',verbose=2)
    embedder_end_time = datetime.datetime.now()
    picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed = \
     (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (VR, EE): ", picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed)

    # Gradient descent results take a long while. Let's cache.
    with open(output_file, 'wb') as f:
        pickle.dump((picked_neighbors_y_gd_transformed, picked_neighbors_y_gd_variance_recalc_transformed,
                     picked_neighbors_y_gd_transformed_random, picked_neighbors_y_gd_variance_recalc_transformed_random,
                     picked_neighbors_y_gd_early_exagg_transformed_random,
                     picked_neighbors_y_gd_early_exagg_transformed,
                     picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random,
                     picked_random_starting_positions,
                     picked_neighbors_y_gd_variance_recalc_early_exagg_transformed,
                     picked_neighbors_y_time_gd_transformed, picked_neighbors_y_time_gd_variance_recalc_transformed,
                     picked_neighbors_y_time_gd_transformed_random,
                     picked_neighbors_y_time_gd_variance_recalc_transformed_random,
                     picked_neighbors_y_time_gd_early_exagg_transformed_random,
                     picked_neighbors_y_time_gd_early_exagg_transformed,
                     picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed_random,
                     picked_neighbors_y_time_gd_variance_recalc_early_exagg_transformed), f)


if __name__ == "__main__":
    main(i=int(sys.argv[1]))
