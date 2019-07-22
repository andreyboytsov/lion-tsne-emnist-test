"""
EXPERIMENT:

Outlier placement test: repeated gradient descent.
"""
import logging
import settings
import generate_data
import numpy as np
import pickle
import datetime
import os


def generate_outlier_results_filename(parameters=settings.parameters):
    outlier_results_file_prefix = '../results/outlier_gd_'
    return outlier_results_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.outlier_parameter_set, parameters)


def generate_time_results_filename(parameters=settings.parameters):
    outlier_results_file_prefix = '../results/outlier_time_gd_'
    return outlier_results_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.outlier_parameter_set, parameters)


def main(parameters=settings.parameters, regenerate=False, only_time=False):
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    Y_mnist= generate_data.load_y_mnist(parameters=parameters)
    outlier_samples, _ = generate_data.load_outliers(parameters=parameters)

    output_file = generate_outlier_results_filename(parameters)
    output_time_file = generate_time_results_filename(parameters)

    first_sample_inc = 0  # Change only if it is one of "Other notebooks just for parallelization"
    last_sample_exclusive = len(outlier_samples)

    # Doing it from scratch takes REALLY long time. If possible, save results & pre-load
    if os.path.isfile(output_file) and not regenerate:
        logging.info("Found previous partially completed test. Starting from there.")
        with open(output_file, 'rb') as f:
            (outliers_y_gd_transformed, outliers_y_gd_variance_recalc_transformed,
             outliers_y_gd_transformed_random,
             outliers_y_gd_variance_recalc_transformed_random,
             outliers_y_gd_early_exagg_transformed_random,
             outliers_y_gd_early_exagg_transformed,
             outliers_y_gd_variance_recalc_early_exagg_transformed_random,
             outliers_random_starting_positions,
             outliers_y_gd_variance_recalc_early_exagg_transformed, covered_samples) = pickle.load(f)
        with open(output_time_file, 'rb') as f:
            (outliers_y_time_gd_transformed, outliers_y_time_gd_variance_recalc_transformed,
             outliers_y_time_gd_transformed_random,
             outliers_y_time_gd_variance_recalc_transformed_random,
             outliers_y_time_gd_early_exagg_transformed_random,
             outliers_y_time_gd_early_exagg_transformed,
             outliers_y_time_gd_variance_recalc_early_exagg_transformed_random,
             outliers_y_time_gd_variance_recalc_early_exagg_transformed, covered_samples) = pickle.load(f)
    else:
        logging.info("No previous partially completed test, or regeneration requested. Starting from scratch.")
        covered_samples = list()

        outliers_y_gd_transformed = np.zeros((len(outlier_samples), Y_mnist.shape[1]))
        outliers_y_gd_variance_recalc_transformed = np.zeros((len(outlier_samples), Y_mnist.shape[1]))
        outliers_y_gd_transformed_random = np.zeros((len(outlier_samples), Y_mnist.shape[1]))
        outliers_y_gd_variance_recalc_transformed_random = np.zeros((len(outlier_samples), Y_mnist.shape[1]))

        outliers_y_gd_early_exagg_transformed_random = np.zeros((len(outlier_samples), Y_mnist.shape[1]))
        outliers_y_gd_early_exagg_transformed = np.zeros((len(outlier_samples), Y_mnist.shape[1]))
        outliers_y_gd_variance_recalc_early_exagg_transformed_random = np.zeros((len(outlier_samples), Y_mnist.shape[1]))
        outliers_y_gd_variance_recalc_early_exagg_transformed = np.zeros((len(outlier_samples), Y_mnist.shape[1]))

        outliers_random_starting_positions = np.zeros((len(outlier_samples), Y_mnist.shape[1]))

        outliers_y_time_gd_transformed = np.zeros((len(outlier_samples), ))
        outliers_y_time_gd_variance_recalc_transformed = np.zeros((len(outlier_samples), ))
        outliers_y_time_gd_transformed_random = np.zeros((len(outlier_samples), ))
        outliers_y_time_gd_variance_recalc_transformed_random = np.zeros((len(outlier_samples), ))

        outliers_y_time_gd_early_exagg_transformed_random = np.zeros((len(outlier_samples), ))
        outliers_y_time_gd_early_exagg_transformed = np.zeros((len(outlier_samples), ))
        outliers_y_time_gd_variance_recalc_early_exagg_transformed_random = np.zeros((len(outlier_samples), ))
        outliers_y_time_gd_variance_recalc_early_exagg_transformed = np.zeros((len(outlier_samples), ))

    for i in range(first_sample_inc, last_sample_exclusive):
        np.random.seed(
            i)  # We reset random seed every time. Otherwise, if you load partial results from file, everything
        # will depend on which parts were loaded, random sequence will "shift" depend on that, and reproducibility will be lost.
        # I.e. if put seed(0) before the loop and start from scratch, then you will have some random sequence [abc] for sample 0,
        # other (continuation of that sequence) [def] for sample 1, etc. But if you already loaded sample 0 from file, you will
        # have [abc] for sample 1, [def] for sample 2, etc. Reproducibility should not depend on what parts are loaded.
        # Hence, random seed every time, and it depends on ABSOLUTE sample number.
        logging.info(" ====================== Sample %d \n\n", i)
        if i in covered_samples:
            logging.info("Already loaded.")
        else:
            outlier = outlier_samples[i].reshape((1, -1))

            embedder_start_time = datetime.datetime.now()
            outliers_y_gd_transformed[i, :] = dTSNE_mnist.transform(outlier, y='closest',
                                                                    verbose=2,
                                                                    optimizer_kwargs={'early_exaggeration': None})
            embedder_end_time = datetime.datetime.now()
            outliers_y_time_gd_transformed[i] = (embedder_end_time - embedder_start_time).total_seconds()
            logging.info("Time: %f s", outliers_y_time_gd_transformed[i])

            embedder_start_time = datetime.datetime.now()
            outliers_y_gd_variance_recalc_transformed[i, :] = dTSNE_mnist.transform(outlier, keep_sigmas=False,
                                                                                    y='closest',
                                                                                    verbose=2, optimizer_kwargs={
                    'early_exaggeration': None})
            embedder_end_time = datetime.datetime.now()
            outliers_y_time_gd_variance_recalc_transformed[i] = \
                (embedder_end_time - embedder_start_time).total_seconds()
            logging.info("Time (VR): %f s", outliers_y_time_gd_variance_recalc_transformed[i])

            # Let's pick random starts at any point. not necessary near the center.
            y_start = np.array([[
                np.random.uniform(np.min(Y_mnist[:, 0]), np.max(Y_mnist[:, 0])),
                np.random.uniform(np.min(Y_mnist[:, 1]), np.max(Y_mnist[:, 1]))
            ]])

            outliers_random_starting_positions[i, :] = y_start

            embedder_start_time = datetime.datetime.now()
            outliers_y_gd_transformed_random[i, :] = dTSNE_mnist.transform(outlier, y=y_start,  # y='random',
                                                                           verbose=2, optimizer_kwargs={
                    'early_exaggeration': None})
            embedder_end_time = datetime.datetime.now()
            outliers_y_time_gd_transformed_random[i] = \
                (embedder_end_time - embedder_start_time).total_seconds()
            logging.info("Time (random): %f s", outliers_y_time_gd_transformed_random[i])


            embedder_start_time = datetime.datetime.now()
            outliers_y_gd_variance_recalc_transformed_random[i, :] = dTSNE_mnist.transform(outlier,
                                                                                           keep_sigmas=False, y=y_start,
                                                                                           # y='random',
                                                                                           verbose=2, optimizer_kwargs={
                    'early_exaggeration': None})
            embedder_end_time = datetime.datetime.now()
            outliers_y_time_gd_variance_recalc_transformed_random[i] = \
                (embedder_end_time - embedder_start_time).total_seconds()
            logging.info("Time (VR, random): %f s", outliers_y_time_gd_variance_recalc_transformed_random[i])

            embedder_start_time = datetime.datetime.now()
            outliers_y_gd_early_exagg_transformed_random[i, :] = dTSNE_mnist.transform(outlier, y=y_start,
                                                                                       # y='random',
                                                                                       verbose=2)
            embedder_end_time = datetime.datetime.now()
            outliers_y_time_gd_early_exagg_transformed_random[i] = \
                (embedder_end_time - embedder_start_time).total_seconds()
            logging.info("Time (EE, random): %f s", outliers_y_time_gd_early_exagg_transformed_random[i])

            embedder_start_time = datetime.datetime.now()
            outliers_y_gd_early_exagg_transformed[i, :] = dTSNE_mnist.transform(outlier, y='closest', verbose=2)
            embedder_end_time = datetime.datetime.now()
            outliers_y_time_gd_early_exagg_transformed[i] = \
                (embedder_end_time - embedder_start_time).total_seconds()
            logging.info("Time (EE): %f s", outliers_y_time_gd_early_exagg_transformed[i])


            embedder_start_time = datetime.datetime.now()
            outliers_y_gd_variance_recalc_early_exagg_transformed_random[i, :] = dTSNE_mnist.transform(outlier,
                                                                                                       y=y_start,
                                                                                                       keep_sigmas=False,
                                                                                                       verbose=2)
            embedder_end_time = datetime.datetime.now()
            outliers_y_time_gd_variance_recalc_early_exagg_transformed_random[i] = \
                (embedder_end_time - embedder_start_time).total_seconds()
            logging.info("Time (VR,EE,random): %f s",
                         outliers_y_time_gd_variance_recalc_early_exagg_transformed_random[i])


            embedder_start_time = datetime.datetime.now()
            outliers_y_gd_variance_recalc_early_exagg_transformed[i, :] = dTSNE_mnist.transform(outlier,
                                                                                                keep_sigmas=False,
                                                                                                y='closest', verbose=2)
            embedder_end_time = datetime.datetime.now()
            outliers_y_time_gd_variance_recalc_early_exagg_transformed[i] = \
                (embedder_end_time - embedder_start_time).total_seconds()
            logging.info("Time (VR,EE): %f s",
                         outliers_y_time_gd_variance_recalc_early_exagg_transformed[i])


        covered_samples.append(i)
        logging.info("Saving...")
        # Gradient descent results take a long while. Let's cache.
        if not only_time:
            with open(output_file, 'wb') as f:
                pickle.dump((outliers_y_gd_transformed, outliers_y_gd_variance_recalc_transformed,
                             outliers_y_gd_transformed_random,
                             outliers_y_gd_variance_recalc_transformed_random,
                             outliers_y_gd_early_exagg_transformed_random,
                             outliers_y_gd_early_exagg_transformed,
                             outliers_y_gd_variance_recalc_early_exagg_transformed_random,
                             outliers_random_starting_positions,
                             outliers_y_gd_variance_recalc_early_exagg_transformed, covered_samples), f)
        with open(output_time_file, 'wb') as f:
            pickle.dump((outliers_y_time_gd_transformed, outliers_y_time_gd_variance_recalc_transformed,
                         outliers_y_time_gd_transformed_random,
                         outliers_y_time_gd_variance_recalc_transformed_random,
                         outliers_y_time_gd_early_exagg_transformed_random,
                         outliers_y_time_gd_early_exagg_transformed,
                         outliers_y_time_gd_variance_recalc_early_exagg_transformed_random,
                         outliers_y_time_gd_variance_recalc_early_exagg_transformed, covered_samples), f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(only_time=False)
