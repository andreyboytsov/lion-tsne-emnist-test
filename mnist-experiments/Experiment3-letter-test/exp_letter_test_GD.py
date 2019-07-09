"""
EXPERIMENT:

letter placement test: repeated gradient descent.
"""
import logging
import settings
import generate_data
import numpy as np
import pickle


def generate_letter_results_filename(parameters=settings.parameters):
    letter_results_file_prefix = '../results/letter_gd_'
    return letter_results_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_parameter_set, parameters)


def main(parameters=settings.parameters):
    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    Y_mnist= generate_data.load_y_mnist(parameters=parameters)
    letter_samples, _, _ = generate_data.load_letters(parameters=parameters)

    # Doing it from scratch takes REALLY long time. If possible, save results & pre-load

    covered_samples = list()

    first_sample_inc = 0  # Change only if it is one of "Other notebooks just for parallelization"
    last_sample_exclusive = len(letter_samples)
    output_file = generate_letter_results_filename(parameters)

    letters_y_gd_transformed = np.zeros((len(letter_samples), Y_mnist.shape[1]))
    letters_y_gd_variance_recalc_transformed = np.zeros((len(letter_samples), Y_mnist.shape[1]))
    letters_y_gd_transformed_random = np.zeros((len(letter_samples), Y_mnist.shape[1]))
    letters_y_gd_variance_recalc_transformed_random = np.zeros((len(letter_samples), Y_mnist.shape[1]))

    letters_y_gd_early_exagg_transformed_random = np.zeros((len(letter_samples), Y_mnist.shape[1]))
    letters_y_gd_early_exagg_transformed = np.zeros((len(letter_samples), Y_mnist.shape[1]))
    letters_y_gd_variance_recalc_early_exagg_transformed_random = np.zeros((len(letter_samples), Y_mnist.shape[1]))
    letters_y_gd_variance_recalc_early_exagg_transformed = np.zeros((len(letter_samples), Y_mnist.shape[1]))

    letters_random_starting_positions = np.zeros((len(letter_samples), Y_mnist.shape[1]))

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
            letter = letter_samples[i].reshape((1, -1))
            letters_y_gd_transformed[i, :] = dTSNE_mnist.transform(letter, y='closest',
                                                                    verbose=2,
                                                                    optimizer_kwargs={'early_exaggeration': None})
            letters_y_gd_variance_recalc_transformed[i, :] = dTSNE_mnist.transform(letter, keep_sigmas=False,
                                                                                    y='closest',
                                                                                    verbose=2, optimizer_kwargs={
                    'early_exaggeration': None})

            # Let's pick random starts at any point. not necessary near the center.
            y_start = np.array([[
                np.random.uniform(np.min(Y_mnist[:, 0]), np.max(Y_mnist[:, 0])),
                np.random.uniform(np.min(Y_mnist[:, 1]), np.max(Y_mnist[:, 1]))
            ]])

            letters_random_starting_positions[i, :] = y_start

            letters_y_gd_transformed_random[i, :] = dTSNE_mnist.transform(letter, y=y_start,  # y='random',
                                                                           verbose=2, optimizer_kwargs={
                    'early_exaggeration': None})
            letters_y_gd_variance_recalc_transformed_random[i, :] = dTSNE_mnist.transform(letter,
                                                                                           keep_sigmas=False, y=y_start,
                                                                                           # y='random',
                                                                                           verbose=2, optimizer_kwargs={
                    'early_exaggeration': None})

            letters_y_gd_early_exagg_transformed_random[i, :] = dTSNE_mnist.transform(letter, y=y_start,
                                                                                       # y='random',
                                                                                       verbose=2)
            letters_y_gd_early_exagg_transformed[i, :] = dTSNE_mnist.transform(letter, y='closest', verbose=2)

            letters_y_gd_variance_recalc_early_exagg_transformed_random[i, :] = dTSNE_mnist.transform(letter,
                                                                                                       y=y_start,
                                                                                                       keep_sigmas=False,
                                                                                                       verbose=2)
            letters_y_gd_variance_recalc_early_exagg_transformed[i, :] = dTSNE_mnist.transform(letter,
                                                                                                keep_sigmas=False,
                                                                                                y='closest', verbose=2)

        covered_samples.append(i)
        logging.info("Saving...")
        # Gradient descent results take a long while. Let's cache.
        with open(output_file, 'wb') as f:
            pickle.dump((letters_y_gd_transformed, letters_y_gd_variance_recalc_transformed,
                         letters_y_gd_transformed_random, letters_y_gd_variance_recalc_transformed_random,
                         letters_y_gd_early_exagg_transformed_random, letters_y_gd_early_exagg_transformed,
                         letters_y_gd_variance_recalc_early_exagg_transformed_random,
                         letters_random_starting_positions,
                         letters_y_gd_variance_recalc_early_exagg_transformed, covered_samples), f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
