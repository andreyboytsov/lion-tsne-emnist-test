"""
EXPERIMENT:

letter_A placement test: repeated gradient descent.
"""
import numpy as np
import pickle
import datetime
import sys


def generate_letter_A_results_filename(i):
    return './letter-A-results/letter_A_gd_'+str(i)+".p"


def main(i):
    with open('dtsne_mnist.p', "rb") as f:
        dTSNE_mnist = pickle.load(f)
    with open('y_mnist.p', "rb") as f:
        Y_mnist = pickle.load(f)
    with open('generated_A_letters.p', "rb") as f:
        letter_A_samples, _ = pickle.load(f)

    # Doing it from scratch takes REALLY long time. If possible, save results & pre-load
    output_file = generate_letter_A_results_filename(i)
    
    np.random.seed(
        i)  # We reset random seed every time. Otherwise, if you load partial results from file, everything
    # will depend on which parts were loaded, random sequence will "shift" depend on that, and reproducibility will be lost.
    # I.e. if put seed(0) before the loop and start from scratch, then you will have some random sequence [abc] for sample 0,
    # other (continuation of that sequence) [def] for sample 1, etc. But if you already loaded sample 0 from file, you will
    # have [abc] for sample 1, [def] for sample 2, etc. Reproducibility should not depend on what parts are loaded.
    # Hence, random seed every time, and it depends on ABSOLUTE sample number.
    print(" ====================== Sample %d \n\n", i)

    letter_A = letter_A_samples[i].reshape((1, -1))

    embedder_start_time = datetime.datetime.now()
    letter_As_y_gd_transformed = dTSNE_mnist.transform(letter_A, y='closest',
                                                            verbose=2,
                                                            optimizer_kwargs={'early_exaggeration': None})
    embedder_end_time = datetime.datetime.now()
    letter_As_y_time_gd_transformed = (embedder_end_time - embedder_start_time).total_seconds()
    print("Time: ", letter_As_y_time_gd_transformed)

    embedder_start_time = datetime.datetime.now()
    letter_As_y_gd_variance_recalc_transformed = dTSNE_mnist.transform(letter_A, keep_sigmas=False,
                                                                            y='closest',
                                                                            verbose=2, optimizer_kwargs={
            'early_exaggeration': None})
    embedder_end_time = datetime.datetime.now()
    letter_As_y_time_gd_variance_recalc_transformed = \
        (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (VR): ", letter_As_y_time_gd_variance_recalc_transformed)

    # Let's pick random starts at any point. not necessary near the center.
    y_start = np.array([[
        np.random.uniform(np.min(Y_mnist[:, 0]), np.max(Y_mnist[:, 0])),
        np.random.uniform(np.min(Y_mnist[:, 1]), np.max(Y_mnist[:, 1]))
    ]])

    letter_As_random_starting_positions = y_start

    embedder_start_time = datetime.datetime.now()
    letter_As_y_gd_transformed_random = dTSNE_mnist.transform(letter_A, y=y_start,  # y='random',
                                                                   verbose=2, optimizer_kwargs={
            'early_exaggeration': None})
    embedder_end_time = datetime.datetime.now()
    letter_As_y_time_gd_transformed_random = \
        (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (random): ", letter_As_y_time_gd_transformed_random)


    embedder_start_time = datetime.datetime.now()
    letter_As_y_gd_variance_recalc_transformed_random = dTSNE_mnist.transform(letter_A,
                                                                                   keep_sigmas=False, y=y_start,
                                                                                   # y='random',
                                                                                   verbose=2, optimizer_kwargs={
            'early_exaggeration': None})
    embedder_end_time = datetime.datetime.now()
    letter_As_y_time_gd_variance_recalc_transformed_random = \
        (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (VR, random): %f s", letter_As_y_time_gd_variance_recalc_transformed_random)

    embedder_start_time = datetime.datetime.now()
    letter_As_y_gd_early_exagg_transformed_random = dTSNE_mnist.transform(letter_A, y=y_start,
                                                                               # y='random',
                                                                               verbose=2)
    embedder_end_time = datetime.datetime.now()
    letter_As_y_time_gd_early_exagg_transformed_random = \
        (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (EE, random): %f s", letter_As_y_time_gd_early_exagg_transformed_random)

    embedder_start_time = datetime.datetime.now()
    letter_As_y_gd_early_exagg_transformed = dTSNE_mnist.transform(letter_A, y='closest', verbose=2)
    embedder_end_time = datetime.datetime.now()
    letter_As_y_time_gd_early_exagg_transformed = \
        (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (EE): %f s", letter_As_y_time_gd_early_exagg_transformed)


    embedder_start_time = datetime.datetime.now()
    letter_As_y_gd_variance_recalc_early_exagg_transformed_random = dTSNE_mnist.transform(letter_A,
                                                                                               y=y_start,
                                                                                               keep_sigmas=False,
                                                                                               verbose=2)
    embedder_end_time = datetime.datetime.now()
    letter_As_y_time_gd_variance_recalc_early_exagg_transformed_random = \
        (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (VR,EE,random): %f s",
                 letter_As_y_time_gd_variance_recalc_early_exagg_transformed_random)


    embedder_start_time = datetime.datetime.now()
    letter_As_y_gd_variance_recalc_early_exagg_transformed = dTSNE_mnist.transform(letter_A,
                                                                                        keep_sigmas=False,
                                                                                        y='closest', verbose=2)
    embedder_end_time = datetime.datetime.now()
    letter_As_y_time_gd_variance_recalc_early_exagg_transformed = \
        (embedder_end_time - embedder_start_time).total_seconds()
    print("Time (VR,EE): %f s",
                 letter_As_y_time_gd_variance_recalc_early_exagg_transformed)

    # Gradient descent results take a long while. Let's cache.
    with open(output_file, 'wb') as f:
        pickle.dump((letter_As_y_gd_transformed, letter_As_y_gd_variance_recalc_transformed,
                    letter_As_y_gd_transformed_random,
                    letter_As_y_gd_variance_recalc_transformed_random,
                    letter_As_y_gd_early_exagg_transformed_random,
                    letter_As_y_gd_early_exagg_transformed,
                    letter_As_y_gd_variance_recalc_early_exagg_transformed_random,
                    letter_As_random_starting_positions,
                    letter_As_y_gd_variance_recalc_early_exagg_transformed,
                    letter_As_y_time_gd_transformed, letter_As_y_time_gd_variance_recalc_transformed,
                    letter_As_y_time_gd_transformed_random,
                    letter_As_y_time_gd_variance_recalc_transformed_random,
                    letter_As_y_time_gd_early_exagg_transformed_random,
                    letter_As_y_time_gd_early_exagg_transformed,
                    letter_As_y_time_gd_variance_recalc_early_exagg_transformed_random,
                    letter_As_y_time_gd_variance_recalc_early_exagg_transformed), f)


if __name__ == "__main__":
    main(i=int(sys.argv[1]))
