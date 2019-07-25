# Collects HPC results to proper result files

# For each of: cluster, outlier, letter, letter A

# TODO generate time filename and normal filename
# TODO Create original
# TODO Load one-by-one

import settings
import logging
import numpy as np
import generate_data
import pickle

input_prefixes = (
    './cluster-results/cluster_attr_gd_',
    './outlier-results/outlier_gd_',
    './letter-results/letter_gd_',
    './letter-A-results/letter_A_gd_',

)

output_files = (
    '../results/cluster_attr_gd_' + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, settings.parameters),
    '../results/outlier_gd_' + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.outlier_parameter_set, settings.parameters),
    '../results/letter_gd_' + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_parameter_set, settings.parameters),
    '../results/letter_A_gd_' + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_A_parameter_set, settings.parameters),
)

output_time_files = (
    '../results/cluster_attr_time_gd_' + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, settings.parameters),
    '../results/outlier_time_gd_' + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.outlier_parameter_set, settings.parameters),
    '../results/letter_time_gd_' + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_parameter_set, settings.parameters),
    '../results/letter_A_time_gd_' + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_A_parameter_set, settings.parameters),
)


lengths = (  # Easier like that
    1000,
    1000,
    1000,
    100,
)

reduced_dimensions = 2


def main():

    for k in range(len(input_prefixes)):
        logging.info("%s", input_prefixes[k])
        covered_samples = list()

        # Let's build all possible combinations. Later we'll decide what to plot
        y_gd_transformed = np.zeros((lengths[k], reduced_dimensions))
        y_gd_variance_recalc_transformed = np.zeros((lengths[k], reduced_dimensions))
        y_gd_transformed_random = np.zeros((lengths[k], reduced_dimensions))
        y_gd_variance_recalc_transformed_random = np.zeros((lengths[k], reduced_dimensions))

        y_gd_early_exagg_transformed_random = np.zeros((lengths[k], reduced_dimensions))
        y_gd_early_exagg_transformed = np.zeros((lengths[k], reduced_dimensions))
        y_gd_variance_recalc_early_exagg_transformed_random = np.zeros((lengths[k], reduced_dimensions))
        y_gd_variance_recalc_early_exagg_transformed = np.zeros((lengths[k], reduced_dimensions))

        random_starting_positions = np.zeros((lengths[k], reduced_dimensions))

        y_time_gd_transformed = np.zeros((lengths[k], ))
        y_time_gd_variance_recalc_transformed = np.zeros((lengths[k], ))
        y_time_gd_transformed_random = np.zeros((lengths[k], ))
        y_time_gd_variance_recalc_transformed_random = np.zeros((lengths[k], ))

        y_time_gd_early_exagg_transformed_random = np.zeros((lengths[k], ))
        y_time_gd_early_exagg_transformed = np.zeros((lengths[k], ))
        y_time_gd_variance_recalc_early_exagg_transformed_random = np.zeros((lengths[k], ))
        y_time_gd_variance_recalc_early_exagg_transformed = np.zeros((lengths[k], ))

        for i in range(lengths[k]):
            with open(input_prefixes[k] + str(i) + '.p', 'rb') as f:
                covered_samples.append(i)
                (y_gd_transformed[i, :], y_gd_variance_recalc_transformed[i, :],
                y_gd_transformed_random[i, :],
                y_gd_variance_recalc_transformed_random[i, :],
                y_gd_early_exagg_transformed_random[i, :],
                y_gd_early_exagg_transformed[i, :],
                y_gd_variance_recalc_early_exagg_transformed_random[i, :],
                random_starting_positions[i, :],
                y_gd_variance_recalc_early_exagg_transformed[i, :],
                y_time_gd_transformed[i], y_time_gd_variance_recalc_transformed[i],
                y_time_gd_transformed_random[i],
                y_time_gd_variance_recalc_transformed_random[i],
                y_time_gd_early_exagg_transformed_random[i],
                y_time_gd_early_exagg_transformed[i],
                y_time_gd_variance_recalc_early_exagg_transformed_random[i],
                y_time_gd_variance_recalc_early_exagg_transformed[i]) = pickle.load(f)

        with open(output_files[k], 'wb') as f:
            pickle.dump((y_gd_transformed, y_gd_variance_recalc_transformed,
                         y_gd_transformed_random,
                         y_gd_variance_recalc_transformed_random,
                         y_gd_early_exagg_transformed_random,
                         y_gd_early_exagg_transformed,
                         y_gd_variance_recalc_early_exagg_transformed_random,
                         random_starting_positions,
                         y_gd_variance_recalc_early_exagg_transformed, covered_samples), f)
        with open(output_time_files[k], 'wb') as f:
            pickle.dump((y_time_gd_transformed, y_time_gd_variance_recalc_transformed,
                         y_time_gd_transformed_random,
                         y_time_gd_variance_recalc_transformed_random,
                         y_time_gd_early_exagg_transformed_random,
                         y_time_gd_early_exagg_transformed,
                         y_time_gd_variance_recalc_early_exagg_transformed_random,
                         y_time_gd_variance_recalc_early_exagg_transformed, covered_samples), f)

        logging.info("Finished %s", input_prefixes[k])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
