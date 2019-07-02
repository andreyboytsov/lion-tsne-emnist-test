import settings
import numpy as np
import pickle
import generate_data
import kernelized_tsne


def generate_outlier_results_filename(parameters=settings.parameters):
    cluster_results_file_prefix = '../results/outlier_test_kernelized_'
    return cluster_results_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)


def main(parameters = settings.parameters, regenerate_parameters_cache=False):
    step = 0.01
    choice_K = np.arange(step, 3 + step, step)  # Let's try those K.

    outlier_samples, _ = generate_data.load_outliers(parameters=parameters)
    kernel_tsne_mapping = kernelized_tsne.generate_kernelized_tsne_mapping_function(
        parameters=parameters,
        regenerate_parameters_cache=regenerate_parameters_cache
    )
    kernelized_detailed_tsne_outliers_results = [kernel_tsne_mapping(outlier_samples, k=k) for k in choice_K]
    kernelized_detailed_tsne_method_list = ["Kernelized tSNE; K=%.2f" % (k) for k in choice_K]

    output_file = generate_outlier_results_filename(parameters=parameters)
    with open(output_file,'wb') as f:
            pickle.dump((kernelized_detailed_tsne_outliers_results, kernelized_detailed_tsne_method_list), f)


if __name__ == '__main__':
    main(parameters=settings.parameters, regenerate_parameters_cache=False)
