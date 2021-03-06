import settings
import numpy as np
import pickle
import generate_data
import kernelized_tsne
import logging
import datetime


def generate_outlier_results_filename(parameters=settings.parameters):
    cluster_results_file_prefix = '../results/outlier_kernelized_'
    return cluster_results_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.outlier_parameter_set, parameters)


def main(parameters = settings.parameters, regenerate_parameters_cache=False):
    step = 0.01
    choice_K = np.arange(step, 2 + step, step)  # Let's try those K.

    outlier_samples, _ = generate_data.load_outliers(parameters=parameters)
    kernel_tsne_mapping = kernelized_tsne.generate_kernelized_tsne_mapping_function(
        parameters=parameters,
        regenerate_parameters_cache=regenerate_parameters_cache
    )
    kernelized_detailed_tsne_method_list = ["Kernelized tSNE; K=%.2f" % (k) for k in choice_K]
    kernelized_detailed_tsne_outliers_results = list()
    kernelized_detailed_tsne_time = np.zeros((len(kernelized_detailed_tsne_method_list),))

    for j in range(len(choice_K)):
        k = choice_K[j]
        logging.info("%f", k)

        embedder_start_time = datetime.datetime.now()
        kernelized_detailed_tsne_outliers_results.append(kernel_tsne_mapping(outlier_samples, k=k))
        embedder_end_time = datetime.datetime.now()
        kernelized_detailed_tsne_time[j] = (embedder_end_time - embedder_start_time).total_seconds()
        logging.info("%f complete: %f s", k, kernelized_detailed_tsne_time[j])

    output_file = generate_outlier_results_filename(parameters=parameters)
    with open(output_file,'wb') as f:
            pickle.dump((kernelized_detailed_tsne_outliers_results, kernelized_detailed_tsne_time,
                         kernelized_detailed_tsne_method_list), f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parameters=settings.parameters, regenerate_parameters_cache=True)
