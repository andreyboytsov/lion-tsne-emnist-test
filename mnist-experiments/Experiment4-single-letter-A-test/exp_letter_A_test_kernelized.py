import settings
import numpy as np
import pickle
import generate_data
import kernelized_tsne
import logging


def generate_letter_A_results_filename(parameters=settings.parameters):
    cluster_results_file_prefix = '../results/letter_A_kernelized_'
    return cluster_results_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.letter_A_parameter_set, parameters)


def main(parameters = settings.parameters, regenerate_parameters_cache=False):
    step = 0.01
    choice_K = np.arange(step, 3 + step, step)  # Let's try those K.

    letter_A_samples, _ = generate_data.load_A_letters(parameters=parameters)
    kernel_tsne_mapping = kernelized_tsne.generate_kernelized_tsne_mapping_function(
        parameters=parameters,
        regenerate_parameters_cache=regenerate_parameters_cache
    )
    kernelized_detailed_tsne_letter_As_results = [kernel_tsne_mapping(letter_A_samples, k=k) for k in choice_K]
    kernelized_detailed_tsne_method_list = ["Kernelized tSNE; K=%.2f" % (k) for k in choice_K]

    output_file = generate_letter_A_results_filename(parameters=parameters)
    with open(output_file,'wb') as f:
            pickle.dump((kernelized_detailed_tsne_letter_As_results, kernelized_detailed_tsne_method_list), f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parameters=settings.parameters, regenerate_parameters_cache=False)
