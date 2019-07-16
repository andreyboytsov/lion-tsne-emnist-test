"""
EXPERIMENT:

letter placement test: RBF interpolation, IDW interpolation.
"""
import logging
import settings

import letter_lion_RBF_IDW_commons
import exp_idw_power_performance

letter_results_file_prefix = '../results/letter_RBF_IDW'

rbf_functions = ('multiquadric', 'linear', 'cubic', 'quintic', 'gaussian',
                 'inverse', 'thin-plate')
idw_powers = (1, 10, 20, 40)
n_digits = 1


def generate_all_embedders(dTSNE_mnist):
    _, _, idw_optimal_power = exp_idw_power_performance.load_idw_power_plot()

    embedders = dict()
    for i in rbf_functions:
        embedders["RBF-"+i] = dTSNE_mnist.generate_embedding_function(embedding_function_type='rbf',
                                                                     function_kwargs={'function': i})
        logging.info("Generated embedder RBF-%s",i)
    for p in idw_powers + (idw_optimal_power, ):
        embedders["IDW-"+str(round(p, n_digits))] = dTSNE_mnist.generate_embedding_function(
            embedding_function_type='weighted-inverse-distance', function_kwargs={'power': p})
        logging.info("Generated embedder IDW-%f",round(p, n_digits))
    return embedders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    letter_lion_RBF_IDW_commons.main(regenerate=False, parameters=settings.parameters,
         generate_all_embedders=generate_all_embedders,
         letter_results_file_prefix=letter_results_file_prefix,
         experiment_name="RBF/IDW")

