"""
EXPERIMENT:

Outlier placement test: RBF interpolation, IDW interpolation.
"""
import logging
import settings

import outlier_lion_RBF_IDW_commons
import exp_idw_power_performance

outlier_results_file_prefix = '../results/outlier_IDW_higher'

idw_powers = (50, 70, 100)
n_digits = 1


def generate_all_embedders(dTSNE_mnist):
    _, _, idw_optimal_power = exp_idw_power_performance.load_idw_power_plot()

    embedders = dict()
    for p in idw_powers:
        embedders["IDW-"+str(round(p, n_digits))] = dTSNE_mnist.generate_embedding_function(
            embedding_function_type='weighted-inverse-distance', function_kwargs={'power': p})
        logging.info("Generated embedder IDW-%f",round(p, n_digits))
    return embedders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    outlier_lion_RBF_IDW_commons.main(regenerate=False, parameters=settings.parameters,
         generate_all_embedders=generate_all_embedders,
         outlier_results_file_prefix=outlier_results_file_prefix,
         experiment_name="IDW")
