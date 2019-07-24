"""
EXPERIMENT:

Cluster attribution test: RBF interpolation, IDW interpolation.
"""
import logging
import settings

import cluster_lion_RBF_IDW_commons
import exp_idw_power_performance

idw_powers = (50, 70, 100)
n_digits = 1

cluster_results_file_prefix = '../results/cluster_attr_IDW_higher'


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
    cluster_lion_RBF_IDW_commons.main(regenerate=True, parameters=settings.parameters,
            experiment_name="IDW", generate_all_embedders=generate_all_embedders,
            cluster_results_file_prefix=cluster_results_file_prefix)
