"""
EXPERIMENT:

Cluster attribution test: LION interpolation.
"""
import logging
import settings

import cluster_lion_RBF_IDW_commons
import exp_lion_power_performance


cluster_results_file_prefix = '../results/cluster_attr_LION'
lion_percentiles = (90, 95, 99, 100)
n_digits = 1


def generate_all_embedders(dTSNE_mnist):
    _, _, lion_optimal_powers = exp_lion_power_performance.load_lion_power_plot()

    embedders = dict()
    # Changing random state to make sure outliers do not overlap
    for p in sorted(lion_percentiles):
        embedders["LION-"+str(p)+"-"+str(round(lion_optimal_powers[p], n_digits))] = \
            dTSNE_mnist.generate_embedding_function(random_state=p,
                    function_kwargs={'radius_x_percentile':p, 'power': lion_optimal_powers[p],
                                     'y_safety_margin': 0})
        logging.info("Generated embedder LION-%d (%f)",p, lion_optimal_powers[p])
    return embedders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cluster_lion_RBF_IDW_commons.main(regenerate=True, parameters=settings.parameters,
            experiment_name="LION", generate_all_embedders=generate_all_embedders,
            cluster_results_file_prefix=cluster_results_file_prefix)
