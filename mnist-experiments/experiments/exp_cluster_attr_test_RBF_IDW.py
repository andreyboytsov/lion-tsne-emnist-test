"""
EXPERIMENT:

Cluster attribution test, RBF interpolation.
"""
import generate_data
import logging
import pickle
import datetime
import settings

cluster_results_file_prefix = '../results/cluster_attr_idw_rbf'

rbf_functions = ('multiquadric', 'linear', 'cubic', 'quintic', 'gaussian',
                 'inverse', 'thin-plate')
idw_powers = (20, 10, 27.9, 1, 40)
n_digits = 1

def main():
    parameters = settings.parameters
    start_time = datetime.datetime.now()
    logging.info("IDW/RBF cluster attribution experiment started: %s", start_time)
    cluster_results_file = cluster_results_file_prefix + generate_data.combine_prefixes(parameters.keys(), parameters)

    dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=parameters)

    results = dict()
    embedders = dict()

    for i in rbf_functions:
        embedders["RBF-"+i] = dTSNE_mnist.generate_embedding_function(embedding_function_type='rbf',
                                                                     function_kwargs={'function': i})
        logging.info("Generated embedder RBF-%s",i)

    for p in idw_powers:
        embedders["IDW-"+str(round(p, n_digits))] = dTSNE_mnist.generate_embedding_function(
            embedding_function_type='weighted-inverse-distance', function_kwargs={'power': p})
        logging.info("Generated embedder IDW-%f",round(p, n_digits))

    for i in embedders.keys():
        logging.info("Trying embedder %s", i)
        embedder_start_time = datetime.datetime.now()
        embedded_neighbors = embedders[i](picked_neighbors)
        embedder_end_time = datetime.datetime.now()
        results[i] = {}
        results[i]["TimePerPoint"] = (embedder_end_time-embedder_start_time)/len(embedded_neighbors)
        results[i]["EmbeddedPoints"] = embedded_neighbors
        logging.info("Time to embed a single point: %s", (embedder_end_time-embedder_start_time)/len(embedded_neighbors))

    with open(cluster_results_file, 'wb') as f:
        pickle.dump(results, f)

    end_time = datetime.datetime.now()
    logging.info("IDW/RBF cluster attribution experiment ended: %s", end_time)
    logging.info("IDW/RBF cluster attribution experiment duration: %s", end_time-start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
