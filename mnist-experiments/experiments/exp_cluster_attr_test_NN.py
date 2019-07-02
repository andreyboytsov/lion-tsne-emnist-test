"""
EXPERIMENT:

Cluster attribution test, neural networks
"""
import generate_data
import os
import settings
from tensorflow import keras
import pickle
import neural_network_commons


def generate_cluster_results_filename(parameters=settings.parameters):
    output_file_prefix = '../results/cluster_attr_nn_'
    return output_file_prefix + generate_data.combine_prefixes(neural_network_commons.required_prefixes, parameters)


def main(regenerate_model1=False, regenerate_model2=False, regenerate_model3=False,
         parameters=settings.parameters):
    models_and_results = neural_network_commons.train_or_load_models(regenerate_model1=regenerate_model1,
        regenerate_model3=regenerate_model3,regenerate_model2=regenerate_model2,parameters=parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=parameters)

    model1, model2, model3 = models_and_results["models"]
    Y_nn1_mnist, Y_nn2_mnist, Y_nn3_mnist = models_and_results["Y_predicted"]

    Y_neighb1_mnist = model1.predict(picked_neighbors)
    Y_neighb2_mnist = model2.predict(picked_neighbors)
    Y_neighb3_mnist = model3.predict(picked_neighbors)

    nn_method_results = [Y_neighb1_mnist, Y_neighb2_mnist, Y_neighb3_mnist]
    nn_models_orig = [Y_nn1_mnist, Y_nn2_mnist, Y_nn3_mnist]
    nn_method_list = ['NN - 2L; 250N; ReLu; D0.25','NN - 2L; 500N; ReLu; D0.5', 'NN - 1L; 500N; tanh']

    output_file = generate_cluster_results_filename(parameters)

    with open(output_file, 'wb') as f:
        pickle.dump((nn_method_results, nn_models_orig, nn_method_list), f)


if __name__ == "__main__":
    main(regenerate_model1=False, parameters = settings.parameters)
