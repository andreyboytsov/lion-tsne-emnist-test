"""
EXPERIMENT:

Cluster attribution test, neural networks
"""
import generate_data
import os
import settings
from tensorflow import keras
import pickle


def generate_cluster_results_filename(parameters=settings.parameters):
    output_file_prefix = '../results/cluster_attr_nn_'
    return output_file_prefix + generate_data.combine_prefixes(
        settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters)

def main(regenerate_model1=False, regenerate_model2=False, regenerate_model3=False,
         parameters=settings.parameters):
    X_mnist = generate_data.load_x_mnist(parameters=parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=parameters)

    model1_weights_file_prefix = '../results/model1'
    model1_json_file_prefix = '../results/model1'
    model2_weights_file_prefix = '../results/model2'
    model2_json_file_prefix = '../results/model2'
    model3_weights_file_prefix = '../results/model3'
    model3_json_file_prefix = '../results/model3'

    model1_weights_file = model1_weights_file_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters,
            postfix='.hd5')
    model1_json_file = model1_json_file_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters,
            postfix='.json')
    model2_weights_file = model2_weights_file_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters,
            postfix='.hd5')
    model2_json_file = model2_json_file_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters,
            postfix='.json')
    model3_weights_file = model3_weights_file_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters,
            postfix='.hd5')
    model3_json_file = model3_json_file_prefix + generate_data.combine_prefixes(
            settings.tsne_parameter_set | settings.x_neighbors_selection_parameter_set, parameters,
            postfix='.json')

    if not os.path.isfile(model1_weights_file) or regenerate_model1:
        # 2 layers, 250 nodes per layer, ReLu activation, dropout regularization with rate of 0.25.]

        model1 = keras.models.Sequential()
        model1.add(keras.layers.Dense(250, activation='relu', kernel_initializer='normal', input_dim=X_mnist.shape[1]))
        model1.add(keras.layers.Dropout(0.25))
        model1.add(keras.layers.Dense(250, activation='relu', kernel_initializer='normal', input_dim=X_mnist.shape[1]))
        model1.add(keras.layers.Dropout(0.25))
        model1.add(keras.layers.Dense(Y_mnist.shape[1], kernel_initializer='normal'))
        model1.compile(loss='mean_squared_error', optimizer='adam')
        model1.fit(X_mnist, Y_mnist,
                   epochs=5000,
                   verbose=1,
                   validation_data=(X_mnist, Y_mnist))
        with open(model1_json_file, "w") as f:
            f.write(model1.to_json())
        model1.save_weights(model1_weights_file)
    else:
        with open(model1_json_file, "r") as f:
            model1 = keras.models.model_from_json(f.read())
        model1.load_weights(model1_weights_file)
        model1.compile(loss='mean_squared_error', optimizer='adam')

    Y_nn1_mnist = model1.predict(X_mnist)

    if not os.path.isfile(model2_weights_file) or regenerate_model2:
        # 2 layers, 500 nodes per layer, ReLu activation, dropout regularization with rate of 0.5.]

        model2 = keras.models.Sequential()
        model2.add(keras.layers.Dense(500, activation='relu', kernel_initializer='normal', input_dim=X_mnist.shape[1]))
        model2.add(keras.layers.Dropout(0.5))
        model2.add(keras.layers.Dense(500, activation='relu', kernel_initializer='normal', input_dim=X_mnist.shape[1]))
        model2.add(keras.layers.Dropout(0.5))
        model2.add(keras.layers.Dense(Y_mnist.shape[1], kernel_initializer='normal'))
        model2.compile(loss='mean_squared_error', optimizer='adam')
        model2.fit(X_mnist, Y_mnist,
                   epochs=5000,
                   verbose=1,
                   validation_data=(X_mnist, Y_mnist))
        with open(model2_json_file, "w") as f:
            f.write(model2.to_json())
        model2.save_weights(model2_weights_file)
    else:
        with open(model2_json_file, "r") as f:
            model2 = keras.models.model_from_json(f.read())
        model2.load_weights(model2_weights_file)
        model2.compile(loss='mean_squared_error', optimizer='adam')

    Y_nn2_mnist = model2.predict(X_mnist)

    if not os.path.isfile(model3_weights_file) or regenerate_model3:
        # 2 layers, 500 nodes per layer, ReLu activation, dropout regularization with rate of 0.5.]

        model3 = keras.models.Sequential()
        model3.add(keras.layers.Dense(500, activation='tanh', kernel_initializer='normal', input_dim=X_mnist.shape[1]))
        model3.add(keras.layers.Dense(Y_mnist.shape[1], kernel_initializer='normal'))
        model3.compile(loss='mean_squared_error', optimizer='adam')
        model3.fit(X_mnist, Y_mnist,
                   epochs=5000,
                   verbose=1,
                   validation_data=(X_mnist, Y_mnist))
        with open(model3_json_file, "w") as f:
            f.write(model3.to_json())
        model3.save_weights(model3_weights_file)
    else:
        with open(model3_json_file, "r") as f:
            model3 = keras.models.model_from_json(f.read())
        model3.load_weights(model3_weights_file)
        model3.compile(loss='mean_squared_error', optimizer='adam')

    Y_nn3_mnist = model3.predict(X_mnist)

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
