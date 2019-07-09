"""
EXPERIMENT:

letter placement test: neural network models.
"""
import generate_data
import settings
import neural_network_commons
import pickle


def generate_letter_results_filename(parameters=settings.parameters):
    letter_results_file_prefix = '../results/letter_nn_'
    return letter_results_file_prefix + generate_data.combine_prefixes(
        neural_network_commons.nn_model_prefixes | settings.letter_parameter_set, parameters)


def main(regenerate_model1=False, regenerate_model2=False, regenerate_model3=False,
         parameters=settings.parameters):
    letter_samples, _ = generate_data.load_letters(parameters=parameters)

    models_and_results = neural_network_commons.train_or_load_models(regenerate_model1=regenerate_model1,
        regenerate_model3=regenerate_model3,regenerate_model2=regenerate_model2,parameters=parameters)

    model1, model2, model3 = models_and_results["models"]
    Y_nn1_mnist, Y_nn2_mnist, Y_nn3_mnist = models_and_results["Y_predicted"]

    Y_outl1_mnist = model1.predict(letter_samples)
    Y_outl2_mnist = model2.predict(letter_samples)
    Y_outl3_mnist = model3.predict(letter_samples)

    nn_models_orig = [Y_nn1_mnist, Y_nn2_mnist, Y_nn3_mnist]
    nn_method_list = ['NN - 2L; 250N; ReLu; D0.25','NN - 2L; 500N; ReLu; D0.5', 'NN - 1L; 500N; tanh']

    nn_letters_results = [Y_outl1_mnist, Y_outl2_mnist, Y_outl3_mnist]
    output_file = generate_letter_results_filename(parameters)

    with open(output_file, 'wb') as f:
        pickle.dump((nn_letters_results, nn_models_orig, nn_method_list), f)


if __name__ == "__main__":
    main(parameters=settings.parameters)
