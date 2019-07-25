"""
EXPERIMENT:

letter_A test: LION interpolation.

For debug purposes only, does not save anything.
"""
import logging
import datetime
import settings
import numpy as np
from scipy.spatial import distance

import generate_data
import exp_lion_power_performance

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

lion_percentiles = (90, 95, 99, 100)
n_digits = 1


def generate_all_embedders(dTSNE_mnist):
    _, _, lion_optimal_powers = exp_lion_power_performance.load_lion_power_plot()

    embedders = dict()
    # Changing random state to make sure letter_As do not overlap
    for p in lion_percentiles:
        embedders["LION-"+str(p)+"-"+str(round(lion_optimal_powers[p], n_digits))] = \
            dTSNE_mnist.generate_embedding_function(random_state=p,
                    function_kwargs={'radius_x_percentile':p, 'power': lion_optimal_powers[p],
                                     'radius_y_close_percentile': 100})
        logging.info("Generated embedder LION-%d (%f)",p, lion_optimal_powers[p])

    return embedders


def get_common_info(parameters):
    res = {}
    res['dTSNE_mnist'] = generate_data.load_dtsne_mnist(parameters=parameters)
    res['X_mnist'] = generate_data.load_x_mnist(parameters=parameters)
    res['Y_mnist'] = generate_data.load_y_mnist(parameters=parameters)
    letter_A_samples, _ = generate_data.load_A_letters(parameters=parameters)
    res['letter_A_samples'] = letter_A_samples
    D_Y = distance.squareform(distance.pdist(res['Y_mnist']))
    # Now find distance to closest neighbor
    np.fill_diagonal(D_Y, np.inf)  # ... but not to itself
    res['nearest_neighbors_y_dist'] = np.min(D_Y, axis=1)  # Actually, whatever axis
    return res


def process_single_embedder(*, embedder, embedder_name, results, regenerate, common_info, parameters):
    if embedder_name not in results:
        results[embedder_name] = {}

    logging.info("Trying embedder %s", embedder_name)

    need_embedding = ("TimePerPoint" not in results[embedder_name]) or\
                     ("EmbeddedPoints" not in results[embedder_name]) or regenerate
    save_embedding = ("EmbeddedPoints" not in results[embedder_name]) or regenerate
    save_time = ("TimePerPoint" not in results[embedder_name]) or regenerate
    logging.info("Embedding is%srequired", " " if need_embedding else " NOT ")

    embedder_start_time = datetime.datetime.now()
    embedded_letter_As = embedder(common_info['letter_A_samples'])\
        if need_embedding else results[embedder_name]["EmbeddedPoints"]
    embedder_end_time = datetime.datetime.now()

    results[embedder_name]["TimePerPoint"] = (embedder_end_time - embedder_start_time) / len(embedded_letter_As) if save_time \
        else results[embedder_name]["TimePerPoint"]
    results[embedder_name]["EmbeddedPoints"] = embedded_letter_As
    logging.info("Time %s", "SAVED" if save_time else "KEPT")
    logging.info("Embedding %s", "SAVED" if save_embedding else "KEPT")

    logging.info("Time to embed a single point: %s", results[embedder_name]["TimePerPoint"])


def main(*, regenerate=False, parameters=settings.parameters):
    start_time = datetime.datetime.now()
    logging.info("IDW/RBF/LION letter_A experiment started: %s", start_time)

    common_info = get_common_info(parameters)
    results = dict()
    embedders = generate_all_embedders(common_info['dTSNE_mnist'])

    for embedder_name in embedders.keys():
        process_single_embedder(embedder=embedders[embedder_name], embedder_name=embedder_name, results=results,
                regenerate=regenerate, common_info=common_info,
                                parameters=parameters)

    end_time = datetime.datetime.now()
    logging.info("letter_A experiment ended: %s", end_time)
    logging.info("letter_A experiment duration: %s", end_time-start_time)

    _, _, lion_optimal_power = exp_lion_power_performance.load_lion_power_plot()
    lion_method_list = ["LION; $r_x$ at %dth perc.; $p$=%.1f" % (i, lion_optimal_power[i])
                        for i in sorted(lion_optimal_power)]

    lion90_name = [i for i in results.keys() if i.startswith('LION-90')][0]
    letter_As_y_lion90 = results[lion90_name]['EmbeddedPoints']
    lion95_name = [i for i in results.keys() if i.startswith('LION-95')][0]
    letter_As_y_lion95 = results[lion95_name]['EmbeddedPoints']
    lion99_name = [i for i in results.keys() if i.startswith('LION-99')][0]
    letter_As_y_lion99 = results[lion99_name]['EmbeddedPoints']
    lion100_name = [i for i in results.keys() if i.startswith('LION-100')][0]
    letter_As_y_lion100 = results[lion100_name]['EmbeddedPoints']

    Y_mnist = generate_data.load_y_mnist(parameters=parameters)
    point_size_gray = 10
    cur_shown_letter_A_indices = 100
    point_size_interest = 15

    plt.figure(dpi=300)
    plt.gcf().set_size_inches(6.8, 6.8)

    font_properties = FontProperties()
    font_properties.set_family('serif')
    font_properties.set_name('Times New Roman')
    font_properties.set_size(8)

    plt.scatter(Y_mnist[:, 0], Y_mnist[:, 1], c= 'gray', zorder=1, label=None, marker='.',
                              s = point_size_gray)
    h1 = plt.scatter(letter_As_y_lion90[:cur_shown_letter_A_indices, 0],
                    letter_As_y_lion90[:cur_shown_letter_A_indices, 1], c='red', zorder=1, label=None, marker='.',
                                   s = point_size_interest)
    h2 = plt.scatter(letter_As_y_lion95[:cur_shown_letter_A_indices, 0],
                    letter_As_y_lion95[:cur_shown_letter_A_indices, 1], c='blue', zorder=1, label=None, marker='.',
                                   s = point_size_interest)
    h3 = plt.scatter(letter_As_y_lion99[:cur_shown_letter_A_indices, 0],
                    letter_As_y_lion99[:cur_shown_letter_A_indices, 1], c='green', zorder=1, label=None, marker='.',
                                   s = point_size_interest)
    h4 = plt.scatter(letter_As_y_lion100[:cur_shown_letter_A_indices, 0],
                    letter_As_y_lion100[:cur_shown_letter_A_indices, 1], c='purple', zorder=1, label=None, marker='.',
                                   s = point_size_interest)
    plt.legend([h1,h2,h3,h4], lion_method_list, ncol=1, prop=font_properties, borderpad=0.1,handlelength=2,
                           columnspacing = 0, loc = 1, handletextpad=-0.7,frameon=True)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(regenerate=False)

