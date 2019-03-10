import settings
from scipy.spatial import distance
import logging
import datetime
import numpy as np
import generate_data
import pickle

lion_power_options = np.arange(0.1, 50.1, step=0.1)
lion_percentile_options = [90, 95, 99, 100]
lion_power_plot_data_file = '../results/lion_power_plot_data'


def main():
    start_time = datetime.datetime.now()
    logging.info("LION power experiment started: %s", start_time)

    lion_power_plot_data = dict()  # Start from scratch

    X_mnist = generate_data.load_x_mnist(parameters=settings.parameters)
    Y_mnist = generate_data.load_y_mnist(parameters=settings.parameters)
    picked_neighbors = generate_data.load_picked_neighbors(parameters=settings.parameters)

    distance_matrix = distance.squareform(distance.pdist(X_mnist))
    np.fill_diagonal(distance_matrix, np.inf)  # We are not interested in distance to itself
    nn_x_distance = np.min(distance_matrix, axis=1)  # Any axis will do
    radius_x = dict()
    for p in lion_percentile_options:
        radius_x[p] = np.percentile(nn_x_distance, p)
    print(radius_x)

    skip_accuracy = False
    recalculate_performance = True

    #if os.path.isfile(lion_power_plot_data_file) and not regenerate_lion_accuracy:
    #    with open(lion_power_plot_data_file, 'rb') as f:
    #        lion_power_plot_data = pickle.load(f)

    for perc in lion_percentile_options:
        for p in lion_power_options:
            print("Processing percentile and power", p, perc)
            key = str(perc) + ";" + "%.3f" % (p)
            print(key)
            if key not in lion_power_plot_data:
                lion_power_plot_data[key] = dict()

            if 'Accuracy' not in lion_power_plot_data[key] and not skip_accuracy:
                print("Accuracy not found for power ", p, " percentile", perc, "\tCalculating...")
                interpolator = dTSNE_mnist.generate_lion_tsne_embedder(verbose=0, random_state=0, function_kwargs={
                    'radius_x_percentile': perc,
                    'power': p})

                per_sample_accuracy = np.zeros((len(picked_neighbors),))
                for i in range(len(picked_neighbors)):
                    # if i%100==0:
                    #    print("\tPower: ",p,"Processing:",i)
                    expected_label = picked_neighbor_labels[i]
                    result = interpolator(picked_neighbors[i], verbose=0)
                    nn_indices = get_nearest_neighbors_in_y(result, n=accuracy_nn)
                    obtained_labels = labels_mnist[nn_indices]
                    per_sample_accuracy[i] = sum(obtained_labels == expected_label) / len(obtained_labels)
                cur_acc = np.mean(per_sample_accuracy)
                # print('================= ',p,perc, cur_acc)
                lion_power_plot_data[key]['Accuracy'] = cur_acc
                with open(lion_power_plot_data_file, 'wb') as f:
                    pickle.dump(lion_power_plot_data, f)

            if 'PowerSquareDist' not in lion_power_plot_data[key] or recalculate_performance:
                print("Power performance not found for power ", p, " percentile", perc, "\tCalculating...")

                y_sum_square_dist = 0.0
                y_sum_abs_dist = 0.0
                y_count = 0.0
                for i in range(len(X_mnist)):
                    distances = distance_matrix[i, :].copy()
                    distances[i] = np.inf  # Not interested in distance to itself
                    # Step 1. Find nearest neighbors in the neighborhood.
                    neighbor_indices = np.where(distances <= radius_x[perc])[0]
                    num_neighbors = len(neighbor_indices)
                    if num_neighbors >= 2:  # Below 2? Cannot interpolate
                        # We are good
                        weights = 1 / distances[neighbor_indices] ** p
                        weights = weights / np.sum(weights)
                        cur_y_result = weights.dot(Y_mnist[neighbor_indices, :])
                        y_sum_square_dist += np.sum(cur_y_result - Y_mnist[i, :]) ** 2
                        y_sum_abs_dist += np.sqrt(np.sum(cur_y_result - Y_mnist[i, :]) ** 2)
                        y_count += 1.0
                new_dict = dict()
                new_dict['PowerSquareDist'] = y_sum_square_dist / y_count
                new_dict['PowerAbsDist'] = y_sum_abs_dist / y_count
                # Just in case it will become unstable due to too few neighbors
                new_dict['PowerSquareDistSum'] = y_sum_square_dist
                new_dict['PowerSquareDistCount'] = y_count
                for ndk in new_dict.keys():
                    lion_power_plot_data[key][ndk] = new_dict[ndk]

                with open(lion_power_plot_data_file, 'wb') as f:
                    pickle.dump(lion_power_plot_data, f)

            print(key, lion_power_plot_data[key])

    end_time = datetime.datetime.now()
    logging.info("LION power experiment ended: %s", end_time)
    logging.info("LION power experiment duration: %s", end_time-start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()



#
# distance_matrix = distance.squareform(distance.pdist(X_mnist))
# np.fill_diagonal(distance_matrix, np.inf)  # We are not interested in distance to itself
# nn_x_distance = np.min(distance_matrix, axis=1)  # Any axis will do
# radius_x = dict()
# for p in lion_percentile_options:
#     radius_x[p] = np.percentile(nn_x_distance, p)
# print(radius_x)
#
# skip_accuracy = False
# recalculate_performance = True
#
# if os.path.isfile(lion_power_plot_data_file) and not regenerate_lion_accuracy:
#     with open(lion_power_plot_data_file, 'rb') as f:
#         lion_power_plot_data = pickle.load(f)
#
# for perc in lion_percentile_options:
#     for p in lion_power_options:
#         print("Processing percentile and power", p, perc)
#         key = str(perc) + ";" + "%.3f" % (p)
#         print(key)
#         if key not in lion_power_plot_data:
#             lion_power_plot_data[key] = dict()
#
#         if 'Accuracy' not in lion_power_plot_data[key] and not skip_accuracy:
#             print("Accuracy not found for power ", p, " percentile", perc, "\tCalculating...")
#             interpolator = dTSNE_mnist.generate_lion_tsne_embedder(verbose=0, random_state=0, function_kwargs={
#                 'radius_x_percentile': perc,
#                 'power': p})
#
#             per_sample_accuracy = np.zeros((len(picked_neighbors),))
#             for i in range(len(picked_neighbors)):
#                 # if i%100==0:
#                 #    print("\tPower: ",p,"Processing:",i)
#                 expected_label = picked_neighbor_labels[i]
#                 result = interpolator(picked_neighbors[i], verbose=0)
#                 nn_indices = get_nearest_neighbors_in_y(result, n=accuracy_nn)
#                 obtained_labels = labels_mnist[nn_indices]
#                 per_sample_accuracy[i] = sum(obtained_labels == expected_label) / len(obtained_labels)
#             cur_acc = np.mean(per_sample_accuracy)
#             # print('================= ',p,perc, cur_acc)
#             lion_power_plot_data[key]['Accuracy'] = cur_acc
#             with open(lion_power_plot_data_file, 'wb') as f:
#                 pickle.dump(lion_power_plot_data, f)
#
#         if 'PowerSquareDist' not in lion_power_plot_data[key] or recalculate_performance:
#             print("Power performance not found for power ", p, " percentile", perc, "\tCalculating...")
#
#             y_sum_square_dist = 0.0
#             y_sum_abs_dist = 0.0
#             y_count = 0.0
#             for i in range(len(X_mnist)):
#                 distances = distance_matrix[i, :].copy()
#                 distances[i] = np.inf  # Not interested in distance to itself
#                 # Step 1. Find nearest neighbors in the neighborhood.
#                 neighbor_indices = np.where(distances <= radius_x[perc])[0]
#                 num_neighbors = len(neighbor_indices)
#                 if num_neighbors >= 2:  # Below 2? Cannot interpolate
#                     # We are good
#                     weights = 1 / distances[neighbor_indices] ** p
#                     weights = weights / np.sum(weights)
#                     cur_y_result = weights.dot(Y_mnist[neighbor_indices, :])
#                     y_sum_square_dist += np.sum(cur_y_result - Y_mnist[i, :]) ** 2
#                     y_sum_abs_dist += np.sqrt(np.sum(cur_y_result - Y_mnist[i, :]) ** 2)
#                     y_count += 1.0
#             new_dict = dict()
#             new_dict['PowerSquareDist'] = y_sum_square_dist / y_count
#             new_dict['PowerAbsDist'] = y_sum_abs_dist / y_count
#             # Just in case it will become unstable due to too few neighbors
#             new_dict['PowerSquareDistSum'] = y_sum_square_dist
#             new_dict['PowerSquareDistCount'] = y_count
#             for ndk in new_dict.keys():
#                 lion_power_plot_data[key][ndk] = new_dict[ndk]
#
#             with open(lion_power_plot_data_file, 'wb') as f:
#                 pickle.dump(lion_power_plot_data, f)
#
#         print(key, lion_power_plot_data[key])







# # CAREFUL: It is not just a plot, it also searches optimal power parameter.
# lion_optimal_power = dict()
#
# font_properties = FontProperties()
# font_properties.set_family('serif')
# font_properties.set_name('Times New Roman')
# font_properties.set_size(8)
#
# legend_list = list()
# f, ax = plt.subplots()
# f.set_dpi(300)
# f.set_size_inches(3.3, 3.3)
# power_perc_combos = lion_power_plot_data.keys()
# # print(power_perc_combos)
# all_percentages = set([int(i.split(";")[0]) for i in power_perc_combos])
# x = sorted(set([float(i.split(";")[1]) for i in power_perc_combos]))
# color_dict = {90: 'blue', 95: 'green', 99: 'red', 100: 'cyan'}
# legend_list = list()
# legend_lines = list()
# # print(all_percentages)
# for perc in all_percentages:
#     y = list()
#     for cur_power in x:
#         key = str(perc) + ";%.3f" % (cur_power)
#         # print(cur_power, perc, lion_power_plot_data[key])
#         y.append(lion_power_plot_data[key]['PowerSquareDist'])
#     h, = plt.plot(x, y, c=color_dict[perc])
#     # print(perc, x[np.argmin(y)])
#     lion_optimal_power[perc] = x[np.argmin(y)]
#     legend_lines.append(h)
#     legend_list.append("$r_x$: " + str(perc) + " NN percentile")
#
# EPS = 1e-5
# y = list()
# x_global = list()
# for cur_power in x:
#     closest_power = [i for i in global_idw_power_performance_abs if np.abs(i - cur_power) < EPS]
#     if len(closest_power) > 0:
#         x_global.append(cur_power)
#         y.append(global_idw_power_performance[closest_power[0]])
# # print("Non-Local", x_global[np.argmin(y)])
# idw_optimal_power = x_global[np.argmin(y)]
# h, = plt.plot(x_global, y, c='purple', linestyle=":")
# legend_lines.append(h)
# legend_list.append("Non-local IDW")
# # plt.title("IDW - Accuracy vs Power") # We'd better use figure caption
# # ax.legend([h1,h2,h3,h4,h5,h6], ["Closest Training Set Image"]+idw_method_list)
# # h = plt.axhline(y=baseline_accuracy, c = 'black', linestyle='--')
# l = plt.legend(legend_lines, legend_list, bbox_to_anchor=[1.00, -0.15], ncol=2, prop=font_properties)
# plt.xlabel("LION-tSNE: Power", fontproperties=font_properties)
# plt.ylabel("Average Square Distance", fontproperties=font_properties)
# plt.ylim([0, 200])
# plt.xlim([0, 50])
# # f.tight_layout()
#
# ax = plt.gca()
# for label in ax.get_xticklabels():
#     label.set_fontproperties(font_properties)
#
# for label in ax.get_yticklabels():
#     label.set_fontproperties(font_properties)
#
# plt.tight_layout(rect=[-0.035, -0.045, 1.044, 1.042])
#
# plt.savefig("Figures/LION-power-training-set.png")
# plt.show(f)