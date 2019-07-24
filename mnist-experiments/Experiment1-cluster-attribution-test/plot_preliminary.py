import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
import generate_data
import settings
import exp_cluster_attr_test_IDW_RBF
import exp_cluster_attr_test_LION
import cluster_lion_RBF_IDW_commons
import exp_cluster_attr_test_NN
import exp_cluster_attr_test_kernelized
import exp_lion_power_performance
import exp_cluster_attr_test_GD
import pickle

# ========================= LOADING ALL THE DATA

shown_indices = 10
parameters = settings.parameters
Y_mnist = generate_data.load_y_mnist(parameters=parameters)
picked_indices = generate_data.load_nearest_training_indices(parameters=parameters)
picked_indices_y_mnist = Y_mnist[picked_indices,:]

lion_power_plot_data_file = exp_lion_power_performance.generate_lion_power_plot_filename(parameters=parameters)

with open(lion_power_plot_data_file, 'rb') as f:
    _, _, lion_optimal_power = pickle.load(f)

#idw_rbf_cluster_results_file = cluster_lion_RBF_IDW_commons.generate_cluster_results_filename(
#    exp_cluster_attr_test_IDW_RBF.cluster_results_file_prefix, parameters)
#with open(idw_rbf_cluster_results_file, "rb") as f:
#    all_RBF_IDW_results = pickle.load(f)

lion_cluster_results_file = cluster_lion_RBF_IDW_commons.generate_cluster_results_filename(
    exp_cluster_attr_test_LION.cluster_results_file_prefix, parameters)
with open(lion_cluster_results_file, "rb") as f:
    all_LION_results = pickle.load(f)


print(all_LION_results.keys())

#picked_neighbors_y_multiquadric = all_RBF_IDW_results["RBF-multiquadric"]['EmbeddedPoints']
#picked_neighbors_y_gaussian = all_RBF_IDW_results["RBF-gaussian"]['EmbeddedPoints']
#picked_neighbors_y_linear = all_RBF_IDW_results["RBF-linear"]['EmbeddedPoints']
#picked_neighbors_y_cubic = all_RBF_IDW_results["RBF-cubic"]['EmbeddedPoints']
#picked_neighbors_y_quintic = all_RBF_IDW_results["RBF-quintic"]['EmbeddedPoints']
#picked_neighbors_y_inverse = all_RBF_IDW_results["RBF-inverse"]['EmbeddedPoints']
#picked_neighbors_y_thin_plate = all_RBF_IDW_results["RBF-thin-plate"]['EmbeddedPoints']

rbf_method_list = ["RBF - Multiquadric","RBF - Gaussian",
                        "RBF - Inverse Multiquadric","RBF - Linear",'RBF - Cubic','RBF - Quintic',
                        'RBF - Thin Plate']

#keys_copy = all_RBF_IDW_results.keys()
#keys_copy -= {"IDW-1","IDW-10","IDW-20","IDW-40"}
#idw_optimal_name = [i for i in keys_copy if i.startswith("IDW")][0]
#print(idw_optimal_name)
#picked_neighbors_y_idw1 = all_RBF_IDW_results['IDW-1']['EmbeddedPoints']
#picked_neighbors_y_idw10 = all_RBF_IDW_results['IDW-10']['EmbeddedPoints']
#picked_neighbors_y_idw20 = all_RBF_IDW_results['IDW-20']['EmbeddedPoints']
#picked_neighbors_y_idw40 = all_RBF_IDW_results['IDW-40']['EmbeddedPoints']
#picked_neighbors_y_idw_optimal = all_RBF_IDW_results[idw_optimal_name]['EmbeddedPoints']

#idw_method_list = ["IDW - Power 1","IDW - Power 10", "IDW - Power 20",
#    "IDW - Power "+idw_optimal_name[-4:], "IDW - Power 40"]

lion90_name = [i for i in all_LION_results.keys() if i.startswith('LION-90')][0]
picked_neighbors_y_lion90 = all_LION_results[lion90_name]['EmbeddedPoints']
lion95_name = [i for i in all_LION_results.keys() if i.startswith('LION-95')][0]
picked_neighbors_y_lion95 = all_LION_results[lion95_name]['EmbeddedPoints']
lion99_name = [i for i in all_LION_results.keys() if i.startswith('LION-99')][0]
picked_neighbors_y_lion99 = all_LION_results[lion99_name]['EmbeddedPoints']
lion100_name = [i for i in all_LION_results.keys() if i.startswith('LION-100')][0]
picked_neighbors_y_lion100 = all_LION_results[lion100_name]['EmbeddedPoints']

lion_method_list = ["LION; $r_x$ at %dth perc.; $p$=%.1f"%(i, lion_optimal_power[i])
                    for i in sorted(lion_optimal_power)]

#kernelized_results_file = exp_cluster_attr_test_kernelized.generate_cluster_results_filename(parameters)
#with open(kernelized_results_file, 'rb') as f:
#    kernelized_detailed_tsne_method_results, kernelized_detailed_tsne_accuracy, \
#            kernelized_detailed_tsne_method_list = pickle.load(f)
#ind = [4,24,49]
#kernelized_tsne_method_list = [kernelized_detailed_tsne_method_list[i][:10]+kernelized_detailed_tsne_method_list[i][-8:]
#                               for i in ind]
#kernelized_tsne_method_results = [kernelized_detailed_tsne_method_results[i] for i in ind]


#gd_results_file = exp_cluster_attr_test_GD.generate_cluster_results_filename(parameters=parameters)
#with open(gd_results_file, 'rb') as f:
#    (picked_neighbors_y_gd_transformed, picked_neighbors_y_gd_variance_recalc_transformed,
#     picked_neighbors_y_gd_transformed_random, picked_neighbors_y_gd_variance_recalc_transformed_random,
#     picked_neighbors_y_gd_early_exagg_transformed_random,
#     picked_neighbors_y_gd_early_exagg_transformed,
#     picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random,
#     picked_random_starting_positions,
#     picked_neighbors_y_gd_variance_recalc_early_exagg_transformed, covered_samples) = pickle.load(f)

print("DATA LOADED")

# ====================== BUILDING ALL THE PLOTS

legend_list = list()
lion_X = 1
lion_Y = 2
rbf_X = 0
rbf_Y = 0
idw_X = 1
idw_Y = 0
gd_X = 0
gd_Y = 1
nn_X = 1
nn_Y = 1
ktsne_X = 0
ktsne_Y = 2


# f, ax = plt.subplots(2,3)
# plt.gcf().set_size_inches(10,10)
plt.figure(dpi=300)
plt.gcf().set_size_inches(6.8, 6.8)
chosen_indices = list(range(shown_indices))
# chosen_indices = [3]

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

n_row = 3
n_col = 2
lw = 0.8
point_size_interest = 15
point_size_gray = 10
cross_size = 20

gs = gridspec.GridSpec(n_row, n_col)
gs.update(wspace=0, hspace=0)  # set the spacing between axes.

ax = list()
gs_ind = 0
for i in range(n_row):
    ax.append(list())
    for j in range(n_col):
        ax[i].append(plt.subplot(gs[gs_ind]))
        gs_ind += 1

for i in range(len(ax)):
    for j in range(len(ax[i])):
        # print(i,j)
        ax[i][j].axes.get_xaxis().set_visible(False)
        ax[i][j].axes.get_yaxis().set_visible(False)


# ====================================== LION =====================================

for l in range(shown_indices):
    if l in chosen_indices:
        ax[lion_Y][lion_X].plot([picked_indices_y_mnist[l, 0], picked_neighbors_y_lion90[l, 0]],
                                [picked_indices_y_mnist[l, 1], picked_neighbors_y_lion90[l, 1]], c='black', label=None,
                                zorder=2,
                                linewidth=lw)
        ax[lion_Y][lion_X].plot([picked_indices_y_mnist[l, 0], picked_neighbors_y_lion95[l, 0]],
                                [picked_indices_y_mnist[l, 1], picked_neighbors_y_lion95[l, 1]], c='black', label=None,
                                zorder=2,
                                linewidth=lw)
        ax[lion_Y][lion_X].plot([picked_indices_y_mnist[l, 0], picked_neighbors_y_lion99[l, 0]],
                                [picked_indices_y_mnist[l, 1], picked_neighbors_y_lion99[l, 1]], c='black', label=None,
                                zorder=2,
                                linewidth=lw)
        ax[lion_Y][lion_X].plot([picked_indices_y_mnist[l, 0], picked_neighbors_y_lion100[l, 0]],
                                [picked_indices_y_mnist[l, 1], picked_neighbors_y_lion100[l, 1]], c='black', label=None,
                                zorder=2,
                                linewidth=lw)

ax[lion_Y][lion_X].scatter(Y_mnist[:, 0], Y_mnist[:, 1], c='gray', zorder=1, label=None, marker='.', s=point_size_gray)
# legend_list.append(str(l))
h1 = ax[lion_Y][lion_X].scatter(picked_indices_y_mnist[:shown_indices, 0],
                                picked_indices_y_mnist[:shown_indices:, 1], c='red', marker='X', s=cross_size, zorder=3)
h2 = ax[lion_Y][lion_X].scatter(picked_neighbors_y_lion90[:shown_indices, 0],
                                picked_neighbors_y_lion90[:shown_indices, 1], c='red', marker='.', zorder=3, alpha=0.9,
                                s=point_size_interest)
h3 = ax[lion_Y][lion_X].scatter(picked_neighbors_y_lion95[:shown_indices, 0],
                                picked_neighbors_y_lion95[:shown_indices, 1], c='blue', marker='.', zorder=3, alpha=0.9,
                                s=point_size_interest)
h4 = ax[lion_Y][lion_X].scatter(picked_neighbors_y_lion99[:shown_indices, 0],
                                picked_neighbors_y_lion99[:shown_indices, 1], c='green', marker='.', zorder=3,
                                alpha=0.9,
                                s=point_size_interest)
h5 = ax[lion_Y][lion_X].scatter(picked_neighbors_y_lion100[:shown_indices, 0],
                                picked_neighbors_y_lion100[:shown_indices, 1], c='purple', marker='.', zorder=3,
                                alpha=0.9,
                                s=point_size_interest)

# ax[lion_Y][lion_X].legend([h1,h2,h3,h4,h5], ["Closest Training Set Image"]+lion_method_list, loc=4, fontsize = 13)
lion_legend_names = [i[6:] for i in lion_method_list]
# for i in range(len(lion_legend_names)):
#    lion_legend_names[i] = lion_legend_names[i].replace('perc','\nperc')
# print(lion_legend_names)
ax[lion_Y][lion_X].legend([h2, h3, h4, h5], lion_legend_names, ncol=1, prop=font_properties, borderpad=0.1,
                          handlelength=2,
                          columnspacing=0, loc=1, handletextpad=-0.7, frameon=True)

plt.tight_layout()
plt.subplots_adjust(wspace=None, hspace=None, left=0.003, right=0.997, top=0.997, bottom=0.003)

ax[rbf_Y][rbf_X].text(-140, 120, "(a) RBF interpolation", fontsize=10)
ax[idw_Y][idw_X].text(-140, 120, "(b) IDW interpolation", fontsize=10)
ax[gd_Y][gd_X].text(-140, 120, "(c) Gradient descent", fontsize=10)
ax[nn_Y][nn_X].text(-140, 120, "(d) Neural networks", fontsize=10)
ax[ktsne_Y][ktsne_X].text(-140, 120, "(e) Kernelized tSNE", fontsize=10)
ax[lion_Y][lion_X].text(-140, 120, "(f) LION tSNE", fontsize=10)

plt.show()