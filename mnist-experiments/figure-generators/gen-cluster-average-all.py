import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
import generate_data
import settings
import cluster_lion_RBF_IDW_commons
import exp_cluster_attr_test_IDW_RBF
import exp_cluster_attr_test_LION
import exp_cluster_attr_test_NN
import exp_cluster_attr_test_kernelized
import exp_lion_power_performance
import exp_cluster_attr_test_GD
import pickle
import numpy as np

# ========================= LOADING ALL THE DATA

shown_indices = 10
illustration_nn = 10
parameters = settings.parameters
Y_mnist = generate_data.load_y_mnist(parameters=parameters)
picked_indices = generate_data.load_nearest_training_indices(parameters=parameters)
picked_indices_y_mnist = Y_mnist[picked_indices,:]
X_mnist_raw = generate_data.load_x_mnist_raw(parameters=parameters)

def get_nearest_neighbors_in_y(y, n=10):
    y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
    return np.argsort(y_distances)[:n]

lion_power_plot_data_file = exp_lion_power_performance.generate_lion_power_plot_filename(parameters=parameters)

with open(lion_power_plot_data_file, 'rb') as f:
    _, _, lion_optimal_power = pickle.load(f)

idw_rbf_cluster_results_file = cluster_lion_RBF_IDW_commons.generate_cluster_results_filename(
    exp_cluster_attr_test_IDW_RBF.cluster_results_file_prefix, parameters)
with open(idw_rbf_cluster_results_file, "rb") as f:
    all_RBF_IDW_results = pickle.load(f)

lion_cluster_results_file = cluster_lion_RBF_IDW_commons.generate_cluster_results_filename(
    exp_cluster_attr_test_LION.cluster_results_file_prefix, parameters)
with open(lion_cluster_results_file, "rb") as f:
    all_LION_results = pickle.load(f)

nn_results_file = exp_cluster_attr_test_NN.generate_cluster_results_filename(parameters)
with open(nn_results_file, 'rb') as f:
        nn_method_results, nn_models_orig, nn_method_list = pickle.load(f)

picked_neighbors_y_multiquadric = all_RBF_IDW_results["RBF-multiquadric"]['EmbeddedPoints']
picked_neighbors_y_gaussian = all_RBF_IDW_results["RBF-gaussian"]['EmbeddedPoints']
picked_neighbors_y_linear = all_RBF_IDW_results["RBF-linear"]['EmbeddedPoints']
picked_neighbors_y_cubic = all_RBF_IDW_results["RBF-cubic"]['EmbeddedPoints']
picked_neighbors_y_quintic = all_RBF_IDW_results["RBF-quintic"]['EmbeddedPoints']
picked_neighbors_y_inverse = all_RBF_IDW_results["RBF-inverse"]['EmbeddedPoints']
picked_neighbors_y_thin_plate = all_RBF_IDW_results["RBF-thin-plate"]['EmbeddedPoints']

rbf_method_list = ["RBF - Multiquadric","RBF - Gaussian",
                        "RBF - Inverse Multiquadric","RBF - Linear",'RBF - Cubic','RBF - Quintic',
                        'RBF - Thin Plate']

keys_copy = all_RBF_IDW_results.keys()
keys_copy -= {"IDW-1","IDW-10","IDW-20","IDW-40"}
idw_optimal_name = [i for i in keys_copy if i.startswith("IDW")][0]
print(idw_optimal_name)
picked_neighbors_y_idw1 = all_RBF_IDW_results['IDW-1']['EmbeddedPoints']
picked_neighbors_y_idw20 = all_RBF_IDW_results['IDW-20']['EmbeddedPoints']
picked_neighbors_y_idw70 = all_RBF_IDW_results['IDW-70']['EmbeddedPoints']
picked_neighbors_y_idw_optimal = all_RBF_IDW_results[idw_optimal_name]['EmbeddedPoints']

idw_method_list = ["IDW - Power 1","IDW - Power 20",
    "IDW - Power "+idw_optimal_name[-4:], "IDW - Power 70"]

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

kernelized_results_file = exp_cluster_attr_test_kernelized.generate_cluster_results_filename(parameters)
with open(kernelized_results_file, 'rb') as f:
    kernelized_detailed_tsne_method_results, kernelized_detailed_tsne_accuracy, \
            kernelized_detailed_tsne_method_list = pickle.load(f)
ind = [4,24,49]
kernelized_tsne_method_list = [kernelized_detailed_tsne_method_list[i][:10]+kernelized_detailed_tsne_method_list[i][-8:]
                               for i in ind]
kernelized_tsne_method_results = [kernelized_detailed_tsne_method_results[i] for i in ind]


gd_results_file = exp_cluster_attr_test_GD.generate_cluster_results_filename(parameters=parameters)
with open(gd_results_file, 'rb') as f:
    (picked_neighbors_y_gd_transformed, picked_neighbors_y_gd_variance_recalc_transformed,
     picked_neighbors_y_gd_transformed_random, picked_neighbors_y_gd_variance_recalc_transformed_random,
     picked_neighbors_y_gd_early_exagg_transformed_random,
     picked_neighbors_y_gd_early_exagg_transformed,
     picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random,
     picked_random_starting_positions,
     picked_neighbors_y_gd_variance_recalc_early_exagg_transformed, covered_samples) = pickle.load(f)

print("DATA LOADED")

rbf_method_results = [picked_neighbors_y_multiquadric, picked_neighbors_y_gaussian, picked_neighbors_y_inverse,
                      picked_neighbors_y_linear, picked_neighbors_y_cubic, picked_neighbors_y_quintic,
                      picked_neighbors_y_thin_plate]
idw_method_results = [picked_neighbors_y_idw1, picked_neighbors_y_idw20,
                      picked_neighbors_y_idw_optimal, picked_neighbors_y_idw70]

gd_method_list = [r'Closest $Y_{init}$',
              r'Random $Y_{init}$',
              r'Closest $Y_{init}$; new $\sigma$',
              r'Random $Y_{init}$; new $\sigma$',
              r'Closest $Y_{init}$; EE',
              r'Random $Y_{init}$; EE',
              r'Closest $Y_{init}$; new $\sigma$; EE',
              r'Random $Y_{init}$; new $\sigma$; EE']

gd_method_results = [
    picked_neighbors_y_gd_transformed,
    picked_neighbors_y_gd_transformed_random,
    picked_neighbors_y_gd_variance_recalc_transformed,
    picked_neighbors_y_gd_variance_recalc_transformed_random,
    picked_neighbors_y_gd_early_exagg_transformed,
    picked_neighbors_y_gd_early_exagg_transformed_random,
    picked_neighbors_y_gd_variance_recalc_early_exagg_transformed,
    picked_neighbors_y_gd_variance_recalc_early_exagg_transformed_random,
]

lion_method_results = [picked_neighbors_y_lion90, picked_neighbors_y_lion95, picked_neighbors_y_lion99,
                       picked_neighbors_y_lion100]
# ==================== Building the plots

width = shown_indices
height = len(rbf_method_results) + len(idw_method_results) + len(gd_method_results) + len(nn_method_results) + \
         len(kernelized_tsne_method_results) + len(lion_method_results)

gs = gridspec.GridSpec(height + 2, width + 1, width_ratios=[6.1] + [1] * width)

width = shown_indices  # total number to show
start_index = 0

f, ax_total = plt.subplots(dpi=300)

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

# First row - number of examples
plt.subplot(gs[0]).set_axis_off()
plt.subplot(gs[width + 1]).set_axis_off()
plt.subplot(gs[0]).text(text="№   ", x=1.0, y=0.4, s=11, ha='right', fontproperties=font_properties)
plt.subplot(gs[width + 1]).text(text="Original Image  ", x=1.0, y=0.4, s=11, ha='right', fontproperties=font_properties)

# f.set_size_inches(7,13)
f.set_size_inches(3.3, 3.3 * 13 / 7)
f.subplots_adjust()

for j in range(len(rbf_method_results)):
    sp = plt.subplot(gs[(j + 2) * (width + 1)])
    # Shorten one of names
    sp.text(text=(rbf_method_list[j] + '  ' if j != 2 else 'RBF - Inv. Mutliq. '), x=1.0, y=0.4, s=11, ha='right',
            fontproperties=font_properties)
    sp.set_axis_off()

# It is just a plot, let's make it simple and use 2 loop rather than vectorization
for i in range(width):
    orig_sp = plt.subplot(gs[width + i + 2])
    orig_sp.imshow(X_mnist_raw[picked_indices[i], :].reshape(28, 28), cmap='gray_r')
    orig_sp.axes.get_xaxis().set_visible(False)
    orig_sp.axes.get_yaxis().set_visible(False)
    num_sp = plt.subplot(gs[i + 1])
    num_sp.text(text=str(i + 1), x=0.5, y=0.5, s=11, ha='center', va='center', fontproperties=font_properties)
    num_sp.set_axis_off()
    for j in range(len(rbf_method_results)):
        sp = plt.subplot(gs[(j + 2) * (width + 1) + i + 1])
        nn_indices = get_nearest_neighbors_in_y(rbf_method_results[j][i, :], n=illustration_nn)
        average_image = np.mean(X_mnist_raw[nn_indices, :], axis=0).reshape(28, 28)
        sp.imshow(average_image, cmap='gray_r')
        sp.axes.get_xaxis().set_visible(False)
        sp.axes.get_yaxis().set_visible(False)

for j in range(len(idw_method_results)):
    sp = plt.subplot(gs[(j + len(rbf_method_results) + 2) * (width + 1)])
    sp.text(text=idw_method_list[j] + '  ', x=1.0, y=0.4, s=11, ha='right', fontproperties=font_properties)
    sp.set_axis_off()

# It is just a plot, let's make it simple and use 2 loop rather than vectorization
for i in range(width):
    for j in range(len(idw_method_results)):
        sp = plt.subplot(gs[(j + len(rbf_method_results) + 2) * (width + 1) + i + 1])
        nn_indices = get_nearest_neighbors_in_y(idw_method_results[j][i, :], n=illustration_nn)
        average_image = np.mean(X_mnist_raw[nn_indices, :], axis=0).reshape(28, 28)
        sp.imshow(average_image, cmap='gray_r')
        sp.axes.get_xaxis().set_visible(False)
        sp.axes.get_yaxis().set_visible(False)

for j in range(len(gd_method_results)):
    sp = plt.subplot(gs[(j + len(rbf_method_results) + len(idw_method_results) + 2) * (width + 1)])
    sp.text(text='GD - ' + gd_method_list[j] + '  ', x=1.0, y=0.4, s=11, ha='right', fontproperties=font_properties)
    sp.set_axis_off()

# It is just a plot, let's make it simple and use 2 loop rather than vectorization
for i in range(width):
    for j in range(len(gd_method_results)):
        sp = plt.subplot(gs[(j + len(rbf_method_results) + len(idw_method_results) + 2) * (width + 1) + i + 1])
        nn_indices = get_nearest_neighbors_in_y(gd_method_results[j][i, :], n=illustration_nn)
        average_image = np.mean(X_mnist_raw[nn_indices, :], axis=0).reshape(28, 28)
        sp.imshow(average_image, cmap='gray_r')
        sp.axes.get_xaxis().set_visible(False)
        sp.axes.get_yaxis().set_visible(False)

for j in range(len(nn_method_results)):
    sp = plt.subplot(
        gs[(j + len(rbf_method_results) + len(idw_method_results) + +len(gd_method_results) + 2) * (width + 1)])
    sp.text(text='NN - ' + nn_method_list[j][5:] + '  ', x=1.0, y=0.4, s=11, ha='right', fontproperties=font_properties)
    sp.set_axis_off()

# It is just a plot, let's make it simple and use 2 loop rather than vectorization
for i in range(width):
    for j in range(len(nn_method_results)):
        def get_nn_nearest_neighbors_in_y(y, n=10):
            y_distances = np.sum((nn_models_orig[j] - y) ** 2, axis=1)
            return np.argsort(y_distances)[:n]


        sp = plt.subplot(gs[(j + len(rbf_method_results) + len(idw_method_results) + +len(gd_method_results) + 2) * (
                    width + 1) + i + 1])
        nn_indices = get_nn_nearest_neighbors_in_y(nn_method_results[j][i, :], n=illustration_nn)
        average_image = np.mean(X_mnist_raw[nn_indices, :], axis=0).reshape(28, 28)
        sp.imshow(average_image, cmap='gray_r')
        sp.axes.get_xaxis().set_visible(False)
        sp.axes.get_yaxis().set_visible(False)

for j in range(len(kernelized_tsne_method_results)):
    sp = plt.subplot(gs[(j + len(rbf_method_results) + len(idw_method_results) + +len(gd_method_results) + len(
        nn_method_results) + 2) * (width + 1)])
    sp.text(text=kernelized_tsne_method_list[j] + '  ', x=1.0, y=0.4, s=11, ha='right', fontproperties=font_properties)
    sp.set_axis_off()

# It is just a plot, let's make it simple and use 2 loop rather than vectorization
for i in range(width):
    for j in range(len(kernelized_tsne_method_results)):
        sp = plt.subplot(gs[(j + len(rbf_method_results) + len(idw_method_results) + +len(gd_method_results) + len(
            nn_method_results) + 2) * (width + 1) + i + 1])
        nn_indices = get_nearest_neighbors_in_y(kernelized_tsne_method_results[j][i, :], n=illustration_nn)
        average_image = np.mean(X_mnist_raw[nn_indices, :], axis=0).reshape(28, 28)
        sp.imshow(average_image, cmap='gray_r')
        sp.axes.get_xaxis().set_visible(False)
        sp.axes.get_yaxis().set_visible(False)

for j in range(len(lion_method_results)):
    sp = plt.subplot(gs[(j + len(rbf_method_results) + len(idw_method_results) + +len(gd_method_results) +
                         len(nn_method_results) + len(kernelized_tsne_method_results) + 2) * (width + 1)])
    # Shorten one of names
    sp.text(text='LION - ' + lion_method_list[j][15:] + '  ', x=1.0, y=0.4, s=11, ha='right',
            fontproperties=font_properties)
    sp.set_axis_off()

# It is just a plot, let's make it simple and use 2 loop rather than vectorization
for i in range(width):
    for j in range(len(lion_method_results)):
        sp = plt.subplot(gs[(j + len(rbf_method_results) + len(idw_method_results) + +len(gd_method_results) +
                             len(nn_method_results) + len(kernelized_tsne_method_results) + 2) * (width + 1) + i + 1])
        nn_indices = get_nearest_neighbors_in_y(lion_method_results[j][i, :], n=illustration_nn)
        average_image = np.mean(X_mnist_raw[nn_indices, :], axis=0).reshape(28, 28)
        sp.imshow(average_image, cmap='gray_r')
        sp.axes.get_xaxis().set_visible(False)
        sp.axes.get_yaxis().set_visible(False)

# gs.tight_layout(f)
gs.update(wspace=0.0, hspace=0.0)
f.subplots_adjust(left=0.12, right=0.995, top=0.995, bottom=0.005)

# plt.subplots_adjust(wspace=None, hspace=None)
plt.savefig("../figures/cluster-average-all.png")