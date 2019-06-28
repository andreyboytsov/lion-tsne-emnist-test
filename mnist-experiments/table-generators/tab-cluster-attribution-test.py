import lion_tsne
import generate_data
import settings
import pickle
import numpy as np

import exp_cluster_attr_test_RBF_IDW_LION
import exp_cluter_attr_test_NN
import exp_cluster_attr_test_kernelized
import exp_lion_power_performance
import exp_cluster_attr_test_GD

parameters = settings.parameters
Y_mnist = generate_data.load_y_mnist(parameters=parameters)
picked_indices = generate_data.load_nearest_training_indices(parameters=parameters)
picked_indices_y_mnist = Y_mnist[picked_indices,:]
X_mnist_raw = generate_data.load_x_mnist_raw(parameters=parameters)
dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
baseline_accuracy = generate_data.get_baseline_accuracy(parameters=parameters)

lion_power_plot_data_file = exp_lion_power_performance.generate_lion_power_plot_filename(parameters=parameters)

with open(lion_power_plot_data_file, 'rb') as f:
    _, _, lion_optimal_power = pickle.load(f)

cluster_results_file = exp_cluster_attr_test_RBF_IDW_LION.generate_cluster_results_filename(parameters)
with open(cluster_results_file, "rb") as f:
    all_RBF_IDW_LION_results = pickle.load(f)

nn_results_file = exp_cluter_attr_test_NN.generate_cluster_results_filename(parameters)
with open(nn_results_file, 'rb') as f:
        nn_method_results, nn_models_orig, nn_method_list = pickle.load(f)

accuracy_multiquadric = all_RBF_IDW_LION_results["RBF-multiquadric"]['Accuracy']
accuracy_gaussian = all_RBF_IDW_LION_results["RBF-gaussian"]['Accuracy']
accuracy_linear = all_RBF_IDW_LION_results["RBF-linear"]['Accuracy']
accuracy_cubic = all_RBF_IDW_LION_results["RBF-cubic"]['Accuracy']
accuracy_quintic = all_RBF_IDW_LION_results["RBF-quintic"]['Accuracy']
accuracy_inverse = all_RBF_IDW_LION_results["RBF-inverse"]['Accuracy']
accuracy_thin_plate = all_RBF_IDW_LION_results["RBF-thin-plate"]['Accuracy']

kl_multiquadric = all_RBF_IDW_LION_results["RBF-multiquadric"]["KL-Divergence"]
kl_gaussian = all_RBF_IDW_LION_results["RBF-gaussian"]["KL-Divergence"]
kl_linear = all_RBF_IDW_LION_results["RBF-linear"]["KL-Divergence"]
kl_cubic = all_RBF_IDW_LION_results["RBF-cubic"]["KL-Divergence"]
kl_quintic = all_RBF_IDW_LION_results["RBF-quintic"]["KL-Divergence"]
kl_inverse = all_RBF_IDW_LION_results["RBF-inverse"]["KL-Divergence"]
kl_thin_plate = all_RBF_IDW_LION_results["RBF-thin-plate"]["KL-Divergence"]

dist_multiquadric = all_RBF_IDW_LION_results["RBF-multiquadric"]["DistancePercentile"]
dist_gaussian = all_RBF_IDW_LION_results["RBF-gaussian"]["DistancePercentile"]
dist_linear = all_RBF_IDW_LION_results["RBF-linear"]["DistancePercentile"]
dist_cubic = all_RBF_IDW_LION_results["RBF-cubic"]["DistancePercentile"]
dist_quintic = all_RBF_IDW_LION_results["RBF-quintic"]["DistancePercentile"]
dist_inverse = all_RBF_IDW_LION_results["RBF-inverse"]["DistancePercentile"]
dist_thin_plate = all_RBF_IDW_LION_results["RBF-thin-plate"]["DistancePercentile"]

rbf_method_list = ['RBF - Multiquadric', 'RBF - Gaussian',
        'RBF - Inverse Multiquadric', 'RBF - Linear', 'RBF - Cubic', 'RBF - Quintic',
        'RBF - Thin Plate']

rbf_accuracy = [accuracy_multiquadric, accuracy_gaussian, accuracy_inverse,
                      accuracy_linear, accuracy_cubic, accuracy_quintic,
                      accuracy_thin_plate]
rbf_distance_percentiles = [dist_multiquadric, dist_gaussian, dist_inverse, dist_linear, dist_cubic, dist_quintic,
                      dist_thin_plate]
rbf_avg_kl = [kl_multiquadric, kl_gaussian, kl_inverse, kl_linear, kl_cubic, kl_quintic, kl_thin_plate]


# =============================================================================================================
keys_copy = all_RBF_IDW_LION_results.keys()
keys_copy -= {"IDW-1","IDW-10","IDW-20","IDW-40"}
idw_optimal_name = [i for i in keys_copy if i.startswith("IDW")][0]
print(idw_optimal_name)
accuracy_idw1 = all_RBF_IDW_LION_results['IDW-1']['Accuracy']
accuracy_idw10 = all_RBF_IDW_LION_results['IDW-10']['Accuracy']
accuracy_idw20 = all_RBF_IDW_LION_results['IDW-20']['Accuracy']
accuracy_idw40 = all_RBF_IDW_LION_results['IDW-40']['Accuracy']
accuracy_idw_optimal = all_RBF_IDW_LION_results[idw_optimal_name]['Accuracy']

idw_method_list = ["IDW - Power 1","IDW - Power 10", "IDW - Power 20",
    "IDW - Power "+idw_optimal_name[-4:], "IDW - Power 40"]

lion90_name = [i for i in all_RBF_IDW_LION_results.keys() if i.startswith('LION-90')][0]
accuracy__lion90 = all_RBF_IDW_LION_results[lion90_name]['Accuracy']
lion95_name = [i for i in all_RBF_IDW_LION_results.keys() if i.startswith('LION-95')][0]
accuracy__lion95 = all_RBF_IDW_LION_results[lion95_name]['Accuracy']
lion99_name = [i for i in all_RBF_IDW_LION_results.keys() if i.startswith('LION-99')][0]
accuracy__lion99 = all_RBF_IDW_LION_results[lion99_name]['Accuracy']
lion100_name = [i for i in all_RBF_IDW_LION_results.keys() if i.startswith('LION-100')][0]
accuracy__lion100 = all_RBF_IDW_LION_results[lion100_name]['Accuracy']

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

idw_method_results = [picked_neighbors_y_idw1, picked_neighbors_y_idw10, picked_neighbors_y_idw20,
                      picked_neighbors_y_idw_optimal, picked_neighbors_y_idw40]

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
# ==================== Building the table

s = ""

s += '''\\begin{table} \small\sf\centering \caption{Cluster attribution test: methods comparison}  \label{tab_cluster_methods_comparison}
    \\begin{tabular}{ m{0.18\\textwidth}  m{0.07\\textwidth}  m{0.07\\textwidth}  m{0.06\\textwidth} }
        \\toprule
            \\textbf{Method}
            & \\textbf{Accuracy}
            & \\textbf{Distance Percentile}
            & \\textbf{KL Divergence}
        \\\\ \\midrule'''

initial_kl_divergence, _ = lion_tsne.kl_divergence_and_gradient(y=dTSNE_mnist.Y, p_matrix=dTSNE_mnist.P_matrix)

print('\t\\textbf{Baseline} & %.2f\\%% & - & %.5f' % (baseline_accuracy * 100, initial_kl_divergence))
print('\t\\\\')

print('\\multicolumn{4}{c}{\\textbf{RBF Interpolation}}')
print('\t\\\\')

for j in range(len(rbf_method_list)):
    print(
        '\t %s & %.2f\\%% & %.3f & %.5f' % (rbf_method_list[j][6:], rbf_accuracy[j] * 100, rbf_distance_percentiles[j],
                                            rbf_avg_kl[j]))
    print('\t\\\\')

print('\\multicolumn{4}{c}{\\textbf{IDW Interpolation}}')
print('\t\\\\')

for j in range(len(idw_method_results)):
    print(
        '\t %s & %.2f\\%% & %.3f & %.5f' % (idw_method_list[j][6:], idw_accuracy[j] * 100, idw_distance_percentiles[j],
                                            idw_avg_kl[j]))
    print('\t\\\\')

print('\\multicolumn{4}{c}{\\textbf{Gradient Descent}}')
print('\t\\\\')

for j in range(len(gd_method_results)):
    print('\t %s & %.2f\\%% & %.3f & %.5f' % (gd_method_list[j], gd_accuracy[j] * 100, gd_distance_percentiles[j],
                                              gd_avg_kl[j]))
    print('\t\\\\')

print('\\multicolumn{4}{c}{\\textbf{Neural Networks}}')
print('\t\\\\')

for j in range(len(nn_method_results)):
    print('\t %s & %.2f\\%% & %.3f & %.5f' % (
    nn_method_list[j][5:], nn_accuracy[j] * 100, nn_distance_percentiles[j], nn_avg_kl[j]))
    print('\t\\\\')

print('\\multicolumn{4}{c}{\\textbf{Kernelized tSNE}}')
print('\t\\\\')

for j in range(len(kernelized_tsne_method_results)):
    print('\t %s & %.2f\\%% & %.3f & %.5f' % (kernelized_tsne_method_list[j][12:], kernelized_tsne_accuracy[j] * 100,
                                              kernelized_tsne_distance_percentiles[j], kernelized_tsne_avg_kl[j]))
    print('\t\\\\')

print('\\multicolumn{4}{c}{\\textbf{LION tSNE}}')
print('\t\\\\')

for j in range(len(lion_method_results)):
    print('\t %s & \\textbf{%.2f\\%%} & \\textbf{%.3f} & \\textbf{%.5f}' %
          (';'.join(lion_method_list[j].split(";")[1:]), lion_accuracy[j] * 100, lion_distance_percentiles[j],
           lion_avg_kl[j]))
    print('\t\\\\')

print('''
    \\bottomrule
    \\end{tabular}
\\end{table}
''')