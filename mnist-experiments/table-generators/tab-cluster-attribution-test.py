import lion_tsne
import generate_data
import settings
import pickle
import numpy as np

import exp_cluster_attr_test_RBF_IDW_LION
import exp_cluster_postprocess_NN
import exp_cluster_postprocess_Kernelized
import exp_lion_power_performance
import exp_cluster_postprocess_GD

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

accuracy_multiquadric = np.mean(all_RBF_IDW_LION_results["RBF-multiquadric"]['Accuracy'])
accuracy_gaussian = np.mean(all_RBF_IDW_LION_results["RBF-gaussian"]['Accuracy'])
accuracy_linear = np.mean(all_RBF_IDW_LION_results["RBF-linear"]['Accuracy'])
accuracy_cubic = np.mean(all_RBF_IDW_LION_results["RBF-cubic"]['Accuracy'])
accuracy_quintic = np.mean(all_RBF_IDW_LION_results["RBF-quintic"]['Accuracy'])
accuracy_inverse = np.mean(all_RBF_IDW_LION_results["RBF-inverse"]['Accuracy'])
accuracy_thin_plate = np.mean(all_RBF_IDW_LION_results["RBF-thin-plate"]['Accuracy'])

kl_multiquadric = np.mean(all_RBF_IDW_LION_results["RBF-multiquadric"]["KL-Divergence"])
kl_gaussian = np.mean(all_RBF_IDW_LION_results["RBF-gaussian"]["KL-Divergence"])
kl_linear = np.mean(all_RBF_IDW_LION_results["RBF-linear"]["KL-Divergence"])
kl_cubic = np.mean(all_RBF_IDW_LION_results["RBF-cubic"]["KL-Divergence"])
kl_quintic = np.mean(all_RBF_IDW_LION_results["RBF-quintic"]["KL-Divergence"])
kl_inverse = np.mean(all_RBF_IDW_LION_results["RBF-inverse"]["KL-Divergence"])
kl_thin_plate = np.mean(all_RBF_IDW_LION_results["RBF-thin-plate"]["KL-Divergence"])

dist_multiquadric = np.mean(all_RBF_IDW_LION_results["RBF-multiquadric"]["DistancePercentile"])
dist_gaussian = np.mean(all_RBF_IDW_LION_results["RBF-gaussian"]["DistancePercentile"])
dist_linear = np.mean(all_RBF_IDW_LION_results["RBF-linear"]["DistancePercentile"])
dist_cubic = np.mean(all_RBF_IDW_LION_results["RBF-cubic"]["DistancePercentile"])
dist_quintic = np.mean(all_RBF_IDW_LION_results["RBF-quintic"]["DistancePercentile"])
dist_inverse = np.mean(all_RBF_IDW_LION_results["RBF-inverse"]["DistancePercentile"])
dist_thin_plate = np.mean(all_RBF_IDW_LION_results["RBF-thin-plate"]["DistancePercentile"])

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
accuracy_idw1 = np.mean(all_RBF_IDW_LION_results['IDW-1']['Accuracy'])
accuracy_idw10 = np.mean(all_RBF_IDW_LION_results['IDW-10']['Accuracy'])
accuracy_idw20 = np.mean(all_RBF_IDW_LION_results['IDW-20']['Accuracy'])
accuracy_idw40 = np.mean(all_RBF_IDW_LION_results['IDW-40']['Accuracy'])
accuracy_idw_optimal = np.mean(all_RBF_IDW_LION_results[idw_optimal_name]['Accuracy'])

kl_idw1 = np.mean(all_RBF_IDW_LION_results['IDW-1']['KL-Divergence'])
kl_idw10 = np.mean(all_RBF_IDW_LION_results['IDW-10']['KL-Divergence'])
kl_idw20 = np.mean(all_RBF_IDW_LION_results['IDW-20']['KL-Divergence'])
kl_idw40 = np.mean(all_RBF_IDW_LION_results['IDW-40']['KL-Divergence'])
kl_idw_optimal = np.mean(all_RBF_IDW_LION_results[idw_optimal_name]['KL-Divergence'])

dist_idw1 = np.mean(all_RBF_IDW_LION_results['IDW-1']['DistancePercentile'])
dist_idw10 = np.mean(all_RBF_IDW_LION_results['IDW-10']['DistancePercentile'])
dist_idw20 = np.mean(all_RBF_IDW_LION_results['IDW-20']['DistancePercentile'])
dist_idw40 = np.mean(all_RBF_IDW_LION_results['IDW-40']['DistancePercentile'])
dist_idw_optimal = np.mean(all_RBF_IDW_LION_results[idw_optimal_name]['DistancePercentile'])

idw_method_list = ["IDW - Power 1","IDW - Power 10", "IDW - Power 20",
    "IDW - Power "+idw_optimal_name[-4:], "IDW - Power 40"]

idw_accuracy = [accuracy_idw1, accuracy_idw10, accuracy_idw20, accuracy_idw40,
                accuracy_idw_optimal]
idw_distance_percentiles = [dist_idw1, dist_idw10, dist_idw20, dist_idw40, dist_idw_optimal]
idw_avg_kl = [kl_idw1, kl_idw10, kl_idw20, kl_idw40, kl_idw_optimal]

lion90_name = [i for i in all_RBF_IDW_LION_results.keys() if i.startswith('LION-90')][0]
accuracy_lion90 = all_RBF_IDW_LION_results[lion90_name]['Accuracy']
lion95_name = [i for i in all_RBF_IDW_LION_results.keys() if i.startswith('LION-95')][0]
accuracy_lion95 = all_RBF_IDW_LION_results[lion95_name]['Accuracy']
lion99_name = [i for i in all_RBF_IDW_LION_results.keys() if i.startswith('LION-99')][0]
accuracy_lion99 = all_RBF_IDW_LION_results[lion99_name]['Accuracy']
lion100_name = [i for i in all_RBF_IDW_LION_results.keys() if i.startswith('LION-100')][0]
accuracy_lion100 = all_RBF_IDW_LION_results[lion100_name]['Accuracy']

kl_lion90 = np.mean(all_RBF_IDW_LION_results[lion90_name]['KL-Divergence'])
kl_lion95 = np.mean(all_RBF_IDW_LION_results[lion95_name]['KL-Divergence'])
kl_lion99 = np.mean(all_RBF_IDW_LION_results[lion99_name]['KL-Divergence'])
kl_lion100 = np.mean(all_RBF_IDW_LION_results[lion100_name]['KL-Divergence'])

dist_lion90 = np.mean(all_RBF_IDW_LION_results[lion90_name]['DistancePercentile'])
dist_lion95 = np.mean(all_RBF_IDW_LION_results[lion95_name]['DistancePercentile'])
dist_lion99 = np.mean(all_RBF_IDW_LION_results[lion99_name]['DistancePercentile'])
dist_lion100 = np.mean(all_RBF_IDW_LION_results[lion100_name]['DistancePercentile'])

lion_method_list = ["LION; $r_x$ at %dth perc.; $p$=%.1f"%(i, lion_optimal_power[i])
                    for i in sorted(lion_optimal_power)]

lion_accuracy = [accuracy_lion90, accuracy_lion95, accuracy_lion99, accuracy_lion100]
lion_distance_percentiles = [dist_lion90, dist_lion95, dist_lion99, dist_lion100]
lion_avg_kl = [kl_lion90, kl_lion95, kl_lion99, kl_lion100]

kernelized_results_file = exp_cluster_postprocess_Kernelized.generate_kernelized_postprocess_filename(parameters)
with open(kernelized_results_file, 'rb') as f:
        kernelized_method_list, kernelized_accuracy, kernelized_avg_kl, kernelized_distance_percentiles = pickle.load(f)

gd_input_file = exp_cluster_postprocess_GD.generate_gd_postprocess_filename(parameters)
with open(gd_input_file, "rb") as f:
    gd_method_list, gd_accuracy, gd_avg_kl, gd_distance_percentiles = pickle.load(f)
    
nn_input_file = exp_cluster_postprocess_NN.generate_nn_postprocess_filename(parameters)
with open(nn_input_file, "rb") as f:
    nn_method_list, nn_accuracy, nn_avg_kl, nn_distance_percentiles = pickle.load(f)

print("DATA LOADED")

# ==================== PRINTING THE TABLE IN HUMAN-READABLE FORMAT

print("HUMAN-READABLE CLUSTER ATTRIBUTION TEST TABLE")
print("Baseline accurancy: ", baseline_accuracy, "(exceeding it is unlikely)")
print("METHOD - ACCURACY - PERCENTILE - KL DIVERGENCE")

for i in range(len(rbf_method_list)):
    print(rbf_method_list[i], rbf_accuracy[i], rbf_distance_percentiles[i], rbf_avg_kl[i])

for i in range(len(idw_method_list)):
    print(idw_method_list[i], idw_accuracy[i], idw_distance_percentiles[i], idw_avg_kl[i])

for i in range(len(nn_method_list)):
    print(nn_method_list[i], nn_accuracy[i], nn_distance_percentiles[i], nn_avg_kl[i])

for i in range(len(kernelized_method_list)):
    print(kernelized_method_list[i], kernelized_accuracy[i], kernelized_distance_percentiles[i], kernelized_avg_kl[i])

for i in range(len(lion_method_list)):
    print(lion_method_list[i], lion_accuracy[i], lion_distance_percentiles[i], lion_avg_kl[i])

print("Notes:")
print("Accuracy - most important")
print("Distance percentile - lower the better, but not at the expense of accuracy")
print("KL divergence - lower the better, but accuracy is way more important")

# ==================== Building the table

print("\n\nTABLE FOR COPY-PASTING TO LATEX\n\n")

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

s += '\t\\textbf{Baseline} & %.2f\\%% & - & %.5f\n' % (baseline_accuracy * 100, initial_kl_divergence)
s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{RBF Interpolation}}\n'
s += '\t\\\\\n'

for j in range(len(rbf_method_list)):
    s += '\t %s & %.2f\\%% & %.3f & %.5f\n' % (rbf_method_list[j][6:], rbf_accuracy[j] * 100, rbf_distance_percentiles[j],
                                            rbf_avg_kl[j])
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{IDW Interpolation}}\n'
s += '\t\\\\\n'

for j in range(len(idw_method_list)):
    s += '\t %s & %.2f\\%% & %.3f & %.5f\n' % (idw_method_list[j][6:], idw_accuracy[j] * 100, idw_distance_percentiles[j],
                                            idw_avg_kl[j])
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{Gradient Descent}}\n'
s += '\t\\\\\n'

for j in range(len(gd_method_list)):
    s += '\t %s & %.2f\\%% & %.3f & %.5f\n' % (gd_method_list[j], gd_accuracy[j] * 100, gd_distance_percentiles[j],
                                              gd_avg_kl[j])
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{Neural Networks}}\n'
s += '\t\\\\\n'

for j in range(len(nn_method_list)):
    s += '\t %s & %.2f\\%% & %.3f & %.5f\n' % (
    nn_method_list[j][5:], nn_accuracy[j] * 100, nn_distance_percentiles[j], nn_avg_kl[j])
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{Kernelized tSNE}}\n'
s += '\t\\\\\n'

for j in range(len(kernelized_method_list)):
    s += '\t %s & %.2f\\%% & %.3f & %.5f\n' % (kernelized_method_list[j][12:], kernelized_accuracy[j] * 100,
                                               kernelized_distance_percentiles[j], kernelized_avg_kl[j])
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{LION tSNE}}\n'
s += '\t\\\\\n'

for j in range(len(lion_method_list)):
    s += '\t %s & \\textbf{%.2f\\%%} & \\textbf{%.3f} & \\textbf{%.5f}\n' % \
          (';'.join(lion_method_list[j].split(";")[1:]), lion_accuracy[j] * 100, lion_distance_percentiles[j],
           lion_avg_kl[j])
    s += '\t\\\\\n'

s += '''
    \\bottomrule
    \\end{tabular}
\\end{table}
'''

tab_text_file = '../tables/tab-cluster-attribution-test.txt'
with open(tab_text_file, 'wt') as f:
    f.write(s)

print(s)
