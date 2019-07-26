import lion_tsne
import generate_data
import settings
import pickle
import numpy as np

import cluster_lion_RBF_IDW_commons
import exp_cluster_attr_test_IDW_RBF
import exp_cluster_attr_test_LION
import exp_cluster_postprocess_Kernelized
import exp_lion_power_performance
import exp_cluster_postprocess_GD
import exp_cluster_postprocess_RBF_IDW_LION
import exp_cluster_attr_test_IDW_higher
import logging

logging.basicConfig(level=logging.INFO)

parameters = settings.parameters
Y_mnist = generate_data.load_y_mnist(parameters=parameters)
picked_indices = generate_data.load_nearest_training_indices(parameters=parameters)
picked_indices_y_mnist = Y_mnist[picked_indices,:]
dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
baseline_accuracy = generate_data.get_baseline_accuracy(parameters=parameters)

lion_power_plot_data_file = exp_lion_power_performance.generate_lion_power_plot_filename(parameters=parameters)

baseline_precision = generate_data.load_baseline_precision(parameters=parameters)

with open(lion_power_plot_data_file, 'rb') as f:
    _, _, lion_optimal_power = pickle.load(f)

idw_rbf_cluster_results_file = cluster_lion_RBF_IDW_commons.generate_cluster_results_filename(
    exp_cluster_attr_test_IDW_RBF.cluster_results_file_prefix, parameters)
with open(idw_rbf_cluster_results_file, "rb") as f:
    all_RBF_IDW_results = pickle.load(f)

idw_rbf_cluster_results_file_higher = cluster_lion_RBF_IDW_commons.generate_cluster_results_filename(
    exp_cluster_attr_test_IDW_higher.cluster_results_file_prefix, parameters)
with open(idw_rbf_cluster_results_file_higher, "rb") as f:
    all_RBF_IDW_results_higher = pickle.load(f)

for i in all_RBF_IDW_results_higher.keys():
    all_RBF_IDW_results[i] = all_RBF_IDW_results_higher[i]

lion_cluster_results_file = cluster_lion_RBF_IDW_commons.generate_cluster_results_filename(
    exp_cluster_attr_test_LION.cluster_results_file_prefix, parameters)
with open(lion_cluster_results_file, "rb") as f:
    all_LION_results = pickle.load(f)

fname = cluster_lion_RBF_IDW_commons.generate_cluster_results_filename\
        (exp_cluster_postprocess_RBF_IDW_LION.output_prefix, parameters)
with open(fname, "rb") as f:
    precision = pickle.load(f)


precision_multiquadric = precision["RBF-multiquadric"]
precision_gaussian = precision["RBF-gaussian"]
precision_linear = precision["RBF-linear"]
precision_cubic = precision["RBF-cubic"]
precision_quintic = precision["RBF-quintic"]
precision_inverse = precision["RBF-inverse"]
precision_thin_plate = precision["RBF-thin-plate"]

accuracy_multiquadric = np.mean(all_RBF_IDW_results["RBF-multiquadric"]["Accuracy"])
accuracy_gaussian = np.mean(all_RBF_IDW_results["RBF-gaussian"]["Accuracy"])
accuracy_linear = np.mean(all_RBF_IDW_results["RBF-linear"]["Accuracy"])
accuracy_cubic = np.mean(all_RBF_IDW_results["RBF-cubic"]["Accuracy"])
accuracy_quintic = np.mean(all_RBF_IDW_results["RBF-quintic"]["Accuracy"])
accuracy_inverse = np.mean(all_RBF_IDW_results["RBF-inverse"]["Accuracy"])
accuracy_thin_plate = np.mean(all_RBF_IDW_results["RBF-thin-plate"]["Accuracy"])

kl_multiquadric = np.mean(all_RBF_IDW_results["RBF-multiquadric"]["KL-Divergence"])
kl_gaussian = np.mean(all_RBF_IDW_results["RBF-gaussian"]["KL-Divergence"])
kl_linear = np.mean(all_RBF_IDW_results["RBF-linear"]["KL-Divergence"])
kl_cubic = np.mean(all_RBF_IDW_results["RBF-cubic"]["KL-Divergence"])
kl_quintic = np.mean(all_RBF_IDW_results["RBF-quintic"]["KL-Divergence"])
kl_inverse = np.mean(all_RBF_IDW_results["RBF-inverse"]["KL-Divergence"])
kl_thin_plate = np.mean(all_RBF_IDW_results["RBF-thin-plate"]["KL-Divergence"])

time_multiquadric = np.mean(all_RBF_IDW_results["RBF-multiquadric"]["TimePerPoint"]).total_seconds() * 1000
time_gaussian = np.mean(all_RBF_IDW_results["RBF-gaussian"]["TimePerPoint"]).total_seconds() * 1000
time_linear = np.mean(all_RBF_IDW_results["RBF-linear"]["TimePerPoint"]).total_seconds() * 1000
time_cubic = np.mean(all_RBF_IDW_results["RBF-cubic"]["TimePerPoint"]).total_seconds() * 1000
time_quintic = np.mean(all_RBF_IDW_results["RBF-quintic"]["TimePerPoint"]).total_seconds() * 1000
time_inverse = np.mean(all_RBF_IDW_results["RBF-inverse"]["TimePerPoint"]).total_seconds() * 1000
time_thin_plate = np.mean(all_RBF_IDW_results["RBF-thin-plate"]["TimePerPoint"]).total_seconds() * 1000

rbf_method_list = ['RBF - Multiquadric', 'RBF - Gaussian',
        'RBF - Inverse Multiquadric', 'RBF - Linear', 'RBF - Cubic', 'RBF - Quintic',
        'RBF - Thin Plate']

rbf_precision = [precision_multiquadric, precision_gaussian, precision_inverse,
                 precision_linear, precision_cubic, precision_quintic,
                 precision_thin_plate]
rbf_time = [time_multiquadric, time_gaussian, time_inverse, time_linear, time_cubic, time_quintic,
                      time_thin_plate]
rbf_accuracy = [accuracy_multiquadric, accuracy_gaussian, accuracy_inverse, accuracy_linear, accuracy_cubic,
                accuracy_quintic, accuracy_thin_plate]
rbf_avg_kl = [kl_multiquadric, kl_gaussian, kl_inverse, kl_linear, kl_cubic, kl_quintic, kl_thin_plate]

print("HUMAN-READABLE CLUSTER ATTRIBUTION TEST TABLE")
print("Baseline accurancy: ", baseline_accuracy, "(exceeding it is unlikely)")
print("Baseline precision: ", baseline_precision, "(only for reference)")
print("METHOD - ACCURACY - PRECISION - KL DIVERGENCE - TIME (ms)")

def get_tabs(s):
    if len(s)>26:
        return "\t"
    if len(s)>22:
        return "\t\t"
    if len(s)>18:
        return "\t\t\t"
    if len(s)>14:
        return "\t\t\t\t"
    return "\t\t\t\t\t"

print ("========== RBF ==========")
for i in range(len(rbf_method_list)):
    print(rbf_method_list[i], get_tabs(rbf_method_list[i]), "%.4f" % rbf_accuracy[i], "%.4f" % rbf_precision[i],
          "%.4f" % rbf_avg_kl[i], "%2.2f" % rbf_time[i])

# =============================================================================================================
keys_copy = all_RBF_IDW_results.keys()
idw_optimal_name = [i for i in keys_copy if i.startswith("IDW") and '.' in i][0]

#print(idw_optimal_name)
accuracy_idw1 = np.mean(all_RBF_IDW_results['IDW-1']['Accuracy'])
accuracy_idw20 = np.mean(all_RBF_IDW_results['IDW-20']['Accuracy'])
accuracy_idw70 = np.mean(all_RBF_IDW_results['IDW-70']['Accuracy'])
accuracy_idw_optimal = np.mean(all_RBF_IDW_results[idw_optimal_name]['Accuracy'])

precision_idw1 = precision["IDW-1"]
precision_idw20 = precision["IDW-20"]
precision_idw70 = precision["IDW-70"]
precision_idw_optimal = precision[idw_optimal_name]

time_idw1 = np.mean(all_RBF_IDW_results['IDW-1']['TimePerPoint']).total_seconds() * 1000
time_idw20 = np.mean(all_RBF_IDW_results['IDW-20']['TimePerPoint']).total_seconds() * 1000
time_idw70 = np.mean(all_RBF_IDW_results['IDW-70']['TimePerPoint']).total_seconds() * 1000
time_idw_optimal = np.mean(all_RBF_IDW_results[idw_optimal_name]['TimePerPoint']).total_seconds() * 1000

kl_idw1 = np.mean(all_RBF_IDW_results['IDW-1']['KL-Divergence'])
kl_idw20 = np.mean(all_RBF_IDW_results['IDW-20']['KL-Divergence'])
kl_idw70 = np.mean(all_RBF_IDW_results['IDW-70']['KL-Divergence'])
kl_idw_optimal = np.mean(all_RBF_IDW_results[idw_optimal_name]['KL-Divergence'])

dist_idw1 = np.mean(all_RBF_IDW_results['IDW-1']['DistancePercentile'])
dist_idw20 = np.mean(all_RBF_IDW_results['IDW-20']['DistancePercentile'])
dist_idw70 = np.mean(all_RBF_IDW_results['IDW-70']['DistancePercentile'])
dist_idw_optimal = np.mean(all_RBF_IDW_results[idw_optimal_name]['DistancePercentile'])

idw_method_list = ["IDW - Power 1","IDW - Power 20",
    "IDW - Power "+idw_optimal_name[-4:], "IDW - Power 70"]

idw_accuracy = [accuracy_idw1, accuracy_idw20, accuracy_idw_optimal, accuracy_idw70]
idw_distance_percentiles = [dist_idw1, dist_idw20, dist_idw_optimal, dist_idw70]
idw_avg_kl = [kl_idw1, kl_idw20, kl_idw_optimal, kl_idw70]
idw_precision = [precision_idw1, precision_idw20, precision_idw_optimal, precision_idw70]
idw_time = [time_idw1, time_idw20, time_idw_optimal, time_idw70]

print ("========== IDW ==========")
for i in range(len(idw_method_list)):
    print(idw_method_list[i], get_tabs(idw_method_list[i]), "%.4f" % idw_accuracy[i], "%.4f" % idw_precision[i],
          "%.4f" % idw_avg_kl[i], "%2.2f" % idw_time[i])

lion90_name = [i for i in all_LION_results.keys() if i.startswith('LION-90')][0]
accuracy_lion90 = all_LION_results[lion90_name]['Accuracy']
lion95_name = [i for i in all_LION_results.keys() if i.startswith('LION-95')][0]
accuracy_lion95 = all_LION_results[lion95_name]['Accuracy']
lion99_name = [i for i in all_LION_results.keys() if i.startswith('LION-99')][0]
accuracy_lion99 = all_LION_results[lion99_name]['Accuracy']
lion100_name = [i for i in all_LION_results.keys() if i.startswith('LION-100')][0]
accuracy_lion100 = all_LION_results[lion100_name]['Accuracy']

precision_lion90 = precision[lion90_name]
precision_lion95 = precision[lion95_name]
precision_lion99 = precision[lion99_name]
precision_lion100 = precision[lion100_name]

kl_lion90 = np.mean(all_LION_results[lion90_name]['KL-Divergence'])
kl_lion95 = np.mean(all_LION_results[lion95_name]['KL-Divergence'])
kl_lion99 = np.mean(all_LION_results[lion99_name]['KL-Divergence'])
kl_lion100 = np.mean(all_LION_results[lion100_name]['KL-Divergence'])

time_lion90 = np.mean(all_LION_results[lion90_name]['TimePerPoint']).total_seconds() * 1000
time_lion95 = np.mean(all_LION_results[lion95_name]['TimePerPoint']).total_seconds() * 1000
time_lion99 = np.mean(all_LION_results[lion99_name]['TimePerPoint']).total_seconds() * 1000
time_lion100 = np.mean(all_LION_results[lion100_name]['TimePerPoint']).total_seconds() * 1000

dist_lion90 = np.mean(all_LION_results[lion90_name]['DistancePercentile'])
dist_lion95 = np.mean(all_LION_results[lion95_name]['DistancePercentile'])
dist_lion99 = np.mean(all_LION_results[lion99_name]['DistancePercentile'])
dist_lion100 = np.mean(all_LION_results[lion100_name]['DistancePercentile'])

lion_method_list = ["LION; $r_x$ at %dth perc.; $p$=%.1f"%(i, lion_optimal_power[i])
                    for i in sorted(lion_optimal_power)]

lion_accuracy = [accuracy_lion90, accuracy_lion95, accuracy_lion99, accuracy_lion100]
lion_distance_percentiles = [dist_lion90, dist_lion95, dist_lion99, dist_lion100]
lion_avg_kl = [kl_lion90, kl_lion95, kl_lion99, kl_lion100]
lion_precision = [precision_lion90, precision_lion95, precision_lion99, precision_lion100]
lion_time = [time_lion90, time_lion95, time_lion99, time_lion100]

print ("========== LION ==========")
for i in range(len(lion_method_list)):
    print(lion_method_list[i], get_tabs(lion_method_list[i]), "%.4f" % lion_accuracy[i], "%.4f" % lion_precision[i],
          "%.4f" % lion_avg_kl[i], "%2.2f" % lion_time[i])


kernelized_results_file = exp_cluster_postprocess_Kernelized.generate_kernelized_postprocess_filename(parameters)
with open(kernelized_results_file, 'rb') as f:
    kernelized_method_list, kernelized_accuracy, kernelized_precision, \
    kernelized_avg_kl, kernelized_per_item_time, kernelized_distance_percentiles = pickle.load(f)


print ("========== KERNELIZED ==========")
for i in range(len(kernelized_method_list)):
    print(kernelized_method_list[i], get_tabs(kernelized_method_list[i]), "%.4f" % kernelized_accuracy[i],
          "%.4f" % kernelized_precision[i], "%.4f" % kernelized_avg_kl[i], "%2.2f" % (kernelized_per_item_time[i]*1000))


gd_input_file = exp_cluster_postprocess_GD.generate_gd_postprocess_filename(parameters)
with open(gd_input_file, "rb") as f:
    gd_method_list, gd_accuracy, gd_precision, gd_time, gd_avg_kl, gd_distance_percentiles = pickle.load(f)

print ("========== GRADIENT DESCENT ==========")
for i in range(len(gd_method_list)):
    print(gd_method_list[i], get_tabs(gd_method_list[i]), "%.4f" % gd_accuracy[i],
          "%.4f" % gd_precision[i], "%.4f" % gd_avg_kl[i], "%2.2f" % (gd_time[i]*1000))


print("DATA LOADED")


def get_time_euphemism(time_s):
    # Time in seconds, response in millis
    if time_s > 100:
        return ">$10^5$"
    if time_s > 10:
        return ">$10^4$"
    if time_s > 1:
        return ">$10^3$"
    return "%2.2f" % (time_s*1000)

# ==================== Building the table

print("\n\nTABLE FOR COPY-PASTING TO LATEX\n\n")

s = ""

s += '''\\begin{table} \small\sf\centering \caption{Digits placement test: methods comparison}  \label{tab_cluster_methods_comparison}
    \\begin{tabular}{ m{0.19\\textwidth}  m{0.07\\textwidth}  m{0.07\\textwidth}  m{0.06\\textwidth} }
        \\toprule
            \\textbf{Method}
            & \\textbf{Precision}
            & \\textbf{KL Div.}
            & \\textbf{Time (ms)}
        \\\\ \\midrule'''

initial_kl_divergence, _ = lion_tsne.kl_divergence_and_gradient(y=dTSNE_mnist.Y, p_matrix=dTSNE_mnist.P_matrix)

s += '\t\\textbf{Baseline} & %.4f & %.5f & -\n' % (baseline_precision, initial_kl_divergence)
s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{RBF Interpolation}}\n'
s += '\t\\\\\n'

for j in range(len(rbf_method_list)):
    s += '\t %s & %.4f & %.5f & %.2f\n' % (rbf_method_list[j][6:], rbf_precision[j], rbf_avg_kl[j], rbf_time[j])
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{IDW Interpolation}}\n'
s += '\t\\\\\n'

for j in range(len(idw_method_list)):
    s += '\t %s & %.4f & %.5f & %.2f\n' % (idw_method_list[j][6:], idw_precision[j], idw_avg_kl[j], idw_time[j])
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{Gradient Descent}}\n'
s += '\t\\\\\n'

for j in range(len(gd_method_list)):
    s += '\t %s & %.4f & %.5f & %s\n' % (gd_method_list[j], gd_precision[j], gd_avg_kl[j],
                                               get_time_euphemism(gd_time[j]))
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{Kernelized tSNE}}\n'
s += '\t\\\\\n'

for j in range(len(kernelized_method_list)):
    s += '\t %s & %.4f & %.5f & %.2f\n' % (kernelized_method_list[j][12:], kernelized_precision[j],
                                               kernelized_avg_kl[j], kernelized_per_item_time[j]*1000)
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{LION tSNE}}\n'
s += '\t\\\\\n'

for j in range(len(lion_method_list)):
    s += '\t %s & \\textbf{%.4f} & \\textbf{%.5f} & \\textbf{%.2f}\n' % \
          (';'.join(lion_method_list[j].split(";")[1:]), lion_precision[j], lion_avg_kl[j], lion_time[j])
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
