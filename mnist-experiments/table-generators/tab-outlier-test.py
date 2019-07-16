import lion_tsne
import settings
import generate_data
import outlier_lion_RBF_IDW_commons
import exp_outlier_test_IDW_RBF
import exp_outlier_test_LION
import numpy as np
import pickle
import exp_lion_power_performance
import exp_outlier_postprocess_kernelized
import exp_outlier_postprocess_NN
import exp_outlier_postprocess_GD

parameters = settings.parameters
dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=settings.parameters)

lion_power_plot_data_file = exp_lion_power_performance.generate_lion_power_plot_filename(parameters=parameters)

with open(lion_power_plot_data_file, 'rb') as f:
    _, _, lion_optimal_power = pickle.load(f)

idw_rbf_outlier_results_file = outlier_lion_RBF_IDW_commons.generate_outlier_results_filename(
    exp_outlier_test_IDW_RBF.outlier_results_file_prefix, parameters)
with open(idw_rbf_outlier_results_file, "rb") as f:
    all_RBF_IDW_results = pickle.load(f)

lion_outlier_results_file = outlier_lion_RBF_IDW_commons.generate_outlier_results_filename(
    exp_outlier_test_LION.outlier_results_file_prefix, parameters)
with open(lion_outlier_results_file, "rb") as f:
    all_LION_results = pickle.load(f)

kl_multiquadric = np.mean(all_RBF_IDW_results["RBF-multiquadric"]["KL-Divergence"])
kl_gaussian = np.mean(all_RBF_IDW_results["RBF-gaussian"]["KL-Divergence"])
kl_linear = np.mean(all_RBF_IDW_results["RBF-linear"]["KL-Divergence"])
kl_cubic = np.mean(all_RBF_IDW_results["RBF-cubic"]["KL-Divergence"])
kl_quintic = np.mean(all_RBF_IDW_results["RBF-quintic"]["KL-Divergence"])
kl_inverse = np.mean(all_RBF_IDW_results["RBF-inverse"]["KL-Divergence"])
kl_thin_plate = np.mean(all_RBF_IDW_results["RBF-thin-plate"]["KL-Divergence"])

dist_multiquadric = np.mean(all_RBF_IDW_results["RBF-multiquadric"]["DistancePercentile"])
dist_gaussian = np.mean(all_RBF_IDW_results["RBF-gaussian"]["DistancePercentile"])
dist_linear = np.mean(all_RBF_IDW_results["RBF-linear"]["DistancePercentile"])
dist_cubic = np.mean(all_RBF_IDW_results["RBF-cubic"]["DistancePercentile"])
dist_quintic = np.mean(all_RBF_IDW_results["RBF-quintic"]["DistancePercentile"])
dist_inverse = np.mean(all_RBF_IDW_results["RBF-inverse"]["DistancePercentile"])
dist_thin_plate = np.mean(all_RBF_IDW_results["RBF-thin-plate"]["DistancePercentile"])

rbf_method_list = ['RBF - Multiquadric', 'RBF - Gaussian',
        'RBF - Inverse Multiquadric', 'RBF - Linear', 'RBF - Cubic', 'RBF - Quintic',
        'RBF - Thin Plate']

rbf_distance_percentiles = [dist_multiquadric, dist_gaussian, dist_inverse, dist_linear, dist_cubic, dist_quintic,
                      dist_thin_plate]
rbf_avg_kl = [kl_multiquadric, kl_gaussian, kl_inverse, kl_linear, kl_cubic, kl_quintic, kl_thin_plate]

# =============================================================================================================
keys_copy = all_RBF_IDW_results.keys()
keys_copy -= {"IDW-1","IDW-10","IDW-20","IDW-40"}
idw_optimal_name = [i for i in keys_copy if i.startswith("IDW")][0]
print(idw_optimal_name)

kl_idw1 = np.mean(all_RBF_IDW_results['IDW-1']['KL-Divergence'])
kl_idw10 = np.mean(all_RBF_IDW_results['IDW-10']['KL-Divergence'])
kl_idw20 = np.mean(all_RBF_IDW_results['IDW-20']['KL-Divergence'])
kl_idw40 = np.mean(all_RBF_IDW_results['IDW-40']['KL-Divergence'])
kl_idw_optimal = np.mean(all_RBF_IDW_results[idw_optimal_name]['KL-Divergence'])

dist_idw1 = np.mean(all_RBF_IDW_results['IDW-1']['DistancePercentile'])
dist_idw10 = np.mean(all_RBF_IDW_results['IDW-10']['DistancePercentile'])
dist_idw20 = np.mean(all_RBF_IDW_results['IDW-20']['DistancePercentile'])
dist_idw40 = np.mean(all_RBF_IDW_results['IDW-40']['DistancePercentile'])
dist_idw_optimal = np.mean(all_RBF_IDW_results[idw_optimal_name]['DistancePercentile'])

idw_method_list = ["IDW - Power 1","IDW - Power 10", "IDW - Power 20",
    "IDW - Power "+idw_optimal_name[-4:], "IDW - Power 40"]

idw_distance_percentiles = [dist_idw1, dist_idw10, dist_idw20, dist_idw40, dist_idw_optimal]
idw_avg_kl = [kl_idw1, kl_idw10, kl_idw20, kl_idw40, kl_idw_optimal]

lion90_name = [i for i in all_LION_results.keys() if i.startswith('LION-90')][0]
lion95_name = [i for i in all_LION_results.keys() if i.startswith('LION-95')][0]
lion99_name = [i for i in all_LION_results.keys() if i.startswith('LION-99')][0]
lion100_name = [i for i in all_LION_results.keys() if i.startswith('LION-100')][0]

kl_lion90 = np.mean(all_LION_results[lion90_name]['KL-Divergence'])
kl_lion95 = np.mean(all_LION_results[lion95_name]['KL-Divergence'])
kl_lion99 = np.mean(all_LION_results[lion99_name]['KL-Divergence'])
kl_lion100 = np.mean(all_LION_results[lion100_name]['KL-Divergence'])

dist_lion90 = np.mean(all_LION_results[lion90_name]['DistancePercentile'])
dist_lion95 = np.mean(all_LION_results[lion95_name]['DistancePercentile'])
dist_lion99 = np.mean(all_LION_results[lion99_name]['DistancePercentile'])
dist_lion100 = np.mean(all_LION_results[lion100_name]['DistancePercentile'])

lion_method_list = ["LION; $r_x$ at %dth perc.; $p$=%.1f"%(i, lion_optimal_power[i])
                    for i in sorted(lion_optimal_power)]

lion_distance_percentiles = [dist_lion90, dist_lion95, dist_lion99, dist_lion100]
lion_avg_kl = [kl_lion90, kl_lion95, kl_lion99, kl_lion100]

kernelized_results_file = exp_outlier_postprocess_kernelized.generate_kernelized_postprocess_filename(parameters)
with open(kernelized_results_file, 'rb') as f:
    kernelized_method_list, kernelized_avg_kl, kernelized_distance_percentiles = pickle.load(f)

gd_input_file = exp_outlier_postprocess_GD.generate_gd_postprocess_filename(parameters)
with open(gd_input_file, "rb") as f:
    gd_method_list, gd_avg_outliers_kl, gd_outliers_distance_percentiles = pickle.load(f)

nn_input_file = exp_outlier_postprocess_NN.generate_nn_postprocess_filename(parameters)
with open(nn_input_file, "rb") as f:
    nn_method_list, nn_avg_outliers_kl, nn_outliers_distance_percentiles = pickle.load(f)


# ================= PRINTING THE TABLE ==========================

print("HUMAN-READABLE OUTLIERS TABLE")
print("METHOD - PERCENTILE - KL DIVERGENCE")

for i in range(len(rbf_method_list)):
    print(rbf_method_list[i], rbf_distance_percentiles[i], rbf_avg_kl[i])

for i in range(len(idw_method_list)):
    print(idw_method_list[i], idw_distance_percentiles[i], idw_avg_kl[i])

for i in range(len(nn_method_list)):
    print(nn_method_list[i], nn_outliers_distance_percentiles[i], nn_avg_outliers_kl[i])

for i in range(len(kernelized_method_list)):
    print(kernelized_method_list[i], kernelized_distance_percentiles[i], kernelized_avg_kl[i])

for i in range(len(lion_method_list)):
    print(lion_method_list[i], lion_distance_percentiles[i], lion_avg_kl[i])

print("Notes:")
print("Distance percentile - higher the better")
print("KL divergence - lower the better, but percentile is way more important")

# ================= GENERATING THE TEX TABLE ==========================

print("\n\nTABLE FOR COPY-PASTING TO LATEX\n\n\n")

s = ""

s +='''\\begin{table*}\\caption{Outliers test: methods comparison} \\label{tab_outliers_methods_comparison}
    \\begin{tabular}{| m{0.39\\textwidth} | m{0.20\\textwidth} | m{0.20\\textwidth} |}
        \\hline
            \\textbf{Method}
            & \\textbf{Distance Percentile}
            & \\textbf{KL Divergence}
        \\\\ \\hline'''

initial_kl_divergence, _ = lion_tsne.kl_divergence_and_gradient(y=dTSNE_mnist.Y, p_matrix=dTSNE_mnist.P_matrix)

s +='\t\\textbf{Baseline} & - & %.5f\n' % (initial_kl_divergence)
s +='\t\\\\ \\hline'

for j in range(len(rbf_method_list)):
    s +='\t %s & %.3f & %.5f\n' % (rbf_method_list[j], rbf_distance_percentiles[j],
                                          rbf_avg_kl[j])
    s +='\t\\\\ \\hline\n'

for j in range(len(idw_method_list)):
    s +='\t %s & %.3f &%.5f\n' % (idw_method_list[j], idw_distance_percentiles[j],
                                         idw_avg_kl[j])
    s +='\t\\\\ \\hline\n'

for j in range(len(gd_method_list)):
    s +='\t GD - %s & %.3f & %.5f\n' % (gd_method_list[j], gd_outliers_distance_percentiles[j],
                                               gd_avg_outliers_kl[j])
    s +='\t\\\\ \\hline\n'

for j in range(len(nn_method_list)):
    s +='\t %s & %.3f &%.5f\n' % (nn_method_list[j], nn_outliers_distance_percentiles[j],
                                        nn_avg_outliers_kl[j])
    s +='\t\\\\ \\hline\n'

for j in range(len(kernelized_method_list)):
    s += '\t %s & %.2f\\%% & %.5f\n' % (kernelized_method_list[j][12:],
                                               kernelized_distance_percentiles[j], kernelized_avg_kl[j])
    s += '\t\\\\\n'

for j in range(len(lion_method_list)):
    s +='\t \\textbf{%s} - %s & \\textbf{%.3f} &\\textbf{%.5f}\n' % \
          (lion_method_list[j].split(";")[0], lion_method_list[j].split(";")[1], lion_distance_percentiles[j],
           lion_avg_kl[j])
    s +='\t\\\\ \\hline\n'

s +='''
    \\end{tabular}
\\end{table*}
'''

tab_text_file = '../tables/tab-outliers-test.txt'
with open(tab_text_file, 'wt') as f:
    f.write(s)

print(s)