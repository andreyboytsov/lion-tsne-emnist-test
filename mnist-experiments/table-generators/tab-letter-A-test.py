import lion_tsne
import settings
import generate_data
import letter_A_lion_RBF_IDW_commons
import exp_letter_A_test_IDW_RBF
import exp_letter_A_test_LION
import numpy as np
import pickle
import exp_lion_power_performance
import exp_letter_A_postprocess_kernelized
import exp_letter_A_postprocess_GD
import exp_letter_A_test_IDW_higher

parameters = settings.parameters
dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=settings.parameters)

lion_power_plot_data_file = exp_lion_power_performance.generate_lion_power_plot_filename(parameters=parameters)

with open(lion_power_plot_data_file, 'rb') as f:
    _, _, lion_optimal_power = pickle.load(f)

idw_rbf_letter_A_results_file = letter_A_lion_RBF_IDW_commons.generate_letter_A_results_filename(
    exp_letter_A_test_IDW_RBF.letter_A_results_file_prefix, parameters)
with open(idw_rbf_letter_A_results_file, "rb") as f:
    all_RBF_IDW_results = pickle.load(f)

idw_rbf_cluster_results_file_higher = letter_A_lion_RBF_IDW_commons.generate_letter_A_results_filename(
    exp_letter_A_test_IDW_higher.letter_A_results_file_prefix, parameters)
with open(idw_rbf_cluster_results_file_higher, "rb") as f:
    all_RBF_IDW_results_higher = pickle.load(f)

for i in all_RBF_IDW_results_higher.keys():
    all_RBF_IDW_results[i] = all_RBF_IDW_results_higher[i]

lion_letter_A_results_file = letter_A_lion_RBF_IDW_commons.generate_letter_A_results_filename(
    exp_letter_A_test_LION.letter_A_results_file_prefix, parameters)
with open(lion_letter_A_results_file, "rb") as f:
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

time_multiquadric = np.mean(all_RBF_IDW_results["RBF-multiquadric"]["TimePerPoint"]).total_seconds() * 1000
time_gaussian = np.mean(all_RBF_IDW_results["RBF-gaussian"]["TimePerPoint"]).total_seconds() * 1000
time_linear = np.mean(all_RBF_IDW_results["RBF-linear"]["TimePerPoint"]).total_seconds() * 1000
time_cubic = np.mean(all_RBF_IDW_results["RBF-cubic"]["TimePerPoint"]).total_seconds() * 1000
time_quintic = np.mean(all_RBF_IDW_results["RBF-quintic"]["TimePerPoint"]).total_seconds() * 1000
time_inverse = np.mean(all_RBF_IDW_results["RBF-inverse"]["TimePerPoint"]).total_seconds() * 1000
time_thin_plate = np.mean(all_RBF_IDW_results["RBF-thin-plate"]["TimePerPoint"]).total_seconds() * 1000

rbf_time = [time_multiquadric, time_gaussian, time_inverse, time_linear, time_cubic, time_quintic,
                      time_thin_plate]

rbf_method_list = ['RBF - Multiquadric', 'RBF - Gaussian',
        'RBF - Inverse Multiquadric', 'RBF - Linear', 'RBF - Cubic', 'RBF - Quintic',
        'RBF - Thin Plate']

rbf_distance_percentiles = [dist_multiquadric, dist_gaussian, dist_inverse, dist_linear, dist_cubic, dist_quintic,
                      dist_thin_plate]
rbf_avg_kl = [kl_multiquadric, kl_gaussian, kl_inverse, kl_linear, kl_cubic, kl_quintic, kl_thin_plate]

print("HUMAN-READABLE OUTLIERS TABLE")
print("METHOD - PERCENTILE - KL DIVERGENCE - TIME")

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
    print(rbf_method_list[i], get_tabs(rbf_method_list[i]), "%2.2f" % rbf_distance_percentiles[i],
          "%.4f" % rbf_avg_kl[i], "%2.2f" % rbf_time[i])


# =============================================================================================================
keys_copy = all_RBF_IDW_results.keys()
idw_optimal_name = [i for i in keys_copy if i.startswith("IDW") and '.' in i][0]

kl_idw1 = np.mean(all_RBF_IDW_results['IDW-1']['KL-Divergence'])
kl_idw20 = np.mean(all_RBF_IDW_results['IDW-20']['KL-Divergence'])
kl_idw70 = np.mean(all_RBF_IDW_results['IDW-70']['KL-Divergence'])
kl_idw_optimal = np.mean(all_RBF_IDW_results[idw_optimal_name]['KL-Divergence'])

dist_idw1 = np.mean(all_RBF_IDW_results['IDW-1']['DistancePercentile'])
dist_idw20 = np.mean(all_RBF_IDW_results['IDW-20']['DistancePercentile'])
dist_idw70 = np.mean(all_RBF_IDW_results['IDW-70']['DistancePercentile'])
dist_idw_optimal = np.mean(all_RBF_IDW_results[idw_optimal_name]['DistancePercentile'])

time_idw1 = np.mean(all_RBF_IDW_results['IDW-1']['TimePerPoint']).total_seconds() * 1000
time_idw20 = np.mean(all_RBF_IDW_results['IDW-20']['TimePerPoint']).total_seconds() * 1000
time_idw70 = np.mean(all_RBF_IDW_results['IDW-70']['TimePerPoint']).total_seconds() * 1000
time_idw_optimal = np.mean(all_RBF_IDW_results[idw_optimal_name]['TimePerPoint']).total_seconds() * 1000

idw_method_list = ["IDW - Power 1", "IDW - Power 20",
    "IDW - Power "+idw_optimal_name[-4:], "IDW - Power 70"]

idw_distance_percentiles = [dist_idw1, dist_idw20, dist_idw_optimal, dist_idw70]
idw_avg_kl = [kl_idw1, kl_idw20, kl_idw_optimal, kl_idw70]
idw_time = [time_idw1, time_idw20, time_idw_optimal, time_idw70]

print ("========== IDW ==========")
for i in range(len(idw_method_list)):
    print(idw_method_list[i], get_tabs(idw_method_list[i]), "%2.2f" % idw_distance_percentiles[i],
          "%.4f" % idw_avg_kl[i], "%2.2f" % idw_time[i])

# ====================================== LION =====================================

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

time_lion90 = np.mean(all_LION_results[lion90_name]['TimePerPoint']).total_seconds() * 1000
time_lion95 = np.mean(all_LION_results[lion95_name]['TimePerPoint']).total_seconds() * 1000
time_lion99 = np.mean(all_LION_results[lion99_name]['TimePerPoint']).total_seconds() * 1000
time_lion100 = np.mean(all_LION_results[lion100_name]['TimePerPoint']).total_seconds() * 1000

lion_time = [time_lion90, time_lion95, time_lion99, time_lion100]

lion_method_list = ["LION; $r_x$ at %dth perc.; $p$=%.1f"%(i, lion_optimal_power[i])
                    for i in sorted(lion_optimal_power)]

lion_distance_percentiles = [dist_lion90, dist_lion95, dist_lion99, dist_lion100]
lion_avg_kl = [kl_lion90, kl_lion95, kl_lion99, kl_lion100]

print ("========== LION ==========")
for i in range(len(lion_method_list)):
    print(lion_method_list[i], get_tabs(lion_method_list[i]), "%2.2f" % lion_distance_percentiles[i],
          "%.4f" % lion_avg_kl[i], "%2.2f" % lion_time[i])


kernelized_results_file = exp_letter_A_postprocess_kernelized.generate_kernelized_postprocess_filename(parameters)
with open(kernelized_results_file, 'rb') as f:
    kernelized_method_list, kernelized_avg_kl, kernelized_per_item_time,\
    kernelized_distance_percentiles = pickle.load(f)

print ("========== KERNELIZED ==========")
for i in range(len(kernelized_method_list)):
    print(kernelized_method_list[i], get_tabs(kernelized_method_list[i]), "%2.2f" % kernelized_distance_percentiles[i],
          "%.4f" % kernelized_avg_kl[i], "%2.2f" % (kernelized_per_item_time[i]*1000))

gd_input_file = exp_letter_A_postprocess_GD.generate_gd_postprocess_filename(parameters)
with open(gd_input_file, "rb") as f:
    gd_method_list, gd_per_item_time, gd_avg_kl, gd_distance_percentiles = pickle.load(f)


print ("========== GRADIENT DESCENT ==========")
for i in range(len(gd_method_list)):
    print(gd_method_list[i], get_tabs(gd_method_list[i]), "%2.2f" % gd_distance_percentiles[i],
          "%.4f" % gd_avg_kl[i], "%2.2f" % (gd_per_item_time[i]*1000))

print("Notes:")
print("Distance percentile - higher the better")
print("KL divergence - lower the better, but percentile is way more important")
print("Time - just make sure it is reasonable")

def get_time_euphemism(time_s):
    # Time in seconds, response in millis
    if time_s > 100:
        return ">$10^5$"
    if time_s > 10:
        return ">$10^4$"
    if time_s > 1:
        return ">$10^3$"
    return "%2.2f" % (time_s*1000)

# ================= GENERATING THE TEX TABLE ==========================

print("\n\nTABLE FOR COPY-PASTING TO LATEX\n\n\n")

s = ""

initial_kl_divergence, _ = lion_tsne.kl_divergence_and_gradient(y=dTSNE_mnist.Y, p_matrix=dTSNE_mnist.P_matrix)

s += '''\\begin{table} \small\sf\centering \caption{Letter A placement test: methods comparison.
    Original KL divergence of the dataset is %.5f}  \label{tab_letter_a_methods_comparison}
    \\begin{tabular}{ m{0.19\\textwidth}  m{0.07\\textwidth}  m{0.07\\textwidth}  m{0.06\\textwidth} }
        \\toprule
            \\textbf{Method}
            & \\textbf{Distance Perc-le}
            & \\textbf{KL Div.}
            & \\textbf{Time (ms)}
        \\\\ \\midrule''' % (initial_kl_divergence)

s += '\\multicolumn{4}{c}{\\textbf{RBF Interpolation}}\n'
s += '\t\\\\\n'

for j in range(len(rbf_method_list)):
    s +='\t %s & %.2f & %.5f & %.2f\n' % (rbf_method_list[j][6:], rbf_distance_percentiles[j],
                                          rbf_avg_kl[j], rbf_time[j])
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{IDW Interpolation}}\n'
s += '\t\\\\\n'

for j in range(len(idw_method_list)):
    s +='\t %s & %.2f &%.5f & %.2f\n' % (idw_method_list[j], idw_distance_percentiles[j],
                                         idw_avg_kl[j], idw_time[j])
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{Gradient Descent}}\n'
s += '\t\\\\\n'

for j in range(len(gd_method_list)):
    s +='\t %s & %.2f & %.5f & %s\n' % (gd_method_list[j], gd_distance_percentiles[j],
                                               gd_avg_kl[j], get_time_euphemism(gd_per_item_time[j]))
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{Kernelized tSNE}}\n'
s += '\t\\\\\n'

for j in range(len(kernelized_method_list)):
    s += '\t %s & %.2f & %.5f & %.2f\n' % (kernelized_method_list[j][12:],
                        kernelized_distance_percentiles[j], kernelized_avg_kl[j], kernelized_per_item_time[j]*1000)
    s += '\t\\\\\n'

s += '\\multicolumn{4}{c}{\\textbf{LION tSNE}}\n'
s += '\t\\\\\n'

for j in range(len(lion_method_list)):
    s +='\t \\textbf{%s} - %s & \\textbf{%.2f} &\\textbf{%.5f}  &\\textbf{%.2f}\n' % \
          (lion_method_list[j].split(";")[0], lion_method_list[j].split(";")[1], lion_distance_percentiles[j],
           lion_avg_kl[j], lion_time[j])
    s += '\t\\\\\n'

s += '''
    \\bottomrule
    \\end{tabular}
\\end{table}
'''

tab_text_file = '../tables/tab-letter_As-test.txt'
with open(tab_text_file, 'wt') as f:
    f.write(s)

print(s)