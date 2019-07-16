import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
import generate_data
import settings
import letter_lion_RBF_IDW_commons
import exp_letter_test_IDW_RBF
import exp_letter_test_LION
import exp_letter_test_NN
import exp_letter_test_kernelized
import exp_lion_power_performance
import exp_letter_test_GD
import pickle

shown_letter_indices = 20
cur_shown_letter_indices = 20
parameters = settings.parameters
Y_mnist = generate_data.load_y_mnist(parameters=parameters)

parameters = settings.parameters
dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=settings.parameters)

lion_power_plot_data_file = exp_lion_power_performance.generate_lion_power_plot_filename(parameters=parameters)

with open(lion_power_plot_data_file, 'rb') as f:
    _, _, lion_optimal_power = pickle.load(f)

idw_rbf_letter_results_file = letter_lion_RBF_IDW_commons.generate_letter_results_filename(
    exp_letter_test_IDW_RBF.letter_results_file_prefix, parameters)
with open(idw_rbf_letter_results_file, "rb") as f:
    all_RBF_IDW_results = pickle.load(f)

lion_letter_results_file = letter_lion_RBF_IDW_commons.generate_letter_results_filename(
    exp_letter_test_LION.letter_results_file_prefix, parameters)
with open(lion_letter_results_file, "rb") as f:
    all_LION_results = pickle.load(f)

rbf_method_list = ['RBF - Multiquadric', 'RBF - Gaussian',
        'RBF - Inverse Multiquadric', 'RBF - Linear', 'RBF - Cubic', 'RBF - Quintic',
        'RBF - Thin Plate']

# =============================================================================================================
keys_copy = all_RBF_IDW_results.keys()
keys_copy -= {"IDW-1","IDW-10","IDW-20","IDW-40"}
idw_optimal_name = [i for i in keys_copy if i.startswith("IDW")][0]

idw_method_list = ["IDW - Power 1","IDW - Power 10", "IDW - Power 20",
    "IDW - Power "+idw_optimal_name[-4:], "IDW - Power 40"]


lion_method_list = ["LION; $r_x$ at %dth perc.; $p$=%.1f"%(i, lion_optimal_power[i])
                    for i in sorted(lion_optimal_power)]

letters_y_multiquadric = all_RBF_IDW_results["RBF-multiquadric"]['EmbeddedPoints']
letters_y_gaussian = all_RBF_IDW_results["RBF-gaussian"]['EmbeddedPoints']
letters_y_linear = all_RBF_IDW_results["RBF-linear"]['EmbeddedPoints']
letters_y_cubic = all_RBF_IDW_results["RBF-cubic"]['EmbeddedPoints']
letters_y_quintic = all_RBF_IDW_results["RBF-quintic"]['EmbeddedPoints']
letters_y_inverse = all_RBF_IDW_results["RBF-inverse"]['EmbeddedPoints']
letters_y_thin_plate = all_RBF_IDW_results["RBF-thin-plate"]['EmbeddedPoints']

letters_y_idw1 = all_RBF_IDW_results['IDW-1']['EmbeddedPoints']
letters_y_idw10 = all_RBF_IDW_results['IDW-10']['EmbeddedPoints']
letters_y_idw20 = all_RBF_IDW_results['IDW-20']['EmbeddedPoints']
letters_y_idw40 = all_RBF_IDW_results['IDW-40']['EmbeddedPoints']
letters_y_idw_optimal = all_RBF_IDW_results[idw_optimal_name]['EmbeddedPoints']

lion90_name = [i for i in all_LION_results.keys() if i.startswith('LION-90')][0]
letters_y_lion90 = all_LION_results[lion90_name]['EmbeddedPoints']
lion95_name = [i for i in all_LION_results.keys() if i.startswith('LION-95')][0]
letters_y_lion95 = all_LION_results[lion95_name]['EmbeddedPoints']
lion99_name = [i for i in all_LION_results.keys() if i.startswith('LION-99')][0]
letters_y_lion99 = all_LION_results[lion99_name]['EmbeddedPoints']
lion100_name = [i for i in all_LION_results.keys() if i.startswith('LION-100')][0]
letters_y_lion100 = all_LION_results[lion100_name]['EmbeddedPoints']

kernelized_results_file = exp_letter_test_kernelized.generate_letter_results_filename(parameters)
with open(kernelized_results_file, 'rb') as f:
    kernelized_detailed_tsne_method_results, \
            kernelized_detailed_tsne_method_list = pickle.load(f)
ind = [4,24,49]
kernelized_tsne_method_list = [kernelized_detailed_tsne_method_list[i][:10]+kernelized_detailed_tsne_method_list[i][-8:]
                               for i in ind]
kernelized_tsne_letters_results = [kernelized_detailed_tsne_method_results[i] for i in ind]


gd_results_file = exp_letter_test_GD.generate_letter_results_filename(parameters=parameters)
with open(gd_results_file, 'rb') as f:
    (letters_y_gd_transformed, letters_y_gd_variance_recalc_transformed,
     letters_y_gd_transformed_random, letters_y_gd_variance_recalc_transformed_random,
     letters_y_gd_early_exagg_transformed_random,
     letters_y_gd_early_exagg_transformed,
     letters_y_gd_variance_recalc_early_exagg_transformed_random,
     picked_random_starting_positions,
     letters_y_gd_variance_recalc_early_exagg_transformed, covered_samples) = pickle.load(f)

nn_results_file = exp_letter_test_NN.generate_letter_results_filename(parameters)
with open(nn_results_file, 'rb') as f:
        nn_letters_results, nn_models_orig, nn_method_list = pickle.load(f)

# ============================================
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
#f, ax = plt.subplots(2,3)
#plt.gcf().set_size_inches(18,12)
chosen_indices = list(range(20))
#cur_shown_letter_indices = chosen_indices
#shown_letter_indices = chosen_indices
#chosen_indices = [3]


#plt.figure().set_dpi(300)
plt.figure(dpi=300)
plt.gcf().set_size_inches(6.8,6.8)

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

#n_row = 2
#n_col = 3

gs = gridspec.GridSpec(n_row, n_col)
gs.update(wspace=0, hspace=0) # set the spacing between axes.

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

# ====================================== RBF =====================================

ax[rbf_X][rbf_Y].scatter(Y_mnist[:, 0], Y_mnist[:, 1], c= 'gray', zorder=1, label=None, marker='.',s=point_size_gray)
h1 = ax[rbf_X][rbf_Y].scatter(letters_y_multiquadric[:shown_letter_indices, 0],
                letters_y_multiquadric[:shown_letter_indices, 1], c='red', zorder=1, label=None, marker='.',
                              s=point_size_interest)
h2 = ax[rbf_X][rbf_Y].scatter(letters_y_gaussian[:shown_letter_indices, 0],
                letters_y_gaussian[:shown_letter_indices, 1], c='blue', zorder=1, label=None, marker='.',
                             s=point_size_interest)
h3 = ax[rbf_X][rbf_Y].scatter(letters_y_inverse[:shown_letter_indices, 0],
                letters_y_inverse[:shown_letter_indices, 1], c='green', zorder=1, label=None, marker='.',
                             s=point_size_interest)
h4 = ax[rbf_X][rbf_Y].scatter(letters_y_linear[:shown_letter_indices, 0],
                letters_y_linear[:shown_letter_indices, 1], c='purple', zorder=1, label=None, marker='.',
                             s=point_size_interest)
h5 = ax[rbf_X][rbf_Y].scatter(letters_y_cubic[:shown_letter_indices, 0],
                letters_y_cubic[:shown_letter_indices, 1], c='cyan', zorder=1, label=None, marker='.',
                             s=point_size_interest)
h6 = ax[rbf_X][rbf_Y].scatter(letters_y_quintic[:shown_letter_indices, 0],
                letters_y_quintic[:shown_letter_indices, 1], c='orange', zorder=1, label=None, marker='.',
                             s=point_size_interest)
h7 = ax[rbf_X][rbf_Y].scatter(letters_y_thin_plate[:shown_letter_indices,0],
                letters_y_thin_plate[:shown_letter_indices,1], c='pink', marker='.', zorder=3,alpha=0.9,
                             s=point_size_interest)

#ax[rbf_X][rbf_Y].legend([h1,h2,h3,h4,h5,h6,h7], ["RBF - Multiquadric","RBF - Gaussian", 'RBF - Inverse Multiquadric', 'RBF - Linear',
#                                  'RBF - Cubic','RBF - Quintic (out of scope)', 'RBF - Thin Plate'], fontsize = 14)
ax[rbf_X][rbf_Y].legend([h1,h2,h3,h4,h5,h6,h7], ["Multiquadric","Gaussian", 'Inverse Multiquadr.', 'Linear',
                                  'Cubic','Quintic', 'Thin Plate'], ncol=1,
                       prop=font_properties, borderpad=0.1,handlelength=2,
                       columnspacing = 0, loc = 1, handletextpad=-0.7,frameon=True)

# ==================================== IDW =======================================

ax[idw_Y][idw_X].scatter(Y_mnist[:, 0], Y_mnist[:, 1], c= 'gray', zorder=1, label=None, marker='.',s=point_size_gray)
h1 = ax[idw_Y][idw_X].scatter(letters_y_idw1[:shown_letter_indices, 0],
                letters_y_idw1[:shown_letter_indices, 1], c='red', zorder=1, label=None, marker='.',s=point_size_interest)
h2 = ax[idw_Y][idw_X].scatter(letters_y_idw10[:shown_letter_indices, 0],
                letters_y_idw10[:shown_letter_indices, 1], c='blue', zorder=1, label=None, marker='.',s=point_size_interest)
h3 = ax[idw_Y][idw_X].scatter(letters_y_idw20[:shown_letter_indices, 0],
                letters_y_idw20[:shown_letter_indices, 1], c='green', zorder=1, label=None, marker='.',s=point_size_interest)
h4 = ax[idw_Y][idw_X].scatter(letters_y_idw_optimal[:shown_letter_indices, 0],
                letters_y_idw_optimal[:shown_letter_indices, 1], c='purple', zorder=1, label=None, marker='.',
                              s=point_size_interest)
h5 = ax[idw_Y][idw_X].scatter(letters_y_idw40[:shown_letter_indices, 0],
                letters_y_idw40[:shown_letter_indices, 1], c='cyan', zorder=1, label=None, marker='.',s=point_size_interest)
ax[idw_Y][idw_X].legend([h1,h2,h3,h4,h5], idw_method_list, ncol=1, prop=font_properties, borderpad=0.1,handlelength=2,
                       columnspacing = 0, loc = 1, handletextpad=-0.7,frameon=True)

# ====================================  GD =======================================
cur_shown_indices = chosen_indices
ax[gd_Y][gd_X].scatter(Y_mnist[:, 0], Y_mnist[:, 1], c= 'gray', zorder=1, label=None, marker='.',s=point_size_gray)
#legend_list.append(str(l))
h2 = ax[gd_Y][gd_X].scatter(letters_y_gd_transformed[cur_shown_indices,0],
                letters_y_gd_transformed[cur_shown_indices,1], c='blue', marker='.',zorder=3,alpha=0.9,s=point_size_interest)
h3 = ax[gd_Y][gd_X].scatter(letters_y_gd_transformed_random[cur_shown_indices,0],
                letters_y_gd_transformed_random[cur_shown_indices,1], c='cyan', marker='.', zorder=3,alpha=0.9,
                            s=point_size_interest)
#h4 = ax.scatter(letters_y_gd_variance_recalc_transformed[:cur_shown_indices,0],
#                letters_y_gd_variance_recalc_transformed[:cur_shown_indices,1], c='green', marker='.', zorder=3,alpha=0.9)
#h5 = ax.scatter(letters_y_gd_variance_recalc_transformed_random[:cur_shown_indices,0],
#                letters_y_gd_variance_recalc_transformed_random[:cur_shown_indices,1],
#                c='brown', marker='.', zorder=3,alpha=0.9)
h6 = ax[gd_Y][gd_X].scatter(letters_y_gd_early_exagg_transformed[cur_shown_indices,0],
                letters_y_gd_early_exagg_transformed[cur_shown_indices,1], c='green', marker='.',zorder=3,alpha=0.9,
                           s=point_size_interest)
h7 = ax[gd_Y][gd_X].scatter(letters_y_gd_early_exagg_transformed_random[cur_shown_indices,0],
                letters_y_gd_early_exagg_transformed_random[cur_shown_indices,1],
                c='red', marker='.', zorder=3,alpha=0.9,s=point_size_interest)
#h8 = ax.scatter(letters_y_gd_variance_recalc_early_exagg_transformed[:cur_shown_indices,0],
#                letters_y_gd_variance_recalc_early_exagg_transformed[:cur_shown_indices,1],
#                c='black', marker='.', zorder=3,alpha=0.9)
#h9 = ax.scatter(letters_y_gd_variance_recalc_early_exagg_transformed_random[:cur_shown_indices,0],
#                letters_y_gd_variance_recalc_early_exagg_transformed_random[:cur_shown_indices,1],
#                c='olive', marker='.', zorder=3,alpha=0.9)
#h10 = ax.scatter(letters_random_starting_positions[cur_shown_indices,0],
#                 letters_random_starting_positions[cur_shown_indices,1], c='black', marker='.', zorder=3,alpha=0.9)

ax[gd_Y][gd_X].legend([h2,h3,h6,h7], [          r'Closest Y; no EE',
                                        r'Random Y; no EE',
                                        #r'Closest Y; new $\sigma$; no EE',
                                        #r'Random Y; new $\sigma$; no EE',
                                        r'Closest Y; EE',
                                        r'Random Y; EE',
                                        #r'Closest Y; new $\sigma$; EE',
                                        #r'Random Y; new $\sigma$; EE',
                                        #'Random Starting Position'
                             ], ncol=1, prop=font_properties, borderpad=0.1,handlelength=2,
                       columnspacing = 0, loc = 1, handletextpad=-0.7,frameon=True)

# ==================================== NN ========================================

ax[nn_Y][nn_X].scatter(Y_mnist[:, 0], Y_mnist[:, 1], c= 'gray', zorder=1, label=None, marker='.',s=point_size_gray)
h1 = ax[nn_Y][nn_X].scatter(nn_letters_results[0][:cur_shown_letter_indices, 0],
                nn_letters_results[0][:cur_shown_letter_indices, 1], c='red', zorder=1, label=None, marker='.',
                            s=point_size_interest)
h2 = ax[nn_Y][nn_X].scatter(nn_letters_results[1][:cur_shown_letter_indices, 0],
                nn_letters_results[1][:cur_shown_letter_indices, 1], c='blue', zorder=1, label=None, marker='.',
                           s = point_size_interest)
h3 = ax[nn_Y][nn_X].scatter(nn_letters_results[2][:cur_shown_letter_indices, 0],
                nn_letters_results[2][:cur_shown_letter_indices, 1], c='green', zorder=1, label=None, marker='.',
                           s=point_size_interest)
ax[nn_Y][nn_X].legend([h1,h2,h3], nn_method_list, ncol=1, prop=font_properties, borderpad=0.1,handlelength=2,
                       columnspacing = 0, loc = 1, handletextpad=-0.7,frameon=True)

# ==================================== kTSNE =====================================

ax[ktsne_Y][ktsne_X].scatter(Y_mnist[:, 0], Y_mnist[:, 1], c= 'gray', zorder=1, label=None, marker='.',
                            s = point_size_gray)
h1 = ax[ktsne_Y][ktsne_X].scatter(kernelized_tsne_letters_results[0][:shown_letter_indices, 0],
                kernelized_tsne_letters_results[0][:shown_letter_indices, 1], c='red', zorder=1, label=None, marker='.',
                                 s = point_size_interest)
h2 = ax[ktsne_Y][ktsne_X].scatter(kernelized_tsne_letters_results[1][:shown_letter_indices, 0],
                kernelized_tsne_letters_results[1][:shown_letter_indices, 1], c='blue', zorder=1, label=None, marker='.',
                                 s = point_size_interest)
h3 = ax[ktsne_Y][ktsne_X].scatter(kernelized_tsne_letters_results[2][:shown_letter_indices, 0],
                kernelized_tsne_letters_results[2][:shown_letter_indices, 1], c='green', zorder=1, label=None, marker='.',
                                 s = point_size_interest)
ax[ktsne_Y][ktsne_X].legend([h1,h2,h3], kernelized_tsne_method_list, ncol=1, prop=font_properties, borderpad=0.1,handlelength=2,
                       columnspacing = 0, loc = 1, handletextpad=-0.7,frameon=True)

# ==================================== LION ======================================

ax[lion_Y][lion_X].scatter(Y_mnist[:, 0], Y_mnist[:, 1], c= 'gray', zorder=1, label=None, marker='.',
                          s = point_size_gray)
h1 = ax[lion_Y][lion_X].scatter(letters_y_lion90[:cur_shown_letter_indices, 0],
                letters_y_lion90[:cur_shown_letter_indices, 1], c='red', zorder=1, label=None, marker='.',
                               s = point_size_interest)
h2 = ax[lion_Y][lion_X].scatter(letters_y_lion95[:cur_shown_letter_indices, 0],
                letters_y_lion95[:cur_shown_letter_indices, 1], c='blue', zorder=1, label=None, marker='.',
                               s = point_size_interest)
h3 = ax[lion_Y][lion_X].scatter(letters_y_lion99[:cur_shown_letter_indices, 0],
                letters_y_lion99[:cur_shown_letter_indices, 1], c='green', zorder=1, label=None, marker='.',
                               s = point_size_interest)
h4 = ax[lion_Y][lion_X].scatter(letters_y_lion100[:cur_shown_letter_indices, 0],
                letters_y_lion100[:cur_shown_letter_indices, 1], c='purple', zorder=1, label=None, marker='.',
                               s = point_size_interest)
ax[lion_Y][lion_X].legend([h1,h2,h3,h4], lion_method_list, ncol=1, prop=font_properties, borderpad=0.1,handlelength=2,
                       columnspacing = 0, loc = 1, handletextpad=-0.7,frameon=True)

# ====================================  Ending ===================================

ax[rbf_X][rbf_Y].set_xlim(ax[idw_Y][idw_X].get_xlim())
ax[rbf_X][rbf_Y].set_ylim(ax[idw_Y][idw_X].get_ylim())

ax[rbf_Y][rbf_X].text(-140,115,"(a) RBF Interpolation",fontsize=10)
ax[idw_Y][idw_X].text(-140,115,"(b) IDW Interpolation",fontsize=10)
ax[gd_Y][gd_X].text(-140,115,"(c) Gradient Descent",fontsize=10)
ax[nn_Y][nn_X].text(-140,115,"(d) Neural Networks",fontsize=10)
ax[ktsne_Y][ktsne_X].text(-140,115,"(e) Kernelized tSNE",fontsize=10)
ax[lion_Y][lion_X].text(-140,115,"(f) LION tSNE",fontsize=10)

plt.tight_layout()
plt.subplots_adjust(wspace=None, hspace=None, left=0.003, right=0.997, top=0.997, bottom=0.003)
plt.savefig("../figures/letter-test-all.png")
