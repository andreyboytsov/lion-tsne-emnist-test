import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import generate_data
import settings
import exp_lion_power_performance
import exp_idw_power_performance
import numpy as np

parameters = settings.parameters
lion_power_plot_data = exp_lion_power_performance.load_lion_power_performance(parameters=parameters)
idw_power_plot_data = exp_idw_power_performance.load_idw_power_performance(parameters=parameters)[-1]


def get_nearest_neighbors(y, Y_mnist, n):
    y_distances = np.sum((Y_mnist - y) ** 2, axis=1)
    return np.argsort(y_distances)[:n]


dTSNE_mnist = generate_data.load_dtsne_mnist()


def get_baseline_precision(X, Y, precision_nn = 50):
    per_sample_precision = list()
    for i in range(len(X)):
        x = X[i, :]
        y = Y[i, :]
        nn_x_indices = get_nearest_neighbors(x, X, n=precision_nn+1) # +1 to account for "itself"
        nn_y_indices = get_nearest_neighbors(y, Y, n=precision_nn+1) # +1 to account for "itself"
        matching_indices = len([j for j in nn_x_indices if j in nn_y_indices and j != i])
        per_sample_precision.append(matching_indices / precision_nn)
    return np.mean(per_sample_precision)

baseline_precision = get_baseline_precision(dTSNE_mnist.X, dTSNE_mnist.Y)


# Accuracy-vs-power plot - zoomed
legend_list = list()
f, ax = plt.subplots(dpi=300)
f.set_size_inches(3.3,2)

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

power_perc_combos = lion_power_plot_data.keys()
all_percentages = set([int(i.split(";")[0]) for i in power_perc_combos])
x = sorted(set([float(i.split(";")[1]) for i in power_perc_combos]))
color_dict = {90:'blue', 95:'green',99:'red',100:'cyan'}
legend_list = list()
legend_lines = list()
print(all_percentages)
for perc in sorted(all_percentages):
    y = list()
    for cur_power in x:
        key = "%d;%.3f"%(perc,cur_power)
        #print(key)
        y.append(lion_power_plot_data[key]["Accuracy"])
    h, = plt.plot(x,y, c=color_dict[perc])
    legend_lines.append(h)
    legend_list.append("Radius in X: "+str(perc)+" NN percentile")
#for i in lion_optimal_power:
#    plt.axvline(x=lion_optimal_power[i], c = 'black', linestyle='--')
h = plt.axhline(y=baseline_precision, c = 'black', linestyle='--')
x = sorted(idw_power_plot_data.keys())
y = [idw_power_plot_data[p] for p in x]
h2, = plt.plot(x,y, c='blue',linestyle=':')
plt.xlabel("LION-tSNE: Power",fontproperties=font_properties)
plt.ylabel("50-NN Precision",fontproperties=font_properties)
plt.ylim([0.40,0.45])
plt.xlim([20,120]) # TODO think of it

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)

f.tight_layout(rect=[-0.04,-0.08,1.04,1.06])
plt.savefig("../figures/LION-and-IDW-power-vs-precision-zoomed.png")