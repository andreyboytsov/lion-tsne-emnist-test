import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import exp_lion_power_performance
import exp_idw_power_performance
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

lion_optimal_power = dict()

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

legend_list = list()
plt.figure(dpi=300)
plt.gcf().set_size_inches(3.3, 3.3)

lion_power_plot_data = exp_lion_power_performance.load_lion_power_performance()
global_idw_power_performance, global_idw_power_performance_abs, _ = exp_idw_power_performance.load_idw_power_performance()
x_global, y_global, _ = exp_idw_power_performance.load_idw_power_plot()
x, y, _ = exp_lion_power_performance.load_lion_power_plot()

power_perc_combos = lion_power_plot_data.keys()
# print(power_perc_combos)
all_percentages = set([int(i.split(";")[0]) for i in power_perc_combos])
#x = sorted(set([float(i.split(";")[1]) for i in power_perc_combos]))

color_dict = {90: 'blue', 95: 'green', 99: 'red', 100: 'cyan'}
legend_list = list()
legend_lines = list()
# print(all_percentages)

for perc in all_percentages:
    h, = plt.plot(x, y[perc], c=color_dict[perc])
    legend_lines.append(h)
    legend_list.append("$r_x$: " + str(perc) + " NN percentile")
    print(h, x[np.argmin(y[perc])])

h, = plt.plot(x_global, y_global, c='purple', linestyle=":")
legend_lines.append(h)
legend_list.append("Non-local IDW")

# plt.title("IDW - Accuracy vs Power") # We'd better use figure caption
# ax.legend([h1,h2,h3,h4,h5,h6], ["Closest Training Set Image"]+idw_method_list)
# h = plt.axhline(y=baseline_accuracy, c = 'black', linestyle='--')

l = plt.legend(legend_lines, legend_list, bbox_to_anchor=[1.00, -0.15], ncol=2, prop=font_properties)
plt.xlabel("LION-tSNE: Power", fontproperties=font_properties)
plt.ylabel("Average Square Distance", fontproperties=font_properties)

plt.ylim([90, 800])
plt.xlim([0, 120])

# f.tight_layout()

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)

plt.tight_layout(rect=[-0.035, -0.045, 1.044, 1.042])

plt.savefig("../figures/LION-power-training-set.png")
#plt.show(f)