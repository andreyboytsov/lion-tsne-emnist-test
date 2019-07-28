# TODO rework much later. This plot is not affected by PCA problem.

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pickle
import numpy as np

with open('precision_plots.p', 'rb') as f:
    accuracy_plot, precision_30_plot, precision_50_plot, lion_extended_percentile_options, power = pickle.load(f)

ptsne_precision = 0.10817 # Calculated elsewhere
baseline_extended_precision = 0.124845 # Calculated elsewhere

legend_list = list()
plt.figure(dpi=300)
plt.gcf().set_size_inches(3.3,2)

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

color_dict = {90:'blue', 95:'green',99:'red',100:'cyan'}

legend_lines = list()

print(lion_extended_percentile_options)

selected_power = {90: 13.8, 95 : 16.5, 99 : 21.1, 100 : 26.6}

p_50_index = np.where([np.abs(i - 50)<0.00001 for i in power])[0][0]

for perc in sorted(lion_extended_percentile_options):
    print("=========================== ", perc, " =============================")
    p_index = np.where([np.abs(i - selected_power[perc])<0.00001 for i in power])[0][0]
    p_max_index = np.argmax(precision_30_plot[perc])
    print(p_index, p_max_index)
    print(power[p_index], precision_30_plot[perc][p_index])
    print(power[p_max_index], precision_30_plot[perc][p_max_index])
    print(precision_30_plot[perc][p_index] / precision_30_plot[perc][p_max_index])
    print(power[p_50_index], precision_30_plot[perc][p_50_index])

for perc in sorted(lion_extended_percentile_options):
    legend_list.append(r"LION: $r_x$ = "+str(perc)+"th NN perc-le")
    a, = plt.plot(power, precision_30_plot[perc], c=color_dict[int(perc)])
    legend_lines.append(a)
plt.legend(legend_list)
h1 = plt.axhline(ptsne_precision, linestyle='--', c='black')
h2 = plt.axhline(baseline_extended_precision, linestyle=':', c='black')
plt.xlim([0,50])
plt.ylim([0.05, 0.13])
plt.xlabel("LION-tSNE: Power",fontproperties=font_properties)
plt.ylabel("30-NN Precision",fontproperties=font_properties)

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)

l = plt.legend([h1,h2]+legend_lines,
           ["PTSNE Precision (%.4f)"%ptsne_precision, "Baseline Precision (%.4f)"%baseline_extended_precision]+legend_list,
          bbox_to_anchor=[0.53,-1.0],loc=8,prop=font_properties)
#plt.tight_layout(rect=[-0.04,-0.04,1.04,1.60])
plt.savefig("./LION-and-ptsne-power-vs-precision.png", bbox_extra_artists=[l],bbox_inches='tight')
#plt.show()