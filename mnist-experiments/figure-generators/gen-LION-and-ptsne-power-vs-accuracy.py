# TODO rework much later. This plot is not affected by PCA problem.

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import generate_data
import settings

legend_list = list()
plt.gcf().set_size_inches(3.3,2)
plt.gcf().set_dpi(300)

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

color_dict = {90:'blue', 95:'green',99:'red',100:'cyan'}
for perc in sorted(lion_extended_percentile_options):
    x = list()
    y = list()
    for p in lion_extended_power_options:
        key = str(perc)+";"+"%.3f"%(p)
        x.append(p)
        y.append(lion_extended_accuracy_plot_data[key])
    legend_list.append(r"LION: $r_x$ = "+str(perc)+"th NN perc-le (%.3f)"%radius_extended_x[perc])
    a, = plt.plot(x,y, c=color_dict[perc])
    legend_lines.append(a)
plt.legend(legend_list)
h1 = plt.axhline(ptsne_accuracy, linestyle='--', c='black')
h2 = plt.axhline(baseline_extended_accuracy, linestyle=':', c='black')
plt.xlim([0,200])
plt.ylim([0.78, 0.92])
plt.xlabel("LION-tSNE: Power",fontproperties=font_properties)
plt.ylabel("10-NN Accuracy",fontproperties=font_properties)

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)

l = plt.legend([h1,h2]+legend_lines,
           ["PTSNE Accuracy (%.4f)"%ptsne_accuracy, "Baseline Accuracy (%.4f)"%baseline_extended_accuracy]+legend_list,
          bbox_to_anchor=[0.53,-1.0],loc=8,prop=font_properties)
#plt.tight_layout(rect=[-0.04,-0.04,1.04,1.60])
plt.savefig("Figures/LION-and-ptsne-power-vs-accuracy.png", bbox_extra_artists=[l],bbox_inches='tight')
plt.show()