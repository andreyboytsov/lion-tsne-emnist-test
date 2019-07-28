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

baseline_precision = generate_data.load_baseline_precision(parameters=parameters)

# Accuracy-vs-power plot
legend_list = list()
f, ax = plt.subplots(dpi=300)
f.set_size_inches(3.3,2)

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

power_perc_combos = lion_power_plot_data.keys()
all_percentages = sorted(set([int(i.split(';')[0]) for i in power_perc_combos]))
x = sorted(set([float(i.split(';')[1]) for i in power_perc_combos]))
color_dict = {90:'blue', 95:'green',99:'red',100:'cyan'}
legend_list = list()
legend_lines = list()
print(all_percentages)
for perc in all_percentages:
    y = list()
    for cur_power in x:
        key = "%d;%.3f"%(perc,cur_power)
        y.append(lion_power_plot_data[key]["Precision"])
    h, = plt.plot(x,y, c=color_dict[perc])
    legend_lines.append(h)
    legend_list.append(r"LION: $r_x$ = "+str(perc)+"th NN perc-le")
#plt.title("IDW - Accuracy vs Power") # We'd better use figure caption
#ax.legend([h1,h2,h3,h4,h5,h6], ["Closest Training Set Image"]+idw_method_list)
h = plt.axhline(y=baseline_precision, c = 'black', linestyle='--')
x = sorted(idw_power_plot_data.keys())
y = [idw_power_plot_data[p] for p in x]
h2, = plt.plot(x,y, c='blue',linestyle=':')
plt.legend([h,h2]+legend_lines,["Baseline Precision (%.4f)"%baseline_precision, "IDW interpolation"]+legend_list,
          bbox_to_anchor=[1.01,0.705],prop=font_properties)
plt.xlabel("LION-tSNE: Power",fontproperties=font_properties)
#plt.ylabel("10-NN Accuracy",fontproperties=font_properties)
plt.ylabel("50-NN Precision",fontproperties=font_properties)

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)

plt.ylim([0,0.5])
plt.xlim([0,120])
f.tight_layout(rect=[-0.04,-0.08,1.04,1.06])
plt.savefig("../figures/LION-and-IDW-power-vs-precision.png")
