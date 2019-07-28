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

# Accuracy-vs-power plot - zoomed
legend_list = list()
f, ax = plt.subplots(dpi=300)
f.set_size_inches(3.3,2)

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

selected_power = {90: 104.8, 95 : 75.8, 99 : 50.6, 100 : 49.8}
selected_idw_power = 44.9

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
        y.append(lion_power_plot_data[key]["Precision"])
    h, = plt.plot(x,y, c=color_dict[perc])
    legend_lines.append(h)
    legend_list.append("Radius in X: "+str(perc)+" NN percentile")
    
    p_index = np.where([np.abs(i - selected_power[perc])<0.00001 for i in x])[0][0]
    p_max_index = np.argmax(y)
    print(perc, x[p_max_index], y[p_max_index], x[p_index], y[p_index], y[p_index]/y[p_max_index])

#for i in lion_optimal_power:
#    plt.axvline(x=lion_optimal_power[i], c = 'black', linestyle='--')
h = plt.axhline(y=baseline_precision, c = 'black', linestyle='--')
x = sorted(idw_power_plot_data.keys())
y = [idw_power_plot_data[p] for p in x]
h2, = plt.plot(x,y, c='blue',linestyle=':')
plt.xlabel("LION-tSNE: Power",fontproperties=font_properties)
plt.ylabel("50-NN Precision",fontproperties=font_properties)
plt.ylim([0.44,0.46])
plt.xlim([0,120]) # TODO think of it

p_max_index = np.argmax(y)
p_index = np.where([np.abs(i - selected_idw_power) < 0.00001 for i in x])[0][0]
print("IDW", x[p_max_index], y[p_max_index], x[p_index], y[p_index], y[p_index]/y[p_max_index])

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)

print(lion_power_plot_data["%d;%.3f"%(90,120)]['Precision'], idw_power_plot_data[x[-1]], x[-1],
      idw_power_plot_data[x[-1]]-lion_power_plot_data["%d;%.3f"%(90,120)]['Precision'])

f.tight_layout(rect=[-0.04,-0.08,1.04,1.06])
plt.savefig("../figures/LION-and-IDW-power-vs-precision-zoomed.png")