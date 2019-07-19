import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import generate_data
import settings

# CAREFUL: It is not just a plot, it also searches optimal power parameter.
lion_extended_optimal_power = dict()

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(9)

legend_list = list()
f, ax = plt.subplots()
f.set_size_inches(3.3,3.3)
f.set_dpi(300)
power_perc_combos = lion_extended_power_plot_data.keys()
#print(power_perc_combos)
all_percentages = sorted(lion_extended_percentile_options)
x = sorted(lion_extended_power_options)
color_dict = {90:'blue', 95:'green',99:'red',100:'cyan'}
legend_list = list()
legend_lines = list()
#print(all_percentages)
#plot_indices = {90: [0,0], 95: [0,1], 99: [1,0], 100: [1,1]}
#ylims = {90: [1000,2000], 95:[100,700], 99:[100,500], 100:[20,80]}
for perc in all_percentages:
#    cur_subplot = ax[plot_indices[perc][0], plot_indices[perc][1]]
    y = list()
#    y_norandom = list()
    for cur_power in x:
        key = str(perc)+";%.3f"%(cur_power)
#        #print(cur_power, perc, lion_power_plot_data[key])
        y.append(lion_extended_validation_plot_data[key])
    h, = ax.plot(x,y, c=color_dict[perc])
    chosen_power = x[np.argmin(y)]
    chosen_acc = lion_extended_accuracy_plot_data[str(perc)+";%.3f"%(chosen_power)]
    print(perc,chosen_power, chosen_acc)
#    #print(perc, x[np.argmin(y)])
#    lion_optimal_power[perc] = x[np.argmin(y)]
    legend_lines.append(h)
#    cur_subplot.set_xlim([5,100])
#    #cur_subplot.set_ylim(ylims[perc])
    legend_list.append("$r_x$: "+str(perc)+" NN percentile")
#    if plot_indices[perc][0]==1:
#        cur_subplot.set_xlabel("LION-tSNE: Power")
#    if plot_indices[perc][1]==0:
#        cur_subplot.set_ylabel("Average Square Distance")
#print("Non-Local", x_global[np.argmin(y)])

ax.set_ylim([0,100])
ax.set_xlim([0,100])

for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)

#plt.title("IDW - Accuracy vs Power") # We'd better use figure caption
#ax.legend([h1,h2,h3,h4,h5,h6], ["Closest Training Set Image"]+idw_method_list)
#h = plt.axhline(y=baseline_accuracy, c = 'black', linestyle='--')
#l = plt.legend(legend_lines,legend_list, bbox_to_anchor=[0.95,-0.2], ncol=2)
#f.legend(legend_lines,legend_list, ncol=2, bbox_to_anchor=[0.955,0.94], prop=font_properties)
f.legend(legend_lines,legend_list, ncol=1, loc=1, prop=font_properties, bbox_to_anchor=[0.925,0.94])
#plt.ylim([725,760])
#plt.xlim([10,40])
f.tight_layout()
f.savefig("Figures/LION-extended-power-training-set.png",bbox_extra_artists=[l],bbox_inches='tight')
plt.show(f)