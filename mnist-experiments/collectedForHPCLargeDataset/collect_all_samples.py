import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

first_sample = True

accuracy = dict()
precision_30 = dict()
precision_50 = dict()

power_list = set() # Will transform later
lion_perc_set = set()

for i in range(10000):
    lion_extended_accuracy_plot_data_file = './lion-processed-samples/lion_extended_accuracy_plot_data_' + str(i) + '.p'
    if os.path.isfile(lion_extended_accuracy_plot_data_file):
        with open(lion_extended_accuracy_plot_data_file, "rb") as f:
            y_result, cur_accuracy, cur_precision_30, cur_precision_50 = pickle.load(f)

        if first_sample:
            for k in cur_accuracy.keys():
                power_list.add(float(k.split(';')[1]))
                lion_perc_set.add(k.split(';')[0])
                accuracy[k] = list()
                precision_30[k] = list()
                precision_50[k] = list()
            power_list = sorted(list(power_list))

        for k in cur_accuracy.keys():
            accuracy[k].append(cur_accuracy[k])
            precision_30[k].append(cur_precision_30[k])
            precision_50[k].append(cur_precision_50[k])

        first_sample = False
    else:
        print("WARNING! Sample ", i, "missing")

print(" ============ SAMPLES LOADED =============")
print('Starting the summary')

print(power_list)
print(lion_perc_set)

accuracy_plot = dict()
precision_30_plot = dict()
precision_50_plot = dict()

for lp in lion_perc_set:
    accuracy_plot[lp] = list()
    precision_30_plot[lp] = list()
    precision_50_plot[lp] = list()
    for p in power_list:
        accuracy_plot[lp].append(np.mean(accuracy[lp+";"+"%.3f" % (p)]))
        precision_30_plot[lp].append(np.mean(precision_30[lp+";"+"%.3f" % (p)]))
        precision_50_plot[lp].append(np.mean(precision_50[lp+";"+"%.3f" % (p)]))


plt.gcf().set_size_inches(8,8)
plt.title("Accuracy")
#chosen_ptsne = 2
legend_list = list()

for lp in lion_perc_set:
    plt.plot(power_list, accuracy_plot[lp])
    legend_list.append(str(lp))
#plt.title("MNIST Dataset - TSNE visualization")
#plt.tight_layout()
plt.legend(legend_list)
plt.show()


plt.gcf().set_size_inches(8,8)
plt.title("Precision-30")
#chosen_ptsne = 2
legend_list = list()

for lp in lion_perc_set:
    plt.plot(power_list, precision_30_plot[lp])
    legend_list.append(str(lp))
#plt.title("MNIST Dataset - TSNE visualization")
#plt.tight_layout()
plt.legend(legend_list)
plt.show()


plt.gcf().set_size_inches(8,8)
plt.title("Precision-50")
#chosen_ptsne = 2
legend_list = list()

for lp in lion_perc_set:
    plt.plot(power_list, precision_50_plot[lp])
    legend_list.append(str(lp))
#plt.title("MNIST Dataset - TSNE visualization")
#plt.tight_layout()
plt.legend(legend_list)
plt.show()
