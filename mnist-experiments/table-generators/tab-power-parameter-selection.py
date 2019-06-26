import exp_idw_power_performance
import exp_lion_power_performance
import settings
import pickle
import math

tab_text_file = '../tables/tab-power-parameter-selection.txt'

parameters = settings.parameters
lion_power_plot_data_file = exp_lion_power_performance.generate_lion_power_plot_filename(parameters)
idw_power_plot_data_file = exp_idw_power_performance.generate_idw_power_plot_filename(parameters)

with open(lion_power_plot_data_file, 'rb') as f:
    _, _, lion_optimal_power = pickle.load(f)

with open(idw_power_plot_data_file, 'rb') as f:
    _, _, idw_optimal_power = pickle.load(f)

#width_share = math.floor(100*0.4 / (len(lion_optimal_power)))/100
width_share = 0.03

s = ""
s += '\t\\begin{table} \caption{Power parameter selection}  \label{tab_power_selection}\n'
s += '\t\t\\begin{tabular}{| M{0.09\\textwidth} '+ ('| M{%.2f\\textwidth}'%width_share)*(len(lion_optimal_power)) + '| M{0.1\\textwidth} |}\n'
s += '\t\t\t\\hline\n'
s += '\t\t\t \\textbf{$r_x$ percentile} &'+ \
      ''.join(['%d & '%i for i in sorted(lion_optimal_power)])+' Non-local IDW\n'
s += '\t\t\t\\\\ \\hline\n'
s += '\t\t\t \\textbf{Selected $p$} &'+\
      ''.join(['%.1f &'%lion_optimal_power[i] for i in sorted(lion_optimal_power)])+\
      '%.1f'%idw_optimal_power+"\n"
s += '\t\\\\ \\hline\n'
s += '\t\t\\end{tabular}\n'
s += '\t\\end{table}\n'

with open(tab_text_file, 'wt') as f:
    f.write(s)

print(s)
