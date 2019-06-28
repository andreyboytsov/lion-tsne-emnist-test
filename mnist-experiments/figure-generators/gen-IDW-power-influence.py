import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import lion_tsne

# This is 1-D illustration of "pull to the center" effect. Plot will appear in the discussion section
x_1d = np.array([[10],[20],[30],[40]])
y_1d = np.array([[10],[40],[1],[50]])
simple_example_model = lion_tsne.LionTSNE(perplexity=2)
simple_example_model.incorporate(x_1d, y_1d)

x = np.arange(0,50,0.1).reshape((-1,1))

powers_and_colors = {0.2 : 'blue', 2 : 'red', 20 : 'green'}
legend_by_p = {0.2 : r'Low $p$', 2 : r'Medium $p$', 20 : r'High $p$'}

plt.figure(dpi=300)
plt.gcf().set_size_inches(3.3,2)

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

legend_list = list()
for p in powers_and_colors:
    interpolator = simple_example_model.generate_embedding_function(embedding_function_type='weighted-inverse-distance',
                                                                   function_kwargs={'power' : p})
    y_weighted = interpolator(x)
    plt.plot(x,y_weighted)
    legend_list.append(legend_by_p[p]) #To ensure proper order
plt.ylim([0,55])
plt.xlim([0,50])
plt.axhline(y=np.mean(y_1d), c = 'black', linestyle='--')
plt.tight_layout()
plt.legend(legend_list+["Average Y"],prop=font_properties,ncol=2,loc=2,bbox_to_anchor=[-0.03,1.035],
           frameon=False,columnspacing=1)
plt.xlabel("X",fontproperties=font_properties)
plt.ylabel("Y",fontproperties=font_properties)

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontproperties(font_properties)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)

plt.scatter(x_1d,y_1d,c='red',marker='x',zorder=3)
plt.tight_layout(rect=[-0.035,-0.08,1.045,1.07])
plt.savefig("../figures/IDW-power-influence.png")