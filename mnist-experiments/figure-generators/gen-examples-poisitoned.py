import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import generate_data
import settings

# Loading the data
shown_indices = 10
parameters=settings.parameters
Y_mnist= generate_data.load_y_mnist(parameters=parameters)
picked_indices = generate_data.load_nearest_training_indices(parameters=parameters)
picked_indices_y_mnist = Y_mnist[picked_indices,:]

# Plot with examples
legend_list = list()
f, ax = plt.subplots(dpi=300)
f.set_size_inches(3.3,3.3)

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

plots_for_legend_list = list()
ax.scatter(Y_mnist[:, 0], Y_mnist[:, 1], c= 'gray', zorder=1, label=None, marker='.',s=10)

color_list = ['blue','red','green','orange','magenta','black','brown','darkcyan','crimson','navy']

for i in range(shown_indices):
    h = ax.scatter(picked_indices_y_mnist[i,0],
                   picked_indices_y_mnist[i,1], marker='X', c=color_list[i], s=60, zorder=3)
    legend_list.append(str(i+1))
    #c = plt.Circle((picked_indices_y_mnist[l,0], picked_indices_y_mnist[l,1]), r, color='black', fill=False,zorder=4)
    #ax.add_artist(c)
    plots_for_legend_list.append(h)

ax.legend(plots_for_legend_list, legend_list, prop=font_properties,loc=4)
plt.tick_params(axis='both', which='both',bottom=False,top=False,labelbottom=False,
                                          left=False,right=False,labelleft=False)

f.tight_layout(rect=[-0.04,-0.04,1.04,1.04])
plt.savefig("../figures/examples-positioned.png")
# plt.show(f)