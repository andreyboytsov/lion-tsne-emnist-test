import matplotlib.pyplot as plt
import generate_data
from matplotlib.font_manager import FontProperties

labels_mnist = generate_data.load_labels_mnist()
Y_mnist= generate_data.load_y_mnist()

plt.figure(dpi=300)
font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(9)

plt.xlim([-180, 180])
plt.ylim([-150, 170])

plt.gcf().set_size_inches(3.3,3.3) #Let's set the plot sizes that just fit paper margins
legend_list = list()
for l in set(sorted(labels_mnist)):
    plt.scatter(Y_mnist[labels_mnist == l, 0], Y_mnist[labels_mnist == l, 1], marker = '.', s=5)
    legend_list.append(str(l))
#plt.title("MNIST Dataset - TSNE visualization")
#plt.tight_layout()
l = plt.legend(legend_list, bbox_to_anchor=(0.99, 1.025), markerscale=8, prop=font_properties)
#plt.tight_layout(rect=[-0.17, -0.1, 1.03, 1.03])

#plt.tick_params(axis='both', which='both',bottom=False,top=False,labelbottom=False,
#                                          left=False,right=False,labelleft=False)

plt.show()