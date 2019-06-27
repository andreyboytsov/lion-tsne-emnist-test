import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
import generate_data
import settings

parameters = settings.parameters
X_mnist_raw = generate_data.load_x_mnist_raw(parameters=parameters)
picked_indices = generate_data.load_nearest_training_indices(parameters=parameters)
picked_neighbors_raw = generate_data.load_picked_neighbors_raw(parameters=parameters)

width = 10  # total number to show
start_index = 0

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

gs = gridspec.GridSpec(3, width + 1, width_ratios=[4] + [1] * width, height_ratios=[0.5, 1, 1])

f, ax_total = plt.subplots(7, width + 1, dpi=300)
f.set_size_inches(3.1, 0.6)
f.subplots_adjust()
for i in range(width):
    ax = [plt.subplot(gs[i + 1]), plt.subplot(gs[width + i + 2]), plt.subplot(gs[2 * width + i + 3])]
    ax[0].text(text=str(i + 1), x=0.5, y=0.4, s=11, ha='center', va='center', fontproperties=font_properties)
    ax[1].imshow(picked_neighbors_raw[i, :].reshape(28, 28), cmap='gray_r')
    ax[2].imshow(X_mnist_raw[picked_indices[i], :].reshape(28, 28), cmap='gray_r')

    ax[0].set_axis_off()
    # ax[0].axes.get_xaxis().set_visible(False)
    # ax[0].axes.get_yaxis().set_visible(False)
    # Set_axis_off does not fit. I want a bounding box.
    ax[1].axes.get_xaxis().set_visible(False)
    ax[2].axes.get_xaxis().set_visible(False)
    ax[1].axes.get_yaxis().set_visible(False)
    ax[2].axes.get_yaxis().set_visible(False)
plt.subplot(gs[0]).set_axis_off()
plt.subplot(gs[0]).text(text="№   ", x=1.0, y=0.33, s=11, ha='right', fontproperties=font_properties)
plt.subplot(gs[width + 1]).set_axis_off()
plt.subplot(gs[width + 1]).text(text="Image $X$  ", x=1.0, y=0.4, s=11, ha='right', fontproperties=font_properties)
plt.subplot(gs[2 * width + 2]).set_axis_off()
plt.subplot(gs[2 * width + 2]).text(text="Closest $X_i$  ", x=1.0, y=0.4, s=11, ha='right',
                                    fontproperties=font_properties)
# gs.tight_layout(f)
# gs.update(wspace=0.1, hspace=0.025)
# f.subplots_adjust(left=-0.155, right=0.99, top=1.7,bottom=-0.5,wspace=0.1, hspace=-0.75)
f.subplots_adjust(wspace=0, hspace=0, left=-0.173, right=0.99, top=1.035, bottom=0.015)

f.savefig("../figures/nearest_neighbors_hor.png", bbox='tight')