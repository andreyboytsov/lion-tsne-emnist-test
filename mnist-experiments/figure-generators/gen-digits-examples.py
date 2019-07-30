import matplotlib.pyplot as plt
import generate_data
import settings
import logging

logging.basicConfig(level=logging.INFO)

nn_samples_raw = generate_data.load_picked_neighbors_raw(parameters=settings.parameters)

width = 10 #total number to show
height = 1
start_index = 0

f, ax = plt.subplots(height,width,dpi=300)
f.set_size_inches(3.3,0.33) # 3.3, 1 - 3 rows, 3.3, 0.66 - 2 rows, 3.3, 0.33 - 1 row
f.subplots_adjust()
#f.tight_layout()
if height > 1:
    for i in range(height):
        for j in range(width):
            ax[i,j].imshow(nn_samples_raw[i*width+j,:].reshape(28,28), cmap='gray_r')
            #Set_axis_off does not fit. I want a bounding box.
            ax[i,j].axes.get_xaxis().set_visible(False)
            ax[i,j].axes.get_yaxis().set_visible(False)
else:
    for j in range(width):
        ax[j].imshow(nn_samples_raw[j, :].reshape(28, 28), cmap='gray_r')
        # Set_axis_off does not fit. I want a bounding box.
        ax[j].axes.get_xaxis().set_visible(False)
        ax[j].axes.get_yaxis().set_visible(False)
#gs.update(wspace=0.1, hspace=0.025)
f.subplots_adjust(left=0.005, right=0.995, top=0.975,bottom=0.025)

f.savefig("../figures/digits_examples.png")