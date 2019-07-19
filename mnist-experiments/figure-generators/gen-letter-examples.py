import matplotlib.pyplot as plt
import generate_data
import settings
import logging

logging.basicConfig(level=logging.INFO)

_, letter_samples_raw, _ = generate_data.load_letters(parameters=settings.parameters)

width = 10 #total number to show
height = 3
start_index = 0

f, ax = plt.subplots(height,width,dpi=300)
f.set_size_inches(3.3,1)
f.subplots_adjust()
#f.tight_layout()
for i in range(height):
    for j in range(width):
        ax[i,j].imshow(letter_samples_raw[i*width+j,:].reshape(28,28), cmap='gray_r')
        #Set_axis_off does not fit. I want a bounding box.
        ax[i,j].axes.get_xaxis().set_visible(False)
        ax[i,j].axes.get_yaxis().set_visible(False)
#gs.update(wspace=0.1, hspace=0.025)
f.subplots_adjust(left=0.005, right=0.995, top=0.975,bottom=0.025)

f.savefig("../figures/letter_examples.png")