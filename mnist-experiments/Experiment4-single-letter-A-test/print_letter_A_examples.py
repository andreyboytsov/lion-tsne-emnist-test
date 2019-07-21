import generate_data
import settings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import settings

parameters = settings.parameters
X_mnist_raw = generate_data.load_x_mnist_raw(parameters=parameters)
letters_A, letters_A_raw = generate_data.load_A_letters(parameters=parameters)

print(letters_A_raw.shape, np.max(letters_A_raw[0,:]), np.min(letters_A_raw[0,:]))

width = 10 #total number to show in one row
start_index = 0

height = 10 # Number of rows /2 to show. half will go to labels, half to pictures.

f, ax = plt.subplots(height,width)
f.set_size_inches(16,16)
f.subplots_adjust()
for i in range(int(height)):
    for j in range(width):
        ax[i][j].imshow(letters_A_raw[start_index + width*i + j,:].reshape(28,28), cmap='gray_r')
        #ax[2*i+1][j].text(text=str(letters_A_labels[start_index + width*i + j])

            #str(chr(

            #(ord('A') if letters_labels[start_index + width*i + j]<26 else ord('a'))+

            #letters_labels[start_index + width*i + j] +

            #(0 if letters_labels[start_index + width*i + j]<26 else -26) ))
            #,x=0.5, y=0.5,s=11,
            #        ha='center', va='center', fontsize=16)
        #ax[2*i+1][j].imshow(X_mnist_raw[start_index + width * i + j, :].reshape(28, 28), cmap='gray_r')
        #ax[2*i+1][j].axes.get_xaxis().set_visible(False)
        #ax[2*i+1][j].axes.get_yaxis().set_visible(False)

        ax[i][j].axes.get_xaxis().set_visible(False)
        ax[i][j].axes.get_yaxis().set_visible(False)
#f.subplots_adjust(left=-0.04, right=0.99, top=0.99,bottom=0.01)
plt.show(f)
