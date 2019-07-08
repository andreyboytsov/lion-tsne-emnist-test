import generate_data
import settings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parameters = settings.parameters
X_mnist_raw = generate_data.load_x_mnist_raw(parameters=parameters)

#letters, letters_raw = generate_data.load_letters(parameters=settings.parameters)
letters_random_seed = parameters.get("letter_random_seed", settings.parameters["letter_random_seed"])
ind_to_pick = parameters.get("letter_indices_to_pick", settings.parameters["letter_indices_to_pick"])
np.random.seed(letters_random_seed)
emnist_balanced_train = np.genfromtxt('../../../emnist/emnist-balanced-train.csv', delimiter=',')
emnist_balanced_train = emnist_balanced_train[np.where(emnist_balanced_train[:,0] >= 10)[0], :]

ind = np.random.choice(np.arange(len(emnist_balanced_train)), size=ind_to_pick)

print(X_mnist_raw.shape, np.max(X_mnist_raw[0,:]), np.min(X_mnist_raw[0,:]))
print(emnist_balanced_train.shape)

letters_raw = emnist_balanced_train[ind, 1:].reshape((-1, 28, 28)).transpose((0,2,1)).reshape((-1, 784)) / 255.0
letters_labels = (emnist_balanced_train[ind, 0]-10).astype(int)

print(letters_raw.shape, np.max(letters_raw[0,:]), np.min(letters_raw[0,:]))
print(letters_labels[:100])

width = 20 #total number to show in one row
start_index = 0

height = 20 # Number of rows /2 to show. half will go to labels, half to pictures.

f, ax = plt.subplots(height,width)
f.set_size_inches(16,16)
f.subplots_adjust()
for i in range(int(height/2)):
    for j in range(width):
        ax[2*i][j].imshow(letters_raw[start_index + width*i + j,:].reshape(28,28), cmap='gray_r')
        ax[2*i+1][j].text(text=str(letters_labels[start_index + width*i + j])

            #str(chr(

            #(ord('A') if letters_labels[start_index + width*i + j]<26 else ord('a'))+

            #letters_labels[start_index + width*i + j] +

            #(0 if letters_labels[start_index + width*i + j]<26 else -26) ))
            ,x=0.5, y=0.5,s=11,
                    ha='center', va='center', fontsize=16)
        #ax[2*i+1][j].imshow(X_mnist_raw[start_index + width * i + j, :].reshape(28, 28), cmap='gray_r')
        #ax[2*i+1][j].axes.get_xaxis().set_visible(False)
        #ax[2*i+1][j].axes.get_yaxis().set_visible(False)

        ax[2*i][j].axes.get_xaxis().set_visible(False)
        ax[2*i][j].axes.get_yaxis().set_visible(False)
        ax[2*i+1][j].set_axis_off()
#f.subplots_adjust(left=-0.04, right=0.99, top=0.99,bottom=0.01)
plt.show(f)
