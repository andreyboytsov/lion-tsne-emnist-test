import matplotlib.pyplot as plt
import generate_data
import numpy as np
import settings
from matplotlib.font_manager import FontProperties

# Step 1. Load all data.
parameters = settings.parameters

Y_mnist= generate_data.load_y_mnist(parameters=parameters)
X_mnist= generate_data.load_x_mnist(parameters=parameters)
dTSNE_mnist = generate_data.load_dtsne_mnist(parameters=parameters)
lion_toy_interp = dTSNE_mnist.generate_lion_tsne_embedder(verbose=0, random_state=0, function_kwargs={'y_safety_margin':0,
                                                                                                     'radius_y_percentile':100})

# Step 2. Generate outliers and embed them
# Usually we separate data generation and figure generation, but here calculations are just too fast and
# data should never be used anywhere else.
n_outl = 392
x_outl = np.zeros([n_outl, X_mnist.shape[1]])
np.random.seed(0)
for i in range(n_outl):
    n = np.random.choice(30,15)
    x_sample = np.array([-100]*X_mnist.shape[1]).reshape(1,-1)
    x_sample[0,n] = 100
    x_outl[i,:] = x_sample

y_outl = lion_toy_interp(x_outl, verbose=0) #Verbose 2 produce quite a bit of output. Verbose 3 produce a lot.


# Step 3. Generating the plot
f,ax = plt.subplots(dpi=300)

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

f.set_size_inches(3.3,3.3) #Let's set the plot sizes that just fit paper margins
legend_list = list()
ax.scatter(Y_mnist[:, 0], Y_mnist[:, 1], marker = '.', s=5, c='gray')
#l = plt.legend([dc,dcr],["Data Radius","Data Radius + $r_y$"], loc=2, borderaxespad = 0)
#ax.plot(y_outl[:, 0], y_outl[:, 1], marker = '.', ms=5,linestyle=":", c='red')
n_o = 212
n_o_2 = 298
g1 = ax.scatter(y_outl[:n_o, 0], y_outl[:n_o, 1], marker = 'o', s=5,c='red', lw=0)
g2 = ax.scatter(y_outl[n_o:n_o_2, 0], y_outl[n_o:n_o_2, 1], marker = 'o', s=5,c='blue',lw=0)
g3 = ax.scatter(y_outl[n_o_2:, 0], y_outl[n_o_2:, 1], marker = 'o', s=5,c='green',lw=0)
ax.plot([min(Y_mnist[:, 0]), min(Y_mnist[:, 0]), max(Y_mnist[:, 0]), max(Y_mnist[:, 0]), min(Y_mnist[:, 0])],
        [min(Y_mnist[:, 1]), max(Y_mnist[:, 1]), max(Y_mnist[:, 1]), min(Y_mnist[:, 1]), min(Y_mnist[:, 1])],
        linestyle='--', c='black',lw=1)
ax.plot([min(Y_mnist[:, 0])-10, min(Y_mnist[:, 0])-10, max(Y_mnist[:, 0])+10, max(Y_mnist[:, 0])+10, min(Y_mnist[:, 0])-10],
        [min(Y_mnist[:, 1])-10, max(Y_mnist[:, 1])+10, max(Y_mnist[:, 1])+10, min(Y_mnist[:, 1])-10, min(Y_mnist[:, 1])-10],
        linestyle='--', c='black',lw=1)
plt.tick_params(axis='both', which='both',bottom=False,top=False,labelbottom=False,
                                          left=False,right=False,labelleft=False)
ax.set_ylim([-150,150])
ax.set_xlim([-160,160])
# f.tight_layout()
#l = plt.legend([g1,g2,g3],["First outliers will be mapped to \nthose positions",
#                       "... then outliers will be mapped\nto outer layer positions",
#                       "... and then to more distant\nlayer(s)"],loc=6, bbox_to_anchor=[-0.025,-0.22],frameon=False)
plt.tight_layout(rect=[-0.035, -0.035, 1.035, 1.035])
f.savefig("../figures/outlier_placement_example.png")
# plt.show(f)