import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.font_manager import FontProperties

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

f,ax = plt.subplots(dpi=300)

f.set_size_inches(1.7,1.7) #3.3 is the max

c = plt.Circle((0,0), 9, fill=None, color='black')
dc = ax.add_artist(c)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

ax.plot([0,0],[0,9], c='black', linewidth=0.5)
ax.plot([0,5],[0,4], c='black', linewidth=0.5)

ax.scatter([0],[0], c='black', s=5)
ax.scatter([5],[4], c='black', s=5)

ax.text(-1, 5, '$r_{close}$',fontproperties=font_properties, rotation=90, ha = 'center', va = 'center')
ax.text(-1, -1, '$y_i$',fontproperties=font_properties, ha = 'center', va = 'center')
ax.text(5.5, 3, '$y$',fontproperties=font_properties, ha = 'center', va = 'center')
ax.text(1,2.5, '$\\alpha$',fontproperties=font_properties)

inner_arc = patches.Arc((0,0), 5, 5,  theta1=40, theta2=90, color='black', linewidth=0.3)
dia = ax.add_artist(inner_arc)

ax.text(3, 1.1, '$d$',fontproperties=font_properties, rotation=40, ha = 'center', va = 'center')

ax.set_ylim([-10,10])
ax.set_xlim([-10,10])
f.tight_layout()
f.savefig("../figures/vicinity_illustration.png")
#plt.show(f)