import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_properties = FontProperties()
font_properties.set_family('serif')
font_properties.set_name('Times New Roman')
font_properties.set_size(8)

f,ax = plt.subplots(dpi=300)

f.set_size_inches(1.7,0.8) #3.3 is the max

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

ax.plot([5,1],[9,1], c='black', linewidth=0.5)
ax.plot([5,9],[9,1], c='black', linewidth=0.5)
ax.plot([1,9],[1,1], c='black', linewidth=0.5)
#ax.plot([0,5],[0,4], c='black', linewidth=0.5)

ax.scatter([5,1,9],[9,1,1], c='black', s=5)

ax.text(5, 8, '1',fontproperties=font_properties, ha = 'center', va = 'top')
ax.text(1, 1.3, '2',fontproperties=font_properties, ha = 'center', va = 'bottom')
ax.text(9, 1.3, '3',fontproperties=font_properties, ha = 'center', va = 'bottom')

ax.text(2.7, 5.6, '$<r_x$',fontproperties=font_properties, rotation=35,ha = 'center', va = 'center')
ax.text(7.3, 5.6, '$<r_x}$',fontproperties=font_properties, rotation=-35,ha = 'center', va = 'center')
ax.text(5, 2, '$>r_x$',fontproperties=font_properties, ha = 'center', va = 'center')

ax.set_ylim([0,10])
ax.set_xlim([0,10])
f.tight_layout(rect=[-0.08,-0.17,1.08,1.17])
f.savefig("../figures/order_of_arrival.png")
#plt.show(f)