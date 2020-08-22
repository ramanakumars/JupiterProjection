from projection_funcs import *

et  = spiceypy.utc2et('2019-02-02 10:07:06')
img = plt.imread('2019-02-02-1007_1-RGBdp.jpg')/255.

vecs     = get_vec_from_image(img, et)

lon, lat = map_project(vecs, et)

fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(lon, vmin=0., vmax=360.)
ax2.imshow(lat, vmin=-90., vmax=90.)
ax3.imshow(img)

plt.tight_layout()

fig.savefig('projected.png', dpi=150)

np.save('lon.npy', lon, allow_pickle=False)
np.save('lat.npy', lat, allow_pickle=False)

plot_map(lon, lat, img)

