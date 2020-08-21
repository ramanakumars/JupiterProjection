import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import spiceypy, glob

KERNEL_DATAFOLDER = '/home/local/Isis/data/juno/kernels/'
pcks  = sorted(glob.glob(KERNEL_DATAFOLDER+"pck/pck*.tpc"))
spks2 = sorted(glob.glob(KERNEL_DATAFOLDER+"spk/jup*.bsp"))
#spks3 = sorted(glob.glob(KERNEL_DATAFOLDER+"spk/de438.bsp"))
lsks  = sorted(glob.glob(KERNEL_DATAFOLDER+"lsk/naif*.tls"))

kernels = [pcks[-1], spks2[-1], lsks[-1]]
for kernel in kernels:
    spiceypy.furnsh(kernel)

c_light = 3.e8 ## light speed in m/s

def fit_ellipse(v):
    ''' 
        from NUMERICALLY  STABLE  DIRECT  LEAST  SQUARESFITTING  OF  ELLIPSES 
        (Halir and Flusser 1998)
    
    '''
    x  = v[:,0]
    y  = v[:,1]
    nx = x.shape[0]

    ### python version
    x = x.reshape((nx,1))
    y = y.reshape((nx,1))

    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    # print(D.shape)
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]

    A, B, C, D, F, G = a
    B = B /2.
    D = D/2.
    F = F/2.
    
    disc = B**2. - A*C

    x0 = (C*D - B*F)/disc
    y0 = (A*F - B*D)/disc

    a  = np.sqrt((2*(A*F**2. + C*D**2. + G*B**2. - 2*B*D*F - A*C*G))/\
        (disc*(np.sqrt((A-C)**2. + 4*B**2) - (A+C))))
    b  = np.sqrt((2*(A*F**2. + C*D**2. + G*B**2. - 2*B*D*F - A*C*G))/\
        (disc*(-np.sqrt((A-C)**2. + 4*B**2) - (A+C))))

    if(B==0):
        if(A < C):
            alpha = 0.
        else:
            alpha = np.pi/2.
    else:
        alpha = np.arctan2((C-A-np.sqrt((A-C)**2. + B**2.)), B)

    return (x0, y0, a, b, alpha)

def find_best_path(paths, x0, y0):
    '''
        finds the index of the contour closest to (x0, y0)
    '''
    npaths = len(paths)
    dists = np.zeros(npaths)
    for ii in range(npaths):
        pathi     = paths[ii].vertices
        dists[ii] = (1./pathi.shape[0])*np.sum((pathi[:,0]-x0)**2. + (pathi[:,1]-y0)**2.)
    return np.argmin(dists)

def get_vec_from_image(img, et):
    '''
        use an image of Jupiter (aligned north upward)
        and calculates the vector in the J2000 frame that 
        each pixels points to
    '''
    ny, nx, _ = img.shape

    halfnx = int(nx/2)
    halfny = int(ny/2)

    ''' position of jupiter in J2000 ref frame '''
    jup_pos_spice, lt = spiceypy.spkpos("JUPITER", et, "J2000", "CN+S", "EARTH")
    jup_pos_spice = jup_pos_spice/np.linalg.norm(jup_pos_spice)

    ''' get the distance to jupiter '''
    dist = lt*c_light
    
    ''' get the coordinate directions in the camera frame '''
    jup2j2000   = spiceypy.pxform("IAU_JUPITER", "J2000", et)
    npole_j2000 = np.matmul(jup2j2000, np.array([0., 0., 1.]))
    ivec_j2000  = np.cross(npole_j2000, -jup_pos_spice)

    ''' fit an ellipse to the image and find the center '''
    mask_img = img[:,:,0].copy()
    mask_img[img[:,:,0] > 0.05] = 1.
    mask_img[img[:,:,0] < 0.05] = 0.

    figtest = plt.figure()
    axtest  = figtest.add_subplot(111)

    cont = axtest.contour(mask_img, 1, colors='k')
    paths = cont.collections[0].get_paths()

    ''' get the contour corresponding to Jupiter assuming
        that Jupiter is closest to the center '''
    pathi = find_best_path(paths, halfnx, halfny)

    ''' change the pathi if it detects some other contour '''
    jupcont = paths[pathi].vertices

    axtest.cla()
    axtest.imshow(img)

    axtest.plot(jupcont[:,0], jupcont[:,1], 'g-')

    figtest.savefig('ellipse.png')

    ''' fit the ellipse and get the fit parameters '''
    x0, y0, a, b, alpha = fit_ellipse(jupcont)

    _, radii  = spiceypy.bodvrd("JUPITER", "RADII", 3)

    ''' assume a linear scale '''
    xscale = 1000.*radii[0]/a
    yscale = 1000.*radii[2]/b

    vecs = np.zeros((ny, nx, 3))

    print("Building vectors:")
    for jj in range(ny):
        print("\r %3d/%3d"%(jj, ny), end='')
        for ii in range(nx):
            vec = dist*jup_pos_spice + \
                     (y0-float(jj))*yscale*npole_j2000 + \
                     (float(ii)-x0)*xscale*ivec_j2000
            vecs[jj,ii,:] = vec/np.linalg.norm(vec)
    print()
    return vecs

def map_project(vecs, et):
    ''' 
        finds the intercept between a set of vectors defined in 
        the J2000 frame and Jupiter's surface ellipsoid
        returns lat/lon of each "pixel" in the vectors array
        lat/lon = -1000 if no intercept is found
    '''
    ny, nx, _ = vecs.shape

    _, radii  = spiceypy.bodvrd("JUPITER", "RADII", 3)
    flattening = (radii[0] - radii[2])/radii[0]

    lat = -1000.*np.ones((ny, nx))
    lon = -1000.*np.ones((ny, nx))
    
    print("Projecting image:")
    for jj in range(ny):
        print("\r %3d/%3d"%(jj, ny), end='')
        for ii in range(nx):
            try:
                spoint, _, _ = spiceypy.sincpt("Ellipsoid", "JUPITER", et, "IAU_JUPITER", "CN+S", "EARTH", "J2000", vecs[jj,ii,:])
            except Exception as e: 
                continue
            loni, lati, alt = spiceypy.recpgr("JUPITER", spoint, radii[0], flattening)

            lat[jj,ii] = np.degrees(lati)
            lon[jj,ii] = np.degrees(loni)
    print()
    return (lon, lat)

def plot_map(lon, lat, img, pixres=0.05):
    '''
        project the image onto a lat/lon grid
    '''
    gridlat = np.arange(-90., 90., pixres)
    gridlon = np.arange(0., 360., pixres)
    
    LAT, LON = np.meshgrid(gridlat, gridlon)

    lon_f  = lon.flatten()
    lat_f  = lat.flatten()
    Rimg_f = img[:,:,0].flatten()
    Gimg_f = img[:,:,1].flatten()
    Bimg_f = img[:,:,2].flatten()

    mask  = np.where((lon_f == -1000.)|(lat_f == -1000.))[0]

    Rimg = np.delete(Rimg_f, mask)
    Gimg = np.delete(Gimg_f, mask)
    Bimg = np.delete(Bimg_f, mask)
    lats = np.delete(lat_f, mask)
    lons = np.delete(lon_f, mask)

    ''' convert to east positive '''
    lons = 360. - lons

    IMG  = np.zeros((gridlat.size, gridlon.size, 3))

    colors = ['R', 'G', 'B']

    for color, imgi in enumerate([Rimg, Gimg, Bimg]):
        print("Interpolating %s"%(colors[color]))
        IMG[:,:,color] = griddata((lons, lats), imgi, (LON, LAT), method='cubic').T

    plt.imsave("map.png", IMG, origin='lower')

et  = spiceypy.utc2et('2019-02-02 10:07:06')
img = plt.imread('2019-02-02-1007_1-RGBdp.jpg')/255.
'''
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
'''

lat = np.load('lat.npy')
lon = np.load('lon.npy')

plot_map(lon, lat, img)
