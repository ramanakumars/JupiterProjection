import numpy as np
import glob, spiceypy, time
import matplotlib.pyplot as plt
c_light = 3.e5 ## light speed in km/s

## radians to arccsecs
RAD2ARCSEC = 206265.

def fit_ellipse_auto(v, RADEC, RADEC0, theta):
    '''
        fit the ellipse to determine the transformation 
        matrix to convert from RA/Dec to pixel

        Fits the following transformation:

        X = rot(-alpha)*A*rot(-theta)*(RADec-RADec0) + X0

        where 
            X      = pixel position
            alpha  = rotation of image on camera plate
            A      = pixel scale (rad/pixel)
            theta  = tilt of Jupiter wrt. J2000 north (input)
            RADec  = RA/Dec of observed point
            RADec0 = obs-sub RA/Dec of target
            X0     = pixel center of target on camera plate
            rot(angle) = rotation matrix about z-axis by angle

        Parameters
        ----------
        v : array_like
            pixel positions of the limb (dim: (m,2))
        RADEC : array_like
            RA/Dec values of the limb (dim: (N,2))
        RADEC0 : array_like
            obs-sub RA/Dec (dim: (2,))
        theta : float
            tilt angle of pole wrt J2000 north

        Output
        ------
        transform : array_like
            full transformation matrix 
            T = rot(-alpha)*A*rot(-theta)
        X0 : array_like
            target center pixel coordinates on camera plate
    '''
    N = RADEC.shape[0]

    ## radius-scale factor in the RA direction
    RR = np.abs(np.cos(RADEC0[1]))

    dtheta = 2.*np.pi/N

    ## solution vector (A, X0, Y0, alpha)
    X = 10.*np.ones(4)
    X[-1] = 0.

    done = False

    it = 0

    ## calculate the vectors in the RA/Dec frame
    ## convert to arcsec so that the convergence 
    ## is better
    dRA   = RR*(RADEC[:,0] - RADEC0[0])*RAD2ARCSEC
    dDec  =    (RADEC[:,1] - RADEC0[1])*RAD2ARCSEC

    ## and rotate them so they line up with the 
    ## axis of the target
    dx =  dRA*np.cos(theta) + dDec*np.sin(theta)
    dy = -dRA*np.sin(theta) + dDec*np.cos(theta)

    ## Jacobian matrix
    J = np.zeros((2*N, 4))
    ## error vector
    b = np.zeros(2*N)

    while not done:
        A, X0, Y0, alpha = X
        
        ## now calculate the vectors in the pixel 
        vshift = v - np.array([X0, Y0])

        ## calculate their angles -- this is how we 
        ## will match positions of the contour
        ## with that of the RA/Dec vector
        vang   = np.arctan2(v[:,0]-X0, v[:,1]-Y0)
        vang[vang<0.]       += 2.*np.pi
        vang[vang>2.*np.pi] -= 2.*np.pi

        ## fill the Jacobian
        ## F1 = transform_y - pixel_x
        J[ ::2,0] = dx*np.cos(alpha) + dy*np.sin(alpha)
        J[ ::2,1] = 1.
        J[ ::2,2] = 0.
        J[ ::2,3] = A*(-dx*np.sin(alpha) + dy*np.cos(alpha))
        
        ## F2 = transform_y - pixel_x
        J[1::2,0] = -dx*np.sin(alpha) + dy*np.cos(alpha)
        J[1::2,1] = 0.
        J[1::2,2] = 1.
        J[1::2,3] = -A*(dx*np.cos(alpha) + dy*np.sin(alpha))

        ## b = [F1, F2]
        b[ ::2]   = A*( dx*np.cos(alpha) + dy*np.sin(alpha))
        b[1::2]   = A*(-dx*np.sin(alpha) + dy*np.cos(alpha))

        bxy = np.dstack((b[::2], b[1::2]))

        for i in range(N):
            ## find the (x,y) that corresponds to this angle
            ## from the north pole
            xi, yi = vshift[np.argmin(np.abs(vang-i*dtheta)),:]

            '''
            J[2*i,  0] = dx[i]*np.cos(alpha) + dy[i]*np.sin(alpha)#RR*dRA*np.cos(theta) - dDec*np.sin(theta)
            #J[2*i,  1] = -dDec#RR*dRA*np.cos(theta) - dDec*np.sin(theta)
            J[2*i,  1] = 1.# np.cos(alpha)
            J[2*i,  2] = 0.#-np.sin(alpha)
            J[2*i,  3] = A*(-dx[i]*np.sin(alpha) + dy[i]*np.cos(alpha))
            #J[2*i+1,0] = dRA# + A*dy[i]#-RR*dRA*np.sin(theta) + dDec*np.cos(theta)
            J[2*i+1,0] = -dx[i]*np.sin(alpha) + dy[i]*np.cos(alpha)# + A*dy[i]#-RR*dRA*np.sin(theta) + dDec*np.cos(theta)
            J[2*i+1,1] = 0.#np.sin(alpha)
            J[2*i+1,2] = 1.#np.cos(alpha)
            J[2*i+1,3] = A*(-dx[i]*np.cos(alpha) - dy[i]*np.sin(alpha))
            #J[2*i+1,5] = -xi*np.cos(theta) + yi*np.sin(theta)
            b[2*i]   = A*( dx[i]*np.cos(alpha) + dy[i]*np.sin(alpha)) - xi
            b[2*i+1] = A*(-dx[i]*np.sin(alpha) + dy[i]*np.cos(alpha)) - yi            
            '''
            b[2*i]   -= xi
            b[2*i+1] -= yi            

        ## Gauss-Newton algorithm
        JTJinv = np.linalg.inv(np.dot(J.T, J))
        dX     = np.asarray(np.dot(np.dot(JTJinv, J.T), b))

        X = X - dX
        if(np.linalg.norm(dX) < 1.e-9):
            done = True
        it += 1

        if(it > 50):
            done = True

    ## retrieve the best fits 
    A, x0, y0, alpha = X
    X0 = np.asarray([x0, y0])

    ## refine the alpha angle by finding the point where 
    ## the error is minimized
    bnorm = np.linalg.norm([b[::2],b[1::2]], axis=0)
    angles = np.linspace(0., 2.*np.pi, N)
    angmax = angles[bnorm.argmin()]

    b1 = np.zeros_like(RADEC)
    b2 = np.zeros_like(RADEC)

    b1[:,0]   = A*( dx*np.cos(alpha) + dy*np.sin(alpha))
    b1[:,1]   = A*(-dx*np.sin(alpha) + dy*np.cos(alpha))
    b2[:,0]   = A*( dx*np.cos(alpha+angmax) + dy*np.sin(alpha+angmax))
    b2[:,1]   = A*(-dx*np.sin(alpha+angmax) + dy*np.cos(alpha+angmax))

    vshift = v - X0
    vang   = np.arctan2(vshift[:,0], vshift[:,1])
    vang[vang<0.]       += 2.*np.pi
    vang[vang>2.*np.pi] -= 2.*np.pi

    for i, thetai in enumerate(angles):
        dist1    = np.linalg.norm(vshift - b1[i,:], axis=1).argmin()
        dist2    = np.linalg.norm(vshift - b2[i,:], axis=1).argmin()
        b1[i,:] -= vshift[dist1,:]
        b2[i,:] -= vshift[dist2,:]

    b1norm = np.linalg.norm(b1, axis=1)
    b2norm = np.linalg.norm(b2, axis=1)

    A  = A*RAD2ARCSEC

    ## update alpha if the new one is better
    if np.mean(b1norm) > np.mean(b2norm):
        alpha += angmax

    return (A, alpha, X0)

def get_ang_dist(t1, t2):
    '''
        get the angular distance between two 
        angular positions

        Parameters
        ----------
        t1 : array_like
            first position (RA, Dec) in radians
        t2 : array_like
            second_position (RA, Dec) in radians


        Outputs
        -------
        theta : float
            angular distance between t1 and t2
    '''
    return np.arccos(np.sin(t1[1])*np.sin(t2[1])\
                     + np.cos(t1[1])*np.cos(t2[1])*np.cos(t1[0]-t2[0]))


def find_best_path(paths, x0, y0):
    '''
        finds the index of the contour closest to (x0, y0)
    '''
    npaths = len(paths)
    dists = np.zeros(npaths)
    for ii in range(npaths):
        pathi = paths[ii].vertices
        if(pathi.size < 20):
            dists[ii] = 1.e6
            continue   
        if((pathi[:,1].max() - pathi[:,1].min()) < 50):
            dists[ii] = 1.e6
            continue
        if((pathi[:,0].max() - pathi[:,0].min()) < 50):
            dists[ii] = 1.e6
            continue
        dists[ii] = np.mean((pathi[:,0]-x0)**2. + (pathi[:,1]-y0)**2.)
    return np.argmin(dists)

def find_spice_kernels(kernel_path):
    pcks  = sorted(glob.glob(kernel_path+"pck/*.tpc"))
    spks1 = sorted(glob.glob(kernel_path+"spk/planets/de*.bsp"))
    spks2 = sorted(glob.glob(kernel_path+"spk/satellites/*.bsp"))
    fks   = sorted(glob.glob(kernel_path+"fk/planets/*.tf"))
    lsks  = sorted(glob.glob(kernel_path+"lsk/naif*.tls"))


    kernels = [pcks[-1], spks1[-1], *spks2, lsks[-1]]
    for kernel in kernels:
        spiceypy.furnsh(kernel)


def get_rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
