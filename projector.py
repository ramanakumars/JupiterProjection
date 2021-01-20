import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import rotate as sprotate
import spiceypy, glob, re, os
from util import *
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import time

class PlateCalibration():
    '''
        PlateCalibration object to store information about
        the camera plate -- including pixel scale and rotation
        angle wrt. RA/Dec north and target north
    '''
    def __init__(self, RADec0, RArotang):
        '''
            `PlateCalibration` class initializer

            Parameters 
            ----------
            RADec0 : numpy.ndarray
                RA/Dec of target center (radians)
            RArotang : float
                Rotation angle of target's north wrt
                J2000 north (radians)
        '''
        ## set default values
        self.RADec0   = RADec0
        self.RArotang = RArotang

        self.scale  = 100.*RAD2ARCSEC ## pixels/arcsec
        self.alpha  = 0.
        
        ## create the rotation matrix from RADec
        ## to align with target's N/S direction
        self.RR     = np.abs(np.cos(self.RADec0[1]))
        self.RADecrot = \
            np.array([[self.RR*np.cos(self.RArotang), np.sin(self.RArotang)],\
                     [-self.RR*np.sin(self.RArotang), np.cos(self.RArotang)]])

    def get_transform_matrix(self):
        '''
            get the rotation matrix to convert from
            camera plate frame to J2000

            Output
            ------
            transform : numpy.matrix
                transformation matrix to rotate and scale input
                RA/Dec (radians) to pixel positions

        '''
        t1    = self.scale*\
                    np.array([[np.cos(self.alpha), np.sin(self.alpha)],\
                    [-np.sin(self.alpha), np.cos(self.alpha)]])

        transform = np.matmul(t1, self.RADecrot)

        return transform


class Projector():
    '''
        Projector object that holds the necessary
        attributes and methods to convert pixel 
        coordinates from planetary images to lat/lon 

        Attributes
        ----------
        ut : string
            Observation time in UTC ISO format:
            yyyy-mm-dd hh:mm:ss
        et : float
            Observation time in seconds after J2000
        target : string
            Name of observing body
        target_frame : string
            IAU_[target] - Target body-centered fixed frame 
        radii : array_like
            planetary radius along each axis
        flattening : float
            polar flattening f=(a-c)/a
        tstate : array_like
            target state at observing time in the J2000 frame
        lt : float
            one-way light travel time to target at et

    '''
    def __init__(self, ut, kernel_path, target='JUPITER'):
        '''
            `Projector` class initializer

            Parameters
            ----------
            ut : string
                Observation time in the UTC ISO format: 
                yyyy-mm-dd hh:mm:ss
            kernel_path : string
                Path to the NAIF generic_kernels folder
            target : string, [default=Jupiter]
                Name of the observing body
        '''
        ## find and load the kernels
        find_spice_kernels(kernel_path)

        ## convert the times
        self.ut = ut
        self.et = spiceypy.utc2et(ut)

        ## calculate target information
        self.target=target
        self.target_frame = 'IAU_'+target
        self.radii = spiceypy.bodvar(spiceypy.bodn2c(target), "RADII", 3)
        self.flattening = (self.radii[0] - self.radii[2])/self.radii[0]

        ## get the target state in J2000
        self.tstate, self.lt = \
            spiceypy.spkezr(target, self.et, 'J2000', 'CN', "EARTH")

        self.tRA, self.tDEC = \
            spiceypy.recrad(self.tstate[:3])[1:]

        self.find_sub_pt()
        self.find_limb()
        
        self.platecal = PlateCalibration(self.RADec0, self.RArotang)

    @classmethod
    def from_img(cls, imgpath, ut, kernel_path, target='JUPITER'):
        '''
            import and store the observation from a PNG/JPG file

            Parameters
            ----------
            imgpath : string
                path to the image file
            ut : string
                Observation time in the UTC ISO format: 
                yyyy-mm-dd hh:mm:ss
            kernel_path : string
                Path to the NAIF generic_kernels folder
            target : string, [default=Jupiter]
                Name of the observing body

        '''
        img = plt.imread(imgpath)
        fname    = imgpath.split('/')[-1]
        filename, ext = os.path.splitext(fname)

        ## normalize the JPGs
        if(ext=='.jpg'):
            img = np.array(img, dtype=float)/255.

        ny, nx, _ = img.shape

        R2P0 = np.array([nx/2, ny/2])

        ## initialize the object
        proj = cls(ut, kernel_path, target)

        ## set the class attributes
        proj.img = img
        proj.filename = filename
        proj.ny   = ny
        proj.nx   = nx
        proj.R2P0 = R2P0

        return proj
    
    @classmethod
    def from_fits(cls, fitspath, kernel_path):
        with fits.open(fitspath) as hdu:
            hdu0 = hdu[0]
            header = hdu0.header

            if('SpeX' in header.tostring()):
                #print([key for key in header.keys()])

                obs_dat  = header['DATE_OBS']
                obs_time = header['TIME_OBS']
                target   = header['OBJECT']

                ut = obs_dat + " " + obs_time
                print(ut)

                proj = cls(ut=ut, kernel_path=kernel_path, target=target)

                proj.platecal.scale = \
                    RAD2ARCSEC/float(header['PLATE_SC'])

                alpha = -np.radians(360. + 90.  - float(header['ROTATOR']))

                while alpha > 2.*np.pi:
                    alpha -= 2.*np.pi
                while alpha < 0.:
                    alpha += 2.*np.pi
                proj.platecal.alpha = alpha

                proj.img = hdu0.data

                proj.ny, proj.nx = proj.img.shape

                proj.R2P0   = np.array([proj.nx/2, proj.ny/2])
                RA0  = np.array(header['TCS_RA'].split(':'), dtype=float)
                RA0  = RA0[0]*15. + RA0[1]/4. + RA0[2]/240.

                Dec0 = np.array(header['TCS_DEC'].split(':'), dtype=float)
                Dec0 = Dec0[0] + \
                    np.sign(Dec0[0])*(Dec0[1]/60. + Dec0[2]/3600.)

                RA0  = np.radians(RA0)
                Dec0 = np.radians(Dec0)

                dRADec = proj.RADec0 - np.array([RA0, Dec0])
                dpix   = dRADec

                rotate = np.array([[np.cos(alpha), -np.sin(alpha)],\
                                   [np.sin(alpha), np.cos(alpha)]])

                dpix   = np.matmul(proj.platecal.get_transform_matrix(), dpix)
                print(dpix)
                print(np.degrees(proj.RArotang))


                proj.R2P0 -= dpix

                #proj.RADec0 = np.array([RA0, Dec0])

        return proj

    def pix2radec(self, pix):
        '''
            converts camera pixel to 
            RA/DEC in radians
            
            Parameters
            ----------
            pix : numpy.ndarray
                (x, y) position on the camera

            Returns
            -------
            radec : numpy.ndarray
                (ra, dec) of the pixel in radians


            See also
            ---------
            radec2pix() : converts RA/Dec to pixel coordinates

        '''
        if(hasattr(self, 'wcs')):
            if(self.wcs):
                return (0,0)
        else:
            ## rotate to make sure that it is (2, N) instead of
            ## (N, 2) even though that is the expected input
            if((len(pix.shape) > 1)&(pix.shape[0] != 2)):
                dpix = (pix-self.R2P0).T
            else:
                dpix = pix-self.R2P0
            radec = np.matmul(self.tPIX2R, dpix)

            ## revert back to the original shape
            if(len(pix.shape) > 1):
                radec = radec.T
            radec += self.RADec0

        return radec

    def radec2pix(self, radec):
        '''
            converts RA/DEC in radians
            to pixel on the camera

            Parameters
            ----------
            radec : numpy.ndarray
                (ra,dec) in radians

            Returns
            ------
            pix : numpy.ndarray
                (x, y) cell-centered pixels on the camera

        '''
        if(hasattr(self, 'wcs')):
            if(self.wcs):
                return (0,0)
        else:
            ## rotate to make sure that it is (2, N) instead of
            ## (N, 2) even though that is the expected input
            if((len(radec.shape) > 1)&(radec.shape[0] != 2)):
                dradec = (radec-self.RADec0).T
            else:
                dradec = radec-self.RADec0

            pix = np.matmul(self.tR2PIX, dradec)

            ## revert back to the original shape
            if(len(radec.shape) > 1):
                pix = pix.T
            pix = pix  + self.R2P0
        
        return pix

    def find_limb(self):
        '''
            Finds the set of points that define the limb of the object
            as seen from earth. Converts the limb points to RA/Dec,
            lat/lon for later use. Also gets the apparent semi-major
            and semi-minor axis 
        '''
        METHOD='TANGENT/ELLIPSOID'
        CORLOC='ELLIPSOID LIMB' 
        refvec=[0., 0., 1.]

        ## space the limb points by 5deg, so we get 72 limb points
        rolstp=np.radians(0.1)
        ncuts =int(2.*np.pi/rolstp)

        ## calculate the limb points in the IAU_[target] frame
        _, limbs, eplimb, vecs = \
            spiceypy.limbpt(METHOD, self.target, self.et, self.target_frame, \
                            'CN', CORLOC, 'EARTH', refvec, rolstp, \
                            ncuts, 4.0, self.lt*c_light, ncuts)

        self.limbJ2000 = np.zeros_like(vecs)
        self.limbRADec = np.zeros((ncuts,2))
        self.limbdist  = np.zeros(ncuts)
        self.limbep    = np.array(eplimb)
        for i in range(ncuts):
            ## get the transformation of the limb points to the J2000 frame
            pxi = spiceypy.pxfrm2(self.target_frame, \
                                  'J2000', eplimb[i], self.et)

            ## transform the vectors from the observer to the limb to J2000
            self.limbJ2000[i,:] = np.matmul(pxi, vecs[i,:])

            ## also convert to RA/Dec
            self.limbdist[i], self.limbRADec[i,0], self.limbRADec[i,1] = \
                spiceypy.recrad(self.limbJ2000[i,:])

        ## the first point is on the north pole
        NP = 0
        EE = int(ncuts/4)
        SP = int(ncuts/2)
        EW = int(3*ncuts/4)

        ## get the apparent semi-major and semi-minor axis
        self.obsa = get_ang_dist(self.limbRADec[EE], self.limbRADec[EW])/2.
        self.obsb = get_ang_dist(self.limbRADec[NP], self.limbRADec[SP])/2.

        EEvec  = self.limbRADec[EE] - np.array([self.subRA, self.subDec])

        self.RArotang = -np.arctan2(EEvec[1], EEvec[0])

    def find_sub_pt(self):
        '''
            Finds the sub-observer point in the IAU_[target] frame
            and also the in the J2000 frame as seen from Earth.
            Also creates a vector from Earth to the sub-obs point in the 
            J2000 frame for future use.
        '''

        ## get the position of the sub-obs point in the J2000 frame
        self.subptvec, self.subptep, self.subpntobsvec = \
            spiceypy.subpnt('INTERCEPT/ELLIPSOID', self.target, \
                            self.et, self.target_frame, 'CN', 'EARTH')

        ## convert to lat/lon
        self.subptlon, self.subptlat, _ = \
            spiceypy.recpgr(self.target, self.subptvec, \
                            self.radii[0], self.flattening)

        ## convert the line of sight vector to J2000
        px1    = spiceypy.pxfrm2(self.target_frame, 'J2000', \
                                 self.subptep, self.et)

        self.subptJ2000 = np.matmul(px1, self.subpntobsvec)

        ## get the RA/DEC of the sub-obs point
        self.subRA, self.subDec = spiceypy.recrad(self.subptJ2000)[1:]
        self.RADec0 = np.array([self.subRA, self.subDec])

    def get_surf_vec(self, lon, lat):
        '''
            return the vector from the observer to the 
            surface point defined by `lon`, `lat` in the J2000 frame

            Parameters
            ----------
            lon : float
                planetographic longitude in radians
            lat : float
                planetographic latitude in radians

            Returns
            -------
            vec : array_like
                vector from the observer to the (lon, lat) point in the 
                J2000 frame at et
        '''
        ## make sure the sub point has been found
        if not hasattr(self, 'subptvec'):
            self.find_sub_pt()

        ## get the position of this lat/lon in the IAU_[target] frame
        surfvec = spiceypy.pgrrec(self.target, lon, lat, 0, \
                                  self.radii[0], self.flattening)

        ## get the difference in the LOS distance from the sub-obs 
        ## point to this point
        dist =  np.linalg.norm(self.subpntobsvec - self.subptvec + surfvec) \
            - np.linalg.norm(self.subpntobsvec)
        
        ## compute the change in light travel time 
        det   = dist/(c_light)
        ep    = self.subptep - det

        ## transform the surfvec to the J2000 frame corresponding to 
        ## when the ray hit the detector
        px2    = spiceypy.pxfrm2(self.target_frame, 'J2000', ep, self.et)
        newvec = self.subptJ2000 + np.matmul(px2, surfvec-self.subptvec)

        
        ## DEBUG
        #spoint, epp, svec = spiceypy.sincpt('ELLIPSOID', self.target, self.et, self.target_frame, 'CN', 'EARTH', \
        #                                'J2000', newvec/np.linalg.norm(newvec))

        #print(np.linalg.norm(spoint - surfvec))

        return newvec

    def draw_grid(self):
        if not hasattr(self, 'limbRA'):
            self.find_limb()

        limbpix = np.zeros(*self.limbRADec.shape)

        ## get the ellipse
        for i in range(self.limbRADec.shape[0]):
            limbpix[i,:] = self.radec2pix(self.limbRA[i], self.limbDec[i])

        
    def fit_img_ellipse(self, gamma=0.1, threshold=0.5, NorthUp=True):
        '''
            use an image of the target and calculates the vector 
            in the J2000 frame that each pixels points to

            Parameters
            ----------
            gamma : float
                Increases the image contrast to find the limb [default: 0.1]
            threshold : float
                threshold pixel value to determine the limb [default: 0.5]
            NorthUp : bool
                flag to check if north is up in the image (i.e. north pole
                has a lower y value compared to the south pole)
        '''
        if(len(self.img.shape) > 2):
            ny, nx, _ = self.img.shape
        else:
            ny, nx = self.img.shape

        done = False

        gamma=0.1
        try: 
            R = self.img[:,:,0]; G = self.img[:,:,1]; B = self.img[:,:,2]
            Rg = np.clip(np.power(R, gamma), 0, 1)
            Gg = np.clip(np.power(G, gamma), 0, 1)
            Bg = np.clip(np.power(B, gamma), 0, 1)
           
            img_gamma = np.min([Rg, Gg, Bg], axis=0)
        except IndexError:

            img_gamma = np.clip(np.power(self.img, gamma), 0, 10)

        figtest = plt.figure(figsize=(10,10))
        axtest  = figtest.add_subplot(111)        

        mask_img = img_gamma.copy()
        mask_img[img_gamma > threshold] = 1.
        mask_img[img_gamma < threshold] = 0.
        
        halfny = int(nx/2)
        halfnx = int(ny/2)
        
        cont = axtest.contour(mask_img, 1, antialiased=True)
        paths = cont.collections[0].get_paths()

        axtest.set_aspect('equal')

        ## get the contour corresponding to Jupiter assuming
        ##    that Jupiter is closest to the center
        pathi = find_best_path(paths, halfnx, halfny)

        ## change the pathi if it detects some other contour 
        jupcont = paths[pathi].vertices

        limbRADec   = self.limbRADec.copy()
        self.RADec0 = np.array([self.subRA, self.subDec])

        ## fit the image
        self.platecal.scale, self.platecal.alpha, self.R2P0  = \
            fit_ellipse_auto(jupcont, limbRADec, \
                             self.RADec0, self.RArotang)

        self.platecal.scale = np.abs(self.platecal.scale)
        while (self.platecal.alpha < 0.):
            self.platecal.alpha += 2.*np.pi
        while(self.platecal.alpha > 2.*np.pi):
            self.platecal.alpha -= 2.*np.pi

        ## get the transformation matrix
        self.tR2PIX = self.platecal.get_transform_matrix()
        self.tPIX2R = np.linalg.inv(self.tR2PIX)
        
        ## plot the guides to check the fit
        limb = self.radec2pix(limbRADec)

        ## check the North pole position
        dNorth = limb[0,:] - self.R2P0

        if(\
            ((dNorth[1]>0.)&(NorthUp)) or \
            ((dNorth[1]<0.)&(~NorthUp)) \
            ):
            self.tR2PIX = np.matmul( np.array([[-1, 0.],[0., -1]]), \
                                    self.tR2PIX )

        #axtest.imshow(img_gamma)
        ax = axtest; ax.cla()
        #fig, ax = plt.subplots(1,1, figsize=(10,10))

        ax.imshow(self.img)

        ## South-Pole
        NPJ2000 = self.get_surf_vec(0., np.pi/2.)
        NPRADec = spiceypy.recrad(NPJ2000)[1:]
        NP      = self.radec2pix(np.asarray(NPRADec))

        ## South-Pole
        SPJ2000 = self.get_surf_vec(0., -np.pi/2.)
        SPRADec = spiceypy.recrad(SPJ2000)[1:]
        SP      = self.radec2pix(np.asarray(SPRADec))

        ## meridian at sub-obs longitude
        lats = np.radians(np.linspace(-80., 80., 50))
        meridian = np.zeros((lats.shape[0], 2))
        for i, lati in enumerate(lats):
            meridianJ2000 = self.get_surf_vec(self.subptlon, lati)
            meridianRADec = np.asarray(spiceypy.recrad(meridianJ2000)[1:])

            meridian[i,:] = self.radec2pix(meridianRADec)
        
        ## equator
        lons = np.radians(np.linspace(-80., 80., 50)) + self.subptlon
        eq = np.zeros((lons.shape[0], 2))
        for i, loni in enumerate(lons):
            eqJ2000 = self.get_surf_vec(loni, 0.)
            eqRADec = np.asarray(spiceypy.recrad(eqJ2000)[1:])

            eq[i,:] = self.radec2pix(eqRADec)
        
        ## -22deg S (GRS band)
        S22 = np.zeros((lons.shape[0], 2))
        for i, loni in enumerate(lons):
            S22J2000 = self.get_surf_vec(loni, np.radians(-22.))
            S22RADec = np.asarray(spiceypy.recrad(S22J2000)[1:])

            S22[i,:] = self.radec2pix(S22RADec)
        
        ## 24deg N NTrZ
        N24 = np.zeros((lons.shape[0], 2))
        for i, loni in enumerate(lons):
            N24J2000 = self.get_surf_vec(loni, np.radians(24.))
            N24RADec = np.asarray(spiceypy.recrad(N24J2000)[1:])

            N24[i,:] = self.radec2pix(N24RADec)

        ax.plot(jupcont[:,0], jupcont[:,1], 'r-')
        ax.plot(limb[:,0],    limb[:,1], '-', color='white',\
                    linewidth=0.5)
        ax.plot(SP[0],        SP[1], 'gx')
        ax.plot(NP[0],        NP[1], 'go')
        ax.plot(meridian[:,0],meridian[:,1], 'b-')
        ax.plot(eq[:,0],eq[:,1], 'b-')
        ax.plot(S22[:,0],S22[:,1], 'g-')
        ax.plot(N24[:,0],N24[:,1], 'g-')
        
        plt.tight_layout()
        plt.show()

    def fit_img_manual(self, scalelim=(5.,25.), alphalim=(0., 360.), \
                       x0lim=(-1,-1), y0lim=(-1,-1)):
        '''
            fit the target image manually using sliders for pixel scale,
            rotation angle and target center

            Parameters
            ----------
            scalelim : tuple
                minimum and maximum limits of pixel scale (arsec/pixel)
            alphalim : tuple
                minimum and maximum limits of rotation angle (degree)
            x0lim : tuple
                minimum and maximum limits of x center
            y0lim : tuple
                minimum and maximum limits of y center
        '''
        limb = self.limbRADec[::5]# - self.RADec0

        ## South-Pole
        SPJ2000 = self.get_surf_vec(0., -np.pi/2.)
        SPRADec = np.array(spiceypy.recrad(SPJ2000)[1:])# - self.RADec0
        
        ## South-Pole
        NPJ2000 = self.get_surf_vec(0., np.pi/2.)
        NPRADec = np.array(spiceypy.recrad(NPJ2000)[1:])# - self.RADec0

        ## meridian at sub-obs longitude
        lats = np.radians(np.linspace(-80., 80., 25))
        meridianRADec = np.zeros((lats.shape[0], 2))
        for i, lati in enumerate(lats):
            meridianJ2000 = self.get_surf_vec(self.subptlon, lati)
            meridianRADec[i,:] =\
                np.asarray(spiceypy.recrad(meridianJ2000)[1:])

        fig, ax = plt.subplots(1, 1, figsize=(10,8))

        img = ax.imshow(self.img.min(axis=2), cmap='gray')

        gammaslider = widgets.FloatSlider(\
                        value=1., \
                        min=0.1, max=1., step=0.01)

        update_img = lambda gamma: \
            self.update_image(ax, img, gamma)

        update = lambda scale, alpha, x0, y0: \
            self.update_ellipse(limb, meridianRADec, SPRADec, NPRADec, \
                                ax, scale, alpha, x0, y0)

        if(min(x0lim)<0.):
            x0lim = (0., self.nx)
        if(min(y0lim)<0.):
            y0lim = (0., self.ny)

        ## create the sliders for the values
        ## use the stored values as default to make 
        ## sure that we can continue from the previous call
        scaleslider = widgets.FloatSlider(\
                        value=self.platecal.scale/RAD2ARCSEC, \
                        min=scalelim[0], max=scalelim[1], step=0.01)
        alphaslider = widgets.FloatSlider(\
                        value=np.degrees(self.platecal.alpha), \
                        min=alphalim[0], max=alphalim[1], step=0.01)
        x0slider = widgets.FloatSlider(\
                        value=self.R2P0[0], \
                        min=x0lim[0], max=x0lim[1], step=0.01)
        y0slider = widgets.FloatSlider(\
                        value=self.R2P0[1], \
                        min=y0lim[0], max=y0lim[1], step=0.01)

        widgets.interact(update_img, \
                         gamma=gammaslider)
        widgets.interact(update, \
                         scale=scaleslider, alpha=alphaslider,\
                         x0=x0slider, y0=y0slider)

    def update_image(self, ax, img, gamma):
        try: 
            R = self.img[:,:,0]; G = self.img[:,:,1]; B = self.img[:,:,2]
            Rg = np.clip(np.power(R, gamma), 0, 1)
            Gg = np.clip(np.power(G, gamma), 0, 1)
            Bg = np.clip(np.power(B, gamma), 0, 1)
           
            img_gamma = np.min([Rg, Gg, Bg], axis=0)
        except IndexError:

            img_gamma = np.clip(np.power(self.img, gamma), 0, 10)


        img.set_data(img_gamma)


    def update_ellipse(self, limb, meridian, SP, NP, \
                       ax, scale, alpha, x0, y0):
        '''
            update the image for the manual calibration

            Parameters:
            limb : numpy.ndarray
                RA/Dec points of the limb
            meridian : numpy.ndarray
                RA/Dec points of the meridian at the sub-obs longitude
            SP : numpy.ndarray
                RA/Dec points of the south pole
            NP : numpy.ndarray
                RA/Dec points of the north pole
            axes : matplotlib.axes.Axes
                matplotlib axis to update the figure
            scale : float
                pixel scale in arsec/pixel
            alpha : float
                rotation angle of the image (degree)
            x0 : float
                target center x-value
            y0 : float
                target center y-value
        '''
        ## remove the old lines first
        [l.remove() for l in ax.lines]

        ## update the calibration with the new values
        self.platecal.scale = scale*RAD2ARCSEC
        self.platecal.alpha = np.radians(alpha)
        self.R2P0           = np.array([x0, y0])

        self.tR2PIX = self.platecal.get_transform_matrix()
        self.tPIX2R = np.linalg.inv(self.tR2PIX)

        ## transform the points
        limbpix     = self.radec2pix(limb)
        meridianpix = self.radec2pix(meridian)
        SPpix       = self.radec2pix(SP)
        NPpix       = self.radec2pix(NP)

        ## sanity check
        [l.remove() for l in ax.lines]

        ## plot out the new lines
        ax.plot(limbpix[:,0], limbpix[:,1], 'r-')
        ax.plot(meridianpix[:,0], meridianpix[:,1], 'b-')
        ax.plot(SPpix[0], SPpix[1], 'gx')
        ax.plot(NPpix[0], NPpix[1], 'go')

    def project(self, get_illum=True, use_full_image=False):
        '''
            retrieve the planetographic coordinates of the center 
            of each pixel in the image

            Parameters
            ----------
            get_illum : bool
                flag to also retrieve illumination angles (incidence,
                emission and phase), and get a Lambertian correction 
                [default: True]
            use_full_image : bool
                flag to process the full image, or use a bounding box
                of 1.2*(planetary radius) around the center to speed up
                computation. Turn off if bounding box produces errors.
                [default: False]
        '''
        if not use_full_image:
            maxsize = np.max([self.obsa, self.obsb])*self.platecal.scale

            xstart, ystart = np.asarray(self.R2P0 - 1.2*maxsize, dtype=int)
            xend, yend     = np.asarray(self.R2P0 + 1.2*maxsize, dtype=int)
            xstart = max([xstart, 0])
            ystart = max([ystart, 0])
            xend   = min([xend, self.nx])
            yend   = min([yend, self.ny])
        else:
            xstart = 0
            xend   = self.nx
            ystart = 0
            yend   = self.ny

        ## create the empty arrays to hold the values
        imgshape = self.img.shape[:2]
        self.lat = -1000.*np.ones(imgshape)
        self.lon = -1000.*np.ones(imgshape)

        if get_illum:
            self.incidence  = np.zeros(imgshape)
            self.emission   = np.zeros(imgshape)
            self.phase      = np.zeros(imgshape)
            self.solar_corr = np.zeros(imgshape)

        for j in range(ystart, yend):
            if(j%10 == 0):
                print("\r %3d/%3d"%(j, yend), end='')
            for i in range(xstart, xend):

                ## find the vector that refers to this pixel 
                pix     = np.array([i,j], dtype=float)
                RADeci  = self.pix2radec(pix)
                veci = spiceypy.radrec(1., RADeci[0], RADeci[1])

                ## check for the intercept
                try:
                    spoint, ep, srfvec = \
                        spiceypy.sincpt(\
                            "Ellipsoid", self.target, self.et,\
                            self.target_frame, "CN", "EARTH",\
                            "J2000", veci)
                except Exception as e: 
                    continue

                ## if the intercept works, determine the planetographic
                ## lat/lon values
                loni, lati, alt = \
                    spiceypy.recpgr(self.target, spoint, \
                                    self.radii[0], self.flattening)

                self.lat[j,i] = np.degrees(lati)
                self.lon[j,i] = np.degrees(loni)

                if get_illum:
                    _, _, phasei, inci, emissi = \
                        spiceypy.ilumin("Ellipsoid", self.target, \
                                self.et, self.target_frame, "CN",\
                                "EARTH", spoint) 

                    ## apply Lambertian correction
                    mu = np.cos(emissi); mu0 = np.cos(inci)
                    self.solar_corr[j,i] = 1./mu0

                    ## save the 
                    self.phase[j,i]      = phasei
                    self.incidence[j,i]  = inci
                    self.emission[j,i]   = emissi

        img = self.img.copy()
        if get_illum:
            if(len(img.shape) > 2):
                for i in range(img.shape[2]):
                    img[:,:,i] *= self.solar_corr


        ## plot them out
        fig = plt.figure(figsize=(8,15))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        ax1.imshow(self.lon, vmin=0., vmax=360.)
        ax1.contour(self.lon, np.arange(0., 360., 15.), colors='k')
        ax2.imshow(self.lat, vmin=-90., vmax=90.)
        ax2.contour(self.lat, np.arange(-90, 90., 15.), colors='k')
        ax3.imshow(img)
        ax3.contour(self.lon, np.arange(0., 360., 30.), colors='k', linewidths=0.5)
        ax3.contour(self.lat, np.arange(-90, 90., 30.), colors='k', linewidths=0.5)


        ax1.set_title("Longitude")
        ax2.set_title("Latitude")

        plt.tight_layout()
        plt.show()
