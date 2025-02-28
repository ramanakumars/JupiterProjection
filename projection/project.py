import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from .spice_utils import get_kernels, KERNEL_DATAFOLDER
import spiceypy as spice
import tqdm
import os
from skimage.io import imread
from skimage.measure import find_contours
from .fit_utils import get_ellipse_params


KECK_LOCATION = (np.radians(19.8263), -np.radians(155.47441))


def gamma_correct(img, gamma=0.1):
    img_norm = img / 65536
    return img_norm ** gamma


def flip_north_south(wcs):
    '''
        Flips the WCS rotation by 180deg to fix north/south alignment issues.
        Use if the WCS fit is not lining up with the north/south pole.
    '''
    wcs.wcs.pc = np.matmul(wcs.wcs.pc, [[-1, 0], [0, -1]])

    return wcs


class ImageProjection:
    '''
        Functions to fit and project the MAGIQ guider images
    '''
    def __init__(self, filename: str, date: str, target: str = 'JUPITER'):
        '''
            Initialize the projector by loading the data and the spice kernels.
            Also load metadata for the positioning data (for Keck and also for Jupiter)

            Inputs
            ------
            filename : str
                Path to the FITS file
            target : str
                Target observing body (currently only works for Jupiter)
        '''
        self.filename = filename

        # read in the data and the observation time
        self.data = imread(self.filename)
        self.obstime = Time(date)

        kernels = get_kernels(KERNEL_DATAFOLDER, 'jupiter')

        for kernel in kernels:
            spice.furnsh(kernel)

        self.et = spice.utc2et(self.obstime.to_datetime().isoformat())

        # calculate target information
        self.target = target
        self.target_frame = 'IAU_' + target
        self.radii = spice.bodvar(spice.bodn2c(target), "RADII", 3)
        self.flattening = (self.radii[0] - self.radii[2]) / self.radii[0]
        self.find_sub_pt()

        # set default parameters based on what we know
        # about the guider camera
        self.wcs = WCS({
            'NAXIS': 2,
            'NAXIS1': self.data.shape[0],
            'NAXIS2': self.data.shape[1],
            'CRPIX1': self.data.shape[0] // 2,
            'CRPIX2': self.data.shape[1] // 2,
            'CRVAL1': np.degrees(self.subpt[0]),
            'CRVAL2': np.degrees(self.subpt[1]),
            'CDELT1': 1e-4,
            'CDELT2': 1e-4,
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CUNIT1': 'deg',
            'CUNIT2': 'deg'
        })
        
        self.find_limb()

    def detect_limb(self, gamma: float = 0.1, threshold: float = 0.7) -> np.ndarray:
        '''
            Find the limb in the observation. Returns a set of coordinates containing the limb

            Inputs
            ------
            gamma : float
                Parameter for gamma stretching the image. Use lower values to highlight the limb more
            threshold : float
                Threshold for discerning between limb and background.

            Outputs
            -------
            limb : np.ndarray
                Set of coordinates (X/Y) for the detected limb

        '''
        data = gamma_correct(self.data.mean(-1), gamma)
        range = data.max() - data.min()
        contours = find_contours(data, threshold * range + data.min())
        best_contour = np.argmax([np.linalg.norm(np.trapz(cont, axis=0)) for cont in contours])

        return contours[best_contour][:, ::-1]

    def fit_limb_from_contour(self, contour: np.ndarray) -> WCS:
        '''
            Fit the ellipse given a set of contour points of the limb. Returns the WCS fit.

            Inputs
            ------
            contour : np.ndarray
                Set of coordinates (X/Y) for the limb

            Outputs
            -------
            wcs: WCS
                The WCS fit for the limb
        '''
        wcs = WCS(self.wcs.to_header())

        return get_ellipse_params(contour, self.limbRADec, self.subpt, wcs, self.data.shape[:2])

    def update_fits_wcs(self, wcs: WCS) -> None:
        '''
            Save the input WCS parameter to the FITS file

            Inputs
            ------
            wcs : WCS
                The updated WCS fits
        '''
        with fits.open(self.filename, 'update') as hdulist:
            hdulist[0].header.update(wcs.to_header())

    def find_sub_pt(self):
        '''
            Finds the sub-observer point in the IAU_[target] frame
            and also the in the J2000 frame as seen from Earth.
            Also creates a vector from Earth to the sub-obs point in the
            J2000 frame for future use.
        '''

        # get the position of the sub-obs point in the J2000 frame
        self.subptvec, self.subptep, self.subpntobsvec = spice.subpnt('INTERCEPT/ELLIPSOID', self.target,
                                                                      self.et, self.target_frame, 'CN+S', "EARTH")

        # convert to lat/lon
        self.subptlon, self.subptlat, _ = spice.recpgr(self.target, self.subptvec, self.radii[0], self.flattening)

        # convert the line of sight vector to J2000
        px1 = spice.pxfrm2(self.target_frame, 'J2000', self.subptep, self.et)

        self.subptJ2000 = np.matmul(px1, self.subpntobsvec)

        # get the RA/DEC of the sub-obs point
        self.subpt = np.asarray(spice.recrad(self.subptJ2000)[1:])

    def find_limb(self):
        '''
            Get the limb and corresponding parameters (epoch, distance, vector) for the planet
            given the observing date
        '''
        rolstep = np.radians(5)
        ncuts = int(2. * np.pi / rolstep)
        _, limbs, eplimb, vecs = spice.limbpt('TANGENT/ELLIPSOID', self.target, self.et, self.target_frame,
                                              'CN+S', "ELLIPSOID LIMB", "EARTH", np.asarray([0, 0, 1]), rolstep,
                                              ncuts, 1e-4, 1e-7, ncuts)

        self.limbJ2000 = np.zeros_like(vecs)
        self.limbRADec = np.zeros((ncuts, 2))
        self.limbdist = np.zeros(ncuts)
        for i in range(ncuts):
            # get the transformation of the limb points to the J2000 frame
            pxi = spice.pxfrm2(self.target_frame, 'J2000', eplimb[i], self.et)

            # transform the vectors from the observer to the limb to J2000
            self.limbJ2000[i, :] = np.matmul(pxi, vecs[i, :])

            # also convert to RA/Dec
            self.limbdist[i], self.limbRADec[i, 0], self.limbRADec[i, 1] = spice.recrad(self.limbJ2000[i, :])

    def project_to_lonlat(self):
        ny, nx, _ = self.data.shape

        self.lonlat = np.nan * np.zeros((ny, nx, 2))

        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        radecs = self.wcs.pixel_to_world(X.flatten(), Y.flatten())

        for n, (i, j) in enumerate(tqdm.tqdm(zip(X.flatten(), Y.flatten()), total=X.size)):
            ra = radecs[n].ra
            dec = radecs[n].dec
            veci = spice.radrec(1., ra.radian, dec.radian)

            # check for the intercept
            try:
                spoint, ep, srfvec = spice.sincpt("Ellipsoid", self.target, self.et,
                                                  self.target_frame, "CN+S", "EARTH",
                                                  "J2000", veci)
            except Exception:
                continue

            # if the intercept works, determine the planetographic
            # lat/lon values
            loni, lati, alt = spice.recpgr(self.target, spoint, self.radii[0], self.flattening)

            self.lonlat[j, i, :] = np.degrees(loni), np.degrees(lati)
