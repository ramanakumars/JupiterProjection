import numpy as np
from scipy import optimize
from shapely.geometry import Polygon
import logging

logger = logging.getLogger(__name__)

RAD2ARCSEC = 206265.


def get_ellipse_params(v, limb, subpt, wcs, img_size):
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
    # radius-scale factor in the RA direction
    RR = np.abs(np.cos(subpt[1]))
    crpix = wcs.wcs.crpix
    scale = wcs.wcs.cdelt.mean()
    alpha = 0.

    # solution vector (A, X0, Y0, alpha)
    vmean = np.mean(v, axis=0)
    X = np.zeros(5)
    initial_scale = np.mean(np.linalg.norm([rr - subpt for rr in limb], axis=1)) / np.mean(np.linalg.norm([rr - vmean for rr in v], axis=1))
    initial_scale = initial_scale
    X[0] = scale / initial_scale - 1
    X[1] = scale / initial_scale - 1
    X[2] = crpix[0] / img_size[1] - 0.5
    X[3] = crpix[1] / img_size[0] - 0.5
    X[4] = alpha / np.pi

    # calculate the vectors in the RA/Dec frame
    # convert to arcsec so that the convergence
    # is better
    dRA = RR * (limb[:, 0] - subpt[0])
    dDec = (limb[:, 1] - subpt[1])

    dRADec = np.asarray([dRA, dDec])
    shape_limb = Polygon(v)

    def func(inp):
        A, B, X0, Y0, alpha = inp

        alpha = alpha * np.pi
        A = (A + 1) * initial_scale
        B = (B + 1) * initial_scale
        X0 = (X0 + 0.5) * img_size[1]
        Y0 = (Y0 + 0.5) * img_size[0]

        rotation_matrix_plate = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        b = np.matmul(rotation_matrix_plate, dRADec).T
        b[:, 0] = b[:, 0] / A
        b[:, 1] = b[:, 1] / B
        b = b + np.array([X0, Y0])

        shape_limb_fit = Polygon(b)

        intersection = shape_limb.intersection(shape_limb_fit).area
        union = shape_limb.union(shape_limb_fit).area

        return (1 - intersection / union)

    bounds = [
        [-1, 1],
        [-1, 1],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-1, 1],
    ]

    output = optimize.dual_annealing(func, bounds, x0=X)
    A, B, x0, y0, alpha = output.x

    def func_alpha(inp):
        return func([A, B, x0, y0, inp[0]])

    # improve the convergence by finding a smaller range in alpha (the plate rotation)
    output = optimize.dual_annealing(func_alpha, [[alpha - abs(alpha) * 0.1, alpha + abs(alpha) * 0.1]], x0=[alpha])

    logging.debug(output.x, output.fun)
    alpha = output.x[0]
    A = (A + 1) * initial_scale
    B = (B + 1) * initial_scale
    alpha = alpha * np.pi

    pc = np.asarray([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    wcs.wcs.pc = pc
    wcs.wcs.crval = np.degrees(subpt)
    wcs.wcs.crpix = (np.asarray([x0, y0]) + 0.5) * img_size[::-1]
    wcs.wcs.cdelt = np.asarray([np.abs(A), np.abs(B)]) * 180. / np.pi

    return wcs
