import tensorflow as tf
import numpy as np
from scipy.constants import micro
from .profile import FunctionProfile
from .lib import *

class SLM(object):
    """Main class for any spatial light modulators."""

    def __init__(self, wavelength, focal_length, Nx, Ny):
        # Read only
        self._wavelength = wavelength
        self._focal_length = focal_length
        self._Nx = Nx
        self._Ny = Ny
        self._scaling_factor = self._wavelength * self._focal_length

    @property
    def scaling_factor(self):
        return self._scaling_factor

    @property
    def Nx(self):
        return self._Nx

    @property
    def Ny(self):
        return self._Ny

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def focal_length(self):
        return self._focal_length


class DMD(SLM):
    def __init__(self, wavelength, focal_length, periodicity, theta, Nx, Ny, micromirror_size,
                 negative_order=False):
        """

        Parameters
        ----------
        wavelength: float
            Wavelength of indicent light.
        focal_length: float
            Focal length of the focusing lens
        periodicity: float.
            The periodicity of the grating profile. Unit: pixel (micromirror size)
        theta: float
            Desired grating angle
        Nx: int
            Number of pixels in x direction
        Ny: int
            Number of pixels in y direction
        micromirror_size: float
            The size of one micromirror. Unit: m
        negative_order: bool
            True: use negative first order instead of first order diffraction beam.
        """
        super().__init__(wavelength, focal_length, Nx, Ny)

        self.dmd_state = np.zeros((self.Ny, self.Nx), dtype=np.bool)

        self._micromirror_size = micromirror_size

        # TODO: adjust for differences in x/y size
        self.p = periodicity * micromirror_size
        self.theta = theta
        self.negative_order = negative_order

    @property
    def micromirror_size(self):
        return self._micromirror_size

    def set_dmd_state_off(self):
        """ Reset dmd_state to be an array (Ny, Nx) of zeros."""
        self.dmd_state = np.zeros((self.Ny, self.Nx), dtype=np.bool)

    def set_dmd_state_on(self):
        """Reset dmd_state to be an array (Ny, Nx) of ones."""
        self.dmd_state = np.ones((self.Ny, self.Nx), dtype=np.bool)

    @tf.function
    def _convert_pixel_index_to_dmd_coordinate(self, i, j):
        """

        Parameters
        ----------
        i : float or :obj:numpy.ndarray

        j : float or :obj:numpy.ndarray

        Returns
        -------1
        x: float or :obj:numpy.ndarray

        y: float or :obj:numpy.ndarray

        """
        i = self._Ny - i
        j = j - self._Nx / 2
        i = i - self._Ny / 2
        return j * self._micromirror_size, i * self._micromirror_size

    @tf.function
    def _convert_dmd_coordinate_to_pixel_index(self, x, y):
        """

        Parameters
        ----------
        x : float or :obj:numpy.ndarray

        y : float or :obj:numpy.ndarray

        Returns
        -------
        j: float or :obj:numpy.ndarray

        j: float or :obj:numpy.ndarray

        """
        x = x / self._micromirror_size
        y = y / self._micromirror_size
        i = y + self._Ny / 2
        j = x + self._Nx / 2
        i = self._Ny - i
        return i, j

    @property
    def fourier_plane_grid(self):
        pix_x = tf.range(self.Nx, dtype=tf.float32)
        pix_y = tf.range(self.Ny, dtype=tf.float32)
        x_dmd, y_dmd = self._convert_pixel_index_to_dmd_coordinate(pix_y, pix_x)
        xx_d, yy_d = np.meshgrid(x_dmd, y_dmd)
        return xx_d, yy_d

    def set_dmd_grating_state(self, phase_in=0, phase_out=0, method="random", negative_order=False):
        px, py = self.fourier_plane_grid
        self.dmd_state = np.array(calculate_dmd_grating(1, phase_in, phase_out, px, py, self.p, self.theta,
                                                        method=method, negative_order=negative_order))

    def image_plane_grid(self):
        pass

    def _calculate_dmd_grating(self, amp, phase_in, phase_out):
        pass




class DLP9500(DMD):
    def __init__(self, wavelength, focal_length, periodicity, theta, negative_order=False):
        super(DLP9500, self).__init__(wavelength, focal_length, periodicity, theta,
                                      1920, 1080, 10.8 * micro, negative_order=negative_order)

class DLP7000(DMD):
    def __init__(self, wavelength, focal_length, periodicity, theta, negative_order=False):
        super(DLP7000, self).__init__(wavelength, focal_length, periodicity, theta,
                                      1024, 768, 13.6 * micro, negative_order=negative_order)

