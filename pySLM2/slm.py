import tensorflow as tf
import numpy as np
from scipy.constants import micro
from functools import lru_cache
from .profile import FunctionProfile
from . import _lib
from ._backend import BACKEND

__all__ = ["SLM", "DMD", "DLP7000", "DLP9500"]


class SLM(object):
    """Main class for any spatial light modulators."""

    def __init__(self, wavelength, focal_length, Nx, Ny, pixel_size):
        # Read only
        self._wavelength = wavelength
        self._focal_length = focal_length
        self._Nx = Nx
        self._Ny = Ny
        self._scaling_factor = self._wavelength * self._focal_length
        self._pixel_size = pixel_size

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

    @property
    def pixel_size(self):
        return self._pixel_size

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
        return j * self._pixel_size, i * self._pixel_size

    @tf.function
    def _convert_slm_coordinate_to_pixel_index(self, x, y):
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
        x = x / self._pixel_size
        y = y / self._pixel_size
        i = y + self._Ny / 2
        j = x + self._Nx / 2
        i = self._Ny - i
        return i, j

    @lru_cache()
    def _fourier_plane_pixel_grid(self):
        pix_j = tf.range(self.Nx, dtype=BACKEND.dtype)
        pix_i = tf.range(self.Ny, dtype=BACKEND.dtype)
        return tf.meshgrid(pix_i, pix_j, indexing="ij")

    @lru_cache()
    def _fourier_plane_grid(self):
        pix_ii, pix_jj = self._fourier_plane_pixel_grid()
        return self._convert_pixel_index_to_dmd_coordinate(pix_ii, pix_jj)

    @property
    def fourier_plane_grid(self):
        x, y = self._fourier_plane_grid()
        return np.array(x), np.array(y)

    @lru_cache()
    def _image_plane_grid(self):
        kx_atom, ky_atom = tf.constant(np.fft.fftfreq(self.Nx, self._pixel_size), dtype=BACKEND.dtype), \
                           tf.constant(-np.fft.fftfreq(self.Ny, self._pixel_size), dtype=BACKEND.dtype)

        kx_atom, ky_atom = tf.signal.fftshift(kx_atom), tf.signal.fftshift(ky_atom)
        x_atom = kx_atom * self.scaling_factor
        y_atom = ky_atom * self.scaling_factor
        return tf.meshgrid(x_atom, y_atom)

    @property
    def image_plane_grid(self):
        x, y = self._image_plane_grid()
        return np.array(x), np.array(y)

    def _profile_to_tensor(self, profile, at_fourier_plane=True, complex=False):
        tensor_dtype = BACKEND.dtype_complex if complex else BACKEND.dtype
        if isinstance(profile, FunctionProfile):
            grid = self._fourier_plane_grid() if at_fourier_plane else self._image_plane_grid()
            tensor = profile._func(*grid)
        elif isinstance(profile, tf.Tensor):
            tensor = profile
        else:
            tensor = tf.constant(profile, dtype=tensor_dtype)
        if tensor.dtype != tensor_dtype:
            tensor = tf.cast(tensor, dtype=tensor_dtype)
        return tensor


class DMD(SLM):
    def __init__(self, wavelength, focal_length, periodicity, theta, Nx, Ny, micromirror_size,
                 negative_order=False):
        """




               +----->  j

         +     +------------------+
         |     |        y         |
         |     |        ^         |
         v     |        |         |
               |        0---> x   |
         i     |                  |
               |                  |
               +------------------+

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
        super().__init__(wavelength, focal_length, Nx, Ny, micromirror_size)

        self.dmd_state = np.zeros((self.Ny, self.Nx), dtype=np.bool)

        self._p = tf.Variable(periodicity * self._pixel_size, dtype=BACKEND.dtype)
        self._theta = tf.Variable(theta, dtype=BACKEND.dtype)

        self.negative_order = negative_order

    @property
    def micromirror_size(self):
        return self._pixel_size

    def set_dmd_state_off(self):
        """ Reset dmd_state to be an array (Ny, Nx) of zeros."""
        self.dmd_state = np.zeros((self.Ny, self.Nx), dtype=np.bool)

    def set_dmd_state_on(self):
        """Reset dmd_state to be an array (Ny, Nx) of ones."""
        self.dmd_state = np.ones((self.Ny, self.Nx), dtype=np.bool)

    def set_dmd_grating_state(self, amp=1, phase_in=0, phase_out=0, method="random", **kwargs):
        amp = self._profile_to_tensor(amp)
        phase_in = self._profile_to_tensor(phase_in)
        phase_out = self._profile_to_tensor(phase_out)

        x, y = self._fourier_plane_grid()
        self.dmd_state = np.array(_lib.calculate_dmd_grating(amp, phase_in, phase_out, x, y, self._p, self._theta,
                                                             method=method, negative_order=self.negative_order,
                                                             **kwargs))

    @tf.function
    def _circular_mask(self, i, j, pix_ii, pix_jj, d):
        pix_rr_square = (pix_jj - j) ** 2 + (pix_ii - i) ** 2
        mask = pix_rr_square < (d / 2) ** 2
        return mask

    def circular_patch(self, i, j, amp, phase_in, phase_out, d, method="random", **kwargs):
        """

        Parameters
        ----------
        i
        j
        amp
        phase_out
        phase_in
        d
        binarize
        kwargs

        Returns
        -------

        """

        amp = self._profile_to_tensor(amp)
        phase_in = self._profile_to_tensor(phase_in)
        phase_out = self._profile_to_tensor(phase_out)

        pix_ii, pix_jj = self._fourier_plane_pixel_grid()
        mask = self._circular_mask(i, j, pix_ii, pix_jj, d)

        x, y = self._fourier_plane_grid()

        # TODO mask the array before passing them into the function
        dmd_state = np.array(_lib.calculate_dmd_grating(amp, phase_in, phase_out, x, y, self._p, self._theta,
                                                        method=method, negative_order=self.negative_order, **kwargs))

        self.dmd_state[mask] = dmd_state[mask]

    @tf.function
    def _calc_amp_phase(self, input_profile, target_profile):
        target_profile_fp = _lib._fourier_transform(tf.signal.ifftshift(target_profile))
        target_profile_fp = tf.signal.fftshift(target_profile_fp)

        phase_in = tf.math.angle(input_profile)
        amp_in = tf.math.abs(input_profile)
        phase_out = tf.math.angle(target_profile_fp)
        amp_out = tf.math.abs(target_profile_fp)

        amp_scaled = amp_out / amp_in
        amp_scaled = amp_scaled / tf.math.reduce_max(amp_scaled)

        return amp_scaled, phase_in, phase_out

    def calculate_dmd_state(self, input_profile, target_profile, method="random", **kwargs):
        # TODO check kwargs for different method
        input_profile = self._profile_to_tensor(input_profile, complex=True)
        target_profile = self._profile_to_tensor(target_profile, at_fourier_plane=False, complex=True)
        amp_scaled, phase_in, phase_out = self._calc_amp_phase(input_profile, target_profile)

        x, y = self._fourier_plane_grid()

        if method == "ifta":
            kwargs["input_profile"] = input_profile
            kwargs["signal_window"] = self._profile_to_tensor(kwargs["signal_window"], at_fourier_plane=False,
                                                              complex=True)

        self.dmd_state = np.array(
            _lib.calculate_dmd_grating(amp_scaled, phase_in, phase_out, x, y, self._p, self._theta,
                                       method=method, negative_order=self.negative_order,
                                       **kwargs))

    @property
    def p(self):
        return self._p.value()

    @property
    def theta(self):
        return self._theta.value()

    @property
    def first_order_origin(self):
        """Find the origin of the first order light in image plane.

        Returns
        -------
        origin_x: float
            The x coordinate of the origin of first order light in image plane.
        origin_y: float
            The y coordinate of the origin of first order light in image plane.
        """
        origin_x = np.cos(self.theta) * self.scaling_factor / self.p
        origin_y = np.sin(self.theta) * self.scaling_factor / self.p
        return origin_x, origin_y


class DLP9500(DMD):
    def __init__(self, wavelength, focal_length, periodicity, theta, negative_order=False):
        super(DLP9500, self).__init__(wavelength, focal_length, periodicity, theta,
                                      1920, 1080, 10.8 * micro, negative_order=negative_order)


class DLP7000(DMD):
    def __init__(self, wavelength, focal_length, periodicity, theta, negative_order=False):
        super(DLP7000, self).__init__(wavelength, focal_length, periodicity, theta,
                                      1024, 768, 13.6 * micro, negative_order=negative_order)
