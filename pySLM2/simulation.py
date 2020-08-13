from functools import lru_cache
import numpy as np
import tensorflow as tf
from .slm import SLM, DMD
from ._backend import BACKEND

__all__ = ["SLMSimulation", "DMDSimulation"]


class SLMSimulation(object):
    def __init__(self, slm, padding_x=0, padding_y=0):
        """


        Fourier Plane

             +----------------------+ ^
             |                      | | padding_y
             |                      | |
             |    +------------+    | v
             |    |            |    |
             |    | DMD grid   |    |
             |    |            |    |
             |    +------------+    |
             |                      |
             |   padded grid        |
             +----------------------+
                               <---->
                                padding_x


        Parameters
        ----------
        dmd
        padding_x
        padding_y
        """

        if not isinstance(padding_x, int):
            raise TypeError("padding_x must be an integer.")

        if not isinstance(padding_y, int):
            raise TypeError("padding_x must be an integer.")

        if isinstance(slm, SLM):
            self._slm = slm
        else:
            raise TypeError("dmd must be a SLM object.")

        self._padding_x = padding_x
        self._padding_y = padding_y

        self._input_field = None
        self._output_field = None
        self._image_plane_field = None

    def clear(self):
        self._input_field = None
        self._output_field = None
        self._image_plane_field = None

    @property
    def fourier_plane_pixel_area(self):
        """float"""
        return self._slm._pixel_size ** 2

    @property
    def image_plane_pixel_area(self):
        return self._slm.scaling_factor ** 2 / (self._slm.Nx + 2 * self._padding_x) / (
                    self._slm.Ny + 2 * self._padding_y) / self._slm.pixel_size ** 2

    @property
    def padding_x(self):
        return self._padding_x

    @property
    def padding_y(self):
        return self._padding_y

    @lru_cache()
    def _image_plane_padded_grid(self):
        kx_atom, ky_atom = tf.constant(np.fft.fftfreq(self._slm.Nx + 2 * self._padding_x, self._slm.pixel_size),
                                       dtype=BACKEND.dtype), \
                           tf.constant(-np.fft.fftfreq(self._slm.Ny + 2 * self._padding_y, self._slm.pixel_size),
                                       dtype=BACKEND.dtype)

        kx_atom, ky_atom = tf.signal.fftshift(kx_atom), tf.signal.fftshift(ky_atom)
        x_atom = kx_atom * self._slm.scaling_factor
        y_atom = ky_atom * self._slm.scaling_factor
        return tf.meshgrid(x_atom, y_atom)

    @property
    def image_plane_padded_grid(self):
        x, y = self._image_plane_padded_grid()
        return np.array(x), np.array(y)

    @lru_cache()
    def _fourier_plane_padded_grid(self):
        pix_j = tf.range(-self._padding_x, self._slm.Nx + self._padding_x, dtype=BACKEND.dtype)
        pix_i = tf.range(-self._padding_y, self._slm.Ny + self._padding_y, dtype=BACKEND.dtype)

        pix_ii, pix_jj = tf.meshgrid(pix_i, pix_j, indexing="ij")

        return self._slm._convert_pixel_index_to_dmd_coordinate(pix_ii, pix_jj)

    @property
    def fourier_plane_padded_grid(self):
        x, y = self._fourier_plane_padded_grid()
        return np.array(x), np.array(y)


class DMDSimulation(SLMSimulation):
    def __init__(self, dmd, **kwargs):
        if not isinstance(dmd, DMD):
            raise TypeError("dmd must be a DMD object.")

        super(DMDSimulation, self).__init__(dmd, **kwargs)