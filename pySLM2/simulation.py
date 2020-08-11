from functools import lru_cache
import numpy as np
import tensorflow as tf
from .slm import DMD
from ._backend import BACKEND

__all__ = ["DMDSimulation"]

class DMDSimulation(object):
    def __init__(self, dmd, padding_x=0, padding_y=0):

        if not isinstance(padding_x, int):
            raise TypeError("padding_x must be an integer.")

        if not isinstance(padding_y, int):
            raise TypeError("padding_x must be an integer.")

        if isinstance(dmd, DMD):
            self.dmd = dmd
        else:
            raise TypeError("dmd must be a DMD object.")

        self._padding_x = padding_x
        self._padding_y = padding_y


    @property
    def padding_x(self):
        return self._padding_x

    @property
    def padding_y(self):
        return self._padding_y

    @lru_cache()
    def _image_plane_padded_grid(self):
        kx_atom, ky_atom = tf.constant(np.fft.fftfreq(self.dmd.Nx + 2 * self._padding_x, self.dmd.micromirror_size), dtype=BACKEND.dtype), \
                           tf.constant(-np.fft.fftfreq(self.dmd.Ny + 2 * self._padding_y, self.dmd.micromirror_size), dtype=BACKEND.dtype)

        kx_atom, ky_atom = tf.signal.fftshift(kx_atom), tf.signal.fftshift(ky_atom)
        x_atom = kx_atom * self.dmd.scaling_factor
        y_atom = ky_atom * self.dmd.scaling_factor
        return tf.meshgrid(x_atom, y_atom)

    @property
    def image_plane_padded_grid(self):
        x, y = self._image_plane_padded_grid()
        return np.array(x), np.array(y)

    @lru_cache()
    def _fourier_plane_padded_grid(self):
        pix_j = tf.range(-self._padding_x, self.dmd.Nx+self._padding_x, dtype=BACKEND.dtype)
        pix_i = tf.range(-self._padding_y, self.dmd.Ny+self._padding_y, dtype=BACKEND.dtype)

        pix_ii, pix_jj = tf.meshgrid(pix_i, pix_j, indexing="ij")

        return self.dmd._convert_pixel_index_to_dmd_coordinate(pix_ii, pix_jj)

    @property
    def fourier_plane_padded_grid(self):
        x, y = self._fourier_plane_padded_grid()
        return np.array(x), np.array(y)
