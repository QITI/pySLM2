from functools import lru_cache
import math

import numpy as np
import tensorflow as tf

from . import _lib
from ._backend import BACKEND
from .slm import SLM, DMD

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
        return self._slm.scaling_factor ** 2 / (self.Nx * self.Ny) / self._slm.pixel_size ** 2

    @property
    def padding_x(self):
        return self._padding_x

    @property
    def padding_y(self):
        return self._padding_y

    @property
    def Nx(self):
        return self._slm.Nx + 2 * self._padding_x

    @property
    def Ny(self):
        return self._slm.Ny + 2 * self._padding_y

    @lru_cache()
    def _image_plane_padded_grid(self):
        kx_atom, ky_atom = tf.constant(np.fft.fftfreq(self.Nx, self._slm.pixel_size),
                                       dtype=BACKEND.dtype), \
                           tf.constant(-np.fft.fftfreq(self.Ny, self._slm.pixel_size),
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

    def propagate_to_image(self, input_profile):
        # TODO use tf function to speed up
        self._input_field = tf.zeros(shape=(self.Ny, self.Nx), dtype=BACKEND.dtype_complex)

        input_profile = self._slm._profile_to_tensor(input_profile)

        self._input_field[self._padding_y:-self._padding_y,
                          self._padding_x:-self._padding_x] = input_profile

        self._output_field[self._padding_y:-self._padding_y,
                           self._padding_x:-self._padding_x] = input_profile * self._slm._state_tensor()

        _image_plane_field_unormalized = tf.signal.fftshift(
            _lib._inverse_fourier_transform(tf.signal.ifftshift(self._output_field)))

        self._image_plane_field = _image_plane_field_unormalized * math.sqrt(
            self.Nx * self.Ny) * self.fourier_plane_pixel_area / self.image_plane_pixel_area

    @tf.function
    def _field_to_intensity(self, field_tensor):
        return tf.math.real(field_tensor)**2 + tf.math.imag(field_tensor)**2

    @property
    def _input_intensity(self):
        return None if self._input_field is None else self._field_to_intensity(self._input_field)

    @property
    def _output_intensity(self):
        return None if self._output_field is None else self._field_to_intensity(self._output_field)

    @property
    def _image_plane_intensity(self):
        return None if self._image_plane_field is None else self._field_to_intensity(self._image_plane_field)

    def _pack_tensor_to_array(self, tensor):
        return None if tensor is None else np.array(tensor)

    @property
    def input_field(self):
        return self._pack_tensor_to_array(self._input_field)

    @property
    def output_field(self):
        return self._pack_tensor_to_array(self._output_field)

    @property
    def image_plane_field(self):
        return self._pack_tensor_to_array(self._image_plane_field)

    @property
    def input_intensity(self):
        return self._pack_tensor_to_array(self._input_intensity)

    @property
    def output_intensity(self):
        return self._pack_tensor_to_array(self._output_intensity)

    @property
    def image_plane_intensity(self):
        return self._pack_tensor_to_array(self._image_plane_intensity)



class DMDSimulation(SLMSimulation):
    def __init__(self, dmd, **kwargs):
        if not isinstance(dmd, DMD):
            raise TypeError("dmd must be a DMD object.")

        super(DMDSimulation, self).__init__(dmd, **kwargs)
