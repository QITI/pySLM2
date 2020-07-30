import tensorflow as tf
import numpy as np
import math
from ._backend import DTYPE

pi = tf.constant(math.pi)


def calculate_dmd_grating(amp, phase_in, phase_out, x, y, p, theta, method="random", negative_order=False, **kwargs):
    if method == "ideal":
        return _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta, negative_order=negative_order)
    elif method == "random":
        r = kwargs.get("r", 1)
        return _calculate_dmd_grating_random(amp, phase_in, phase_out, x, y, p, theta, negative_order=negative_order,
                                             r=r)
    else:
        raise ValueError("{0} is not a valid method!".format(method))


@tf.function
def _grating_phase(x, y, theta, p):
    return (2 * pi / p) * (tf.cos(theta) * x + tf.sin(theta) * y)


@tf.function
def _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta, negative_order=False):
    grating_phase = _grating_phase(x, y, theta, p)
    if negative_order:
        phase_in = -phase_in
    grating = amp * (tf.cos(grating_phase - (phase_out - phase_in)) / 2 + 0.5)
    return grating


@tf.function
def _calculate_dmd_grating_random(amp, phase_in, phase_out, x, y, p, theta, negative_order=False, r=1):
    grating_phase = _grating_phase(x, y, theta, p)
    if negative_order is True:
        phase_in = -phase_in

    p = tf.acos(tf.cos(grating_phase - (phase_out - phase_in)))
    # TODO more efficient expression for phase extraction?

    w = tf.math.asin(amp) + pi / 2
    patch_state = (tf.math.tanh(r * (p + w / 2)) - tf.math.tanh(r * (p - w / 2)))
    threshold = tf.random.uniform(shape=patch_state.shape)
    patch_state = (patch_state > threshold)

    return patch_state


@tf.function
def _ifta_correct_profile(profile, profile_ideal, signal_window):
    # return tf.where(signal_window, profile_ideal, profile)
    return (profile_ideal - profile) * signal_window + profile


@tf.function
def _ifta_binarize_hologram(hologram, threthold):
    def _ifta_binarize_hologram_element(hologram_element):
        if hologram_element > (1 - threthold):
            return 1.0
        elif hologram_element < threthold:
            return 0.0
        else:
            return hologram_element

    return tf.map_fn(_ifta_binarize_hologram_element, hologram)


@tf.function
def _calculate_dmd_grating_ifta(amp, phase_in, phase_out, x, y, p, theta, input_profile, signal_window,
                                negative_order=False, N=200):
    grating_ideal = _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta,
                                                 negative_order=negative_order)
    profile_ideal = tf.signal.fft2d(input_profile * grating_ideal)
    grating = tf.identity(grating_ideal)
    for i in range(N):
        pass
