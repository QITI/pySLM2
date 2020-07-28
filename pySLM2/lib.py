import tensorflow as tf
import numpy as np
import math


@tf.function
def _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta, negative_order=False):
    pi = tf.constant(math.pi)
    grating_phase = (2 * pi / p) * (tf.cos(theta) * x + np.sin(theta) * y)
    if negative_order:
        phase_in = -phase_in

    grating = amp * (np.cos(grating_phase - (phase_out - phase_in))/2 + 0.5)
    return grating


@tf.function
def _ifta_correct_profile(profile, profile_ideal, signal_window):
    return tf.where(signal_window, profile_ideal, profile)


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
def _calculate_dmd_grating_ifta(amp, phase_in, phase_out, x, y, p, theta, signal_window, negative_order=False, N=200):
    grating_ideal = _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta, negative_order=negative_order)



