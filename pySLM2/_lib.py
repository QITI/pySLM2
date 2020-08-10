import tensorflow as tf
import numpy as np
import math
from ._backend import BACKEND


def _numpy_fft2(profile_array):
    transformed_array = np.fft.fft2(profile_array)
    if BACKEND.dtype_complex == tf.complex64:
        transformed_array = transformed_array.astype(np.complex64)
    return transformed_array


def _numpy_ifft2(profile_array):
    transformed_array = np.fft.ifft2(profile_array)
    if BACKEND.dtype_complex == tf.complex64:
        transformed_array = transformed_array.astype(np.complex64)
    return transformed_array


@tf.function
def _fourier_transform(profile_tensor):
    if BACKEND._fft_backend == BACKEND.FFT_BACKEND_NUMPY:
        transformed_tensor = tf.numpy_function(_numpy_fft2, [profile_tensor], BACKEND.dtype_complex)
        transformed_tensor = tf.reshape(transformed_tensor, profile_tensor.shape)
        return transformed_tensor
    else:
        return tf.signal.fft2d(profile_tensor)


@tf.function
def _inverse_fourier_transform(profile_tensor):
    if BACKEND._fft_backend == BACKEND.FFT_BACKEND_NUMPY:
        transformed_tensor = tf.numpy_function(_numpy_ifft2, [profile_tensor], BACKEND.dtype_complex)
        transformed_tensor = tf.reshape(transformed_tensor, profile_tensor.shape)
        return transformed_tensor
    else:
        return tf.signal.fft2d(profile_tensor)


def calculate_dmd_grating(amp, phase_in, phase_out, x, y, p, theta, method="random", negative_order=False, **kwargs):
    if method == "ideal":
        return _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta, negative_order=negative_order)
    elif method == "random":
        r = kwargs.get("r", 1)
        return _calculate_dmd_grating_random(amp, phase_in, phase_out, x, y, p, theta, negative_order=negative_order,
                                             r=r)
    elif method == "ifta":
        input_profile = kwargs.get("input_profile")
        signal_window = kwargs.get("signal_window")
        N = kwargs.get("N", 200)
        return _calculate_dmd_grating_ifta(amp, phase_in, phase_out, x, y, p, theta, input_profile, signal_window,
                                    negative_order=False, N=N)

    else:
        raise ValueError("{0} is not a valid method!".format(method))


@tf.function
def _grating_phase(x, y, theta, p):
    return (2 * math.pi / p) * (tf.cos(theta) * x + tf.sin(theta) * y)


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
    if negative_order:
        phase_in = -phase_in

    p = tf.acos(tf.cos(grating_phase - (phase_out - phase_in)))
    # TODO more efficient expression for phase extraction?

    w = tf.math.asin(amp)  # + pi / 2

    grating = (tf.math.tanh(r * (p + w / 2)) - tf.math.tanh(r * (p - w / 2)))
    threshold = tf.random.uniform(shape=grating.shape)
    grating = (grating > threshold)

    return grating


def _ifta_correct_profile(profile, profile_ideal, signal_window):
    return (profile_ideal - profile) * signal_window + profile


def _ifta_binarize_hologram(hologram, threthold):
    hologram_binarized = tf.where(hologram > (1-threthold), tf.ones_like(hologram), hologram)
    hologram_binarized = tf.where(hologram < threthold, tf.zeros_like(hologram), hologram_binarized)
    return hologram_binarized

@tf.function
def _calculate_dmd_grating_ifta(amp, phase_in, phase_out, x, y, p, theta, input_profile, signal_window,
                                negative_order=False, N=200):
    grating_ideal = _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta,
                                                 negative_order=negative_order)

    grating_unbinarized = tf.identity(grating_ideal)

    profile_ideal = _inverse_fourier_transform(input_profile * tf.cast(grating_ideal, BACKEND.dtype_complex))
    signal_window = tf.signal.ifftshift(signal_window)
    step = 0.5 / N

    for i in tf.range(N, dtype=BACKEND.dtype):
        grating_binarized = _ifta_binarize_hologram(grating_unbinarized, (i+1) * step)
        modulated_profile = input_profile*tf.cast(grating_binarized, BACKEND.dtype_complex)
        profile = _inverse_fourier_transform(modulated_profile)
        profile_corrected = _ifta_correct_profile(profile, profile_ideal, signal_window)
        grating_unbinarized = tf.math.real(_fourier_transform(profile_corrected) / input_profile)

    return grating_unbinarized > 0.5

