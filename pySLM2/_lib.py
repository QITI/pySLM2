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
        return tf.signal.ifft2d(profile_tensor)


def calculate_dmd_grating(amp, phase_in, phase_out, x, y, p, theta, method="random", negative_order=False, **kwargs):
    '''
    Calculate a DMD grating pattern. Algorithms available are "ideal", "simple", "random" and "ifta".
    
    Parameters
    ----------
    amp : float
        Amplitude of the grating pattern.
    phase_in : float
        Phase of the grating pattern at the input plane.
    phase_out : float
        Phase of the grating pattern at the output plane.
    x : tf.Tensor
        X coordinate of the plane.
    y : tf.Tensor
        Y coordinate of the plane.
    p : float
        Period of the grating pattern.
    theta : float
        Angle of the grating pattern.
    method : str
        Method to calculate the grating pattern. Options are "ideal", "simple", "random", "ifta".
    negative_order : bool
        If True, the grating pattern will be calculated for the negative order.
    **kwargs : dict
        Additional parameters for the selected method.
    '''
    negative_order = tf.constant(negative_order, dtype=tf.bool)
    if method == "ideal":
        return _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta, negative_order=negative_order)
    elif method == "simple":
        return _calculate_dmd_grating_simple(amp, phase_in, phase_out, x, y, p, theta, negative_order=negative_order)
    elif method == "random":
        r = kwargs.get("r", 10.0)
        r = tf.constant(r, dtype=BACKEND.dtype)
        return _calculate_dmd_grating_random(amp, phase_in, phase_out, x, y, p, theta, negative_order=negative_order,
                                             r=r)
    elif method == "ifta":
        input_profile = kwargs.get("input_profile")
        signal_window = kwargs.get("signal_window")
        N = kwargs.get("N", 200)
        N = tf.constant(N, dtype=BACKEND.dtype_int)
        s = kwargs.get("s", 1.0)
        s = tf.constant(s, dtype=BACKEND.dtype)
        return _calculate_dmd_grating_ifta(amp, phase_in, phase_out, x, y, p, theta, input_profile, signal_window,
                                           negative_order=negative_order, N=N, s=s)

    else:
        raise ValueError("{0} is not a valid method!".format(method))


@tf.function
def _grating_phase(x, y, theta, p):
    '''
    Calculate the phase of a grating pattern.
    
    Parameters
    ----------
    x : tf.Tensor
        X coordinate of the plane.
    y : tf.Tensor
        Y coordinate of the plane.  
    theta : float
        Angle of the grating pattern.
    p : float
        Period of the grating pattern.

    Returns
    -------
    tf.Tensor
        Phase of the grating pattern.
    '''
    return (2 * math.pi / p) * (tf.cos(theta) * x + tf.sin(theta) * y)


@tf.function
def _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta, negative_order=False):
    '''
    Calculate a grey scale DMD grating pattern.
    
    Parameters
    ----------
    amp : float
        Amplitude of the grating pattern.
    phase_in : float
        Phase of the grating pattern at the input plane.
    phase_out : float
        Phase of the grating pattern at the output plane.
    x : tf.Tensor
        X coordinate of the plane.
    y : tf.Tensor
        Y coordinate of the plane.
    p : float
        Period of the grating pattern.
    theta : float
        Angle of the grating pattern.
    negative_order : bool
        If True, the grating pattern will be calculated for the negative order.
    
    Returns
    -------
    tf.Tensor
        Grating pattern.
    '''
    grating_phase = _grating_phase(x, y, theta, p)
    if negative_order:
        phase_in = -phase_in
        phase_out = -phase_out

    grating = amp * (tf.cos(grating_phase - (phase_out - phase_in)) / 2 + 0.5)
    return grating


@tf.function
def _calculate_dmd_grating_simple(amp, phase_in, phase_out, x, y, p, theta, negative_order=False):
    '''
    Calculate a grey scale DMD grating pattern then binarize it with a threshold of 0.5.
    '''
    grating_ideal = _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta,
                                                 negative_order=negative_order)
    return grating_ideal > 0.5


@tf.function
def _calculate_dmd_grating_random(amp, phase_in, phase_out, x, y, p, theta, negative_order=False, r=1.0):
    '''
    Calculate a binarized DMD grating pattern with random thresholds for binarization."

    Parameters
    ----------
    amp : tf.Tensor
        Amplitude of the grating pattern.
    phase_in : tf.Tensor
        Phase of the grating pattern at the input plane.
    phase_out : tf.Tensor
        Phase of the grating pattern at the output plane.
    x : tf.Tensor
        X coordinate of the plane.
    y : tf.Tensor
        Y coordinate of the plane.
    p : float
        Period of the grating pattern.
    theta : float
        Angle of the grating pattern.
    negative_order : bool
        If True, the grating pattern will be calculated for the negative order.
    r : float
        Steepness of the tanh function that binarizes the grating pattern in a smooth way.
    
    Returns
    -------
    tf.Tensor
        Binarized grating pattern.
    '''
    grating_phase = _grating_phase(x, y, theta, p)
    if negative_order:
        phase_in = -phase_in
        phase_out = -phase_out  # verify if this is correct.

    p = tf.acos(tf.cos(grating_phase - (phase_out - phase_in)))
    # TODO more efficient expression for phase extraction?

    w = tf.math.asin(amp)  # + pi / 2

    grating = 0.5*(tf.math.tanh(r * (p + w)) - tf.math.tanh(r * (p - w)))
    threshold = tf.random.uniform(shape=grating.shape, dtype=BACKEND.dtype)
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
                                negative_order=False, N=200, s=tf.constant(1.0)):
    '''
    Inverse Fourier Transform Algorithm (IFTA) for the calculation of the DMD grating pattern.

    Parameters
    ----------
    amp : tf.Tensor
        Amplitude of the grating pattern.
    phase_in : tf.Tensor
        Phase of the grating pattern at the input plane.
    phase_out : tf.Tensor
        Phase of the grating pattern at the output plane.
    x : tf.Tensor
        X coordinate of the plane.
    y : tf.Tensor
        Y coordinate of the plane.
    p : float
        Period of the grating pattern.
    theta : float
        Angle of the grating pattern.
    input_profile : tf.Tensor
        The input profile of the beam at Fourier plane.
    signal_window : tf.Tensor
        Signal window to be applied to the input profile.
    negative_order : bool
        If True, the grating pattern will be calculated for the negative order.
    N : int
        Number of iterations.
    s : float
        Step size.

    Returns
    -------
    tf.Tensor
        Binarized grating pattern.
    '''
    grating_ideal = s * _calculate_dmd_grating_ideal(amp, phase_in, phase_out, x, y, p, theta,
                                                    negative_order=negative_order)

    grating_unbinarized = tf.identity(grating_ideal)

    profile_ideal = _inverse_fourier_transform(input_profile * tf.cast(grating_ideal, BACKEND.dtype_complex))
    signal_window = tf.signal.ifftshift(signal_window)
    step = 0.5 / tf.cast(N, dtype=BACKEND.dtype)

    mask = input_profile == 0.0
    zeros = tf.zeros_like(input_profile, dtype=BACKEND.dtype)

    for i in tf.range(N):
        i = tf.cast(i, dtype=BACKEND.dtype)
        grating_binarized = _ifta_binarize_hologram(grating_unbinarized, (i+1) * step)
        modulated_profile = input_profile*tf.cast(grating_binarized, BACKEND.dtype_complex)
        image_plane_profile = _inverse_fourier_transform(modulated_profile)
        profile_corrected = _ifta_correct_profile(image_plane_profile, profile_ideal, signal_window)
        grating_unbinarized = tf.where(
            mask,
            zeros,
            tf.math.real(_fourier_transform(profile_corrected) / input_profile)
        )

    return grating_unbinarized > 0.5


def calculate_lcos_slm_hologram(input_profile, target_amp_profile, method="gs", **kwargs):
    '''
    Calculate the hologram phase profile for an LCOS SLM. Algorithms available are "gs" and "mraf".
    
    Parameters
    ----------
    input_profile : tf.Tensor
        The input profile of the beam at Fourier plane.
    target_amp_profile : tf.Tensor
        The target amplitude profile of the beam at image plane.
    method : str
        Method to calculate the hologram phase profile. Options are "gs" and "mraf".
    **kwargs : dict
        Additional parameters for the selected method.

    Returns
    -------
    tf.Tensor
        Hologram phase profile.
    '''
    if method=="gs":
        N = kwargs.get("N", 200)
        N = tf.constant(N, dtype=BACKEND.dtype_int)
        return _calculate_lcos_slm_hologram_gs(input_profile, target_amp_profile, N)
    elif method=="mraf":
        N = kwargs.get("N", 200)
        N = tf.constant(N, dtype=BACKEND.dtype_int)
        signal_window = kwargs.get("signal_window")
        mixing_factor = kwargs.get("mixing_factor", 0.4)
        mixing_factor = tf.constant(mixing_factor, dtype=BACKEND.dtype)
        return _calculate_lcos_slm_hologram_mraf(input_profile, target_amp_profile, N, signal_window, mixing_factor)
    else:
        raise ValueError("{0} is not a valid method!".format(method))


@tf.function
def _calculate_lcos_slm_hologram_gs(input_profile, target_amp_profile, N):
    '''
    Gerchberg-Saxton (GS) algorithm for the calculation of the hologram phase profile.

    Parameters
    ----------
    input_profile : tf.Tensor
        The input profile of the beam at Fourier plane.
    target_amp_profile : tf.Tensor
        The target amplitude profile of the beam at image plane.
    N : int
        Number of iterations.

    Returns
    -------
    tf.Tensor
        Hologram phase profile.
    '''
    phase_profile = 2 * math.pi * tf.random.uniform(shape=input_profile.shape, dtype=BACKEND.dtype)

    target_amp_profile = tf.signal.ifftshift(target_amp_profile)
    target_amp_profile = tf.cast(target_amp_profile, dtype=BACKEND.dtype_complex)

    for _ in tf.range(N):
        phase_profile = tf.cast(phase_profile, dtype=BACKEND.dtype_complex)
        modulated_profile = input_profile * tf.exp(1j * phase_profile)
        image_plane_profile = _inverse_fourier_transform(modulated_profile)
        profile_angle = tf.math.angle(image_plane_profile)
        profile_angle = tf.cast(profile_angle, dtype=BACKEND.dtype_complex)
        profile_corrected = target_amp_profile * tf.exp(1j * profile_angle)
        phase_profile = tf.math.angle(_fourier_transform(profile_corrected))

    return phase_profile

@tf.function
def _calculate_lcos_slm_hologram_mraf(input_profile, target_amp_profile, N, signal_window, mixing_factor):
    '''
    Mixed Random Amplitude and Phase (MRAF) algorithm for the calculation of the hologram phase profile.
    
    Parameters
    ----------
    input_profile : tf.Tensor
        The input profile of the beam at Fourier plane.
    target_amp_profile : tf.Tensor
        The target amplitude profile of the beam at image plane.
    N : int
        Number of iterations.
    signal_window : tf.Tensor
        Signal window, in which the target amplitude profile is applied.
    mixing_factor : float
        Mixing factor. The value is between 0 and 1. It means the proportion of the target amplitude profile in the final hologram.
    
    
    Returns
    -------
    tf.Tensor
        Hologram phase profile.
    '''
    phase_profile = 2 * math.pi * tf.random.uniform(shape=input_profile.shape, dtype=BACKEND.dtype)

    target_amp_profile = tf.signal.ifftshift(target_amp_profile)
    target_amp_profile = tf.cast(target_amp_profile, dtype=BACKEND.dtype_complex)
    
    one_minus_m = 1 - mixing_factor

    mixing_factor = tf.cast(mixing_factor, dtype=BACKEND.dtype_complex)
    one_minus_m = tf.cast(one_minus_m, dtype=BACKEND.dtype_complex)

    signal_window = tf.signal.ifftshift(signal_window)
    inverse_signal_window = 1 - signal_window

    for _ in tf.range(N):
        phase_profile = tf.cast(phase_profile, dtype=BACKEND.dtype_complex)
        modulated_profile = input_profile * tf.exp(1j * phase_profile)
        image_plane_profile = _inverse_fourier_transform(modulated_profile)
        profile_angle = tf.math.angle(image_plane_profile)
        profile_angle = tf.cast(profile_angle, dtype=BACKEND.dtype_complex)
        profile_corrected = mixing_factor * target_amp_profile * tf.exp(1j * profile_angle) * signal_window + \
                            one_minus_m * image_plane_profile * inverse_signal_window
        phase_profile = tf.math.angle(_fourier_transform(profile_corrected))

    return phase_profile
