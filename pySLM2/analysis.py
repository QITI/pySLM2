from .profile import Zernike
import numpy as np


def remove_trivial_phase(profile, radius, remove_defocus=False, verbose=False):
    """Remove the trivial phase map from the profile within the aperture.

    The trivial phase maps include piston and tilt by default.

    Parameters
    ----------
    profile: numpy.ndarray
        The phase map profile
    radius: int or float
        The aperture radiusa
    remove_defocus: bool
        If true, remove the defocus from the profile.
    verbose: bool
        If true, print the coefficient of the removed phase map.

    Returns
    -------
    profile_nontrivial: numpy.ndarray
        Phase map after removing trivial

    """
    Ny, Nx = profile.shape
    xx, yy = np.meshgrid(np.arange(Nx) - Nx // 2, np.arange(Ny, 0, -1) - Ny // 2)
    trivial_terms = [(0, 0), (1, -1), (1, 1)]

    if remove_defocus:
        trivial_terms.append((2, 0))

    aperture = xx ** 2 + yy ** 2 < radius ** 2

    for n, m in trivial_terms:
        z = Zernike(1, radius, n, m, extrapolate=False)(xx, yy)
        c = np.mean(profile[aperture] * z[aperture])
        if verbose:
            print("n = {0}, m = {1}, coefficient = {2}".format(n, m, c))
        profile = profile - c * z
    return profile


def ansi_j_to_n_m(idx):
    """Convert ANSI single term to (n,m) two-term index.

    See Also
    --------
    The MIT License (MIT)
    Copyright (c) 2017-2020 Brandon Dube
    https://github.com/brandondube/prysm
    """
    n = int(np.ceil((-3 + np.sqrt(9 + 8*idx))/2))
    m = 2 * idx - n * (n + 2)
    return n, m


def zernike_decomposition(profile, radius, num_terms, verbose=False):
    """Extract Zernike coefficients of a phase profile.

    Parameters
    ----------
    profile: numpy.ndarray
        The phase map profile
    radius: int or float
        The aperture radiusa
    num_terms: int
        Number of ANSI Zernike coefficients
    verbose: bool
        If true, print the coefficient of the removed phase map.

    Returns
    -------
    coefficients: list of floats
        Zernike coefficients. The length is the list is the same as num_terms.
    """
    Ny, Nx = profile.shape
    xx, yy = np.meshgrid(np.arange(Nx) - Nx // 2, np.arange(Ny, 0, -1) - Ny // 2)

    aperture = xx ** 2 + yy ** 2 < radius ** 2

    coefficients = []
    for idx in range(num_terms):
        n, m = ansi_j_to_n_m(idx)
        z = Zernike(1, radius, n, m, extrapolate=False)(xx, yy)
        c = np.mean(profile[aperture] * z[aperture])
        if verbose:
            print("n = {0}, m = {1}, coefficient = {2}".format(n, m, c))
        coefficients.append(c)

    return coefficients


