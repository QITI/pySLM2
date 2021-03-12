from .profile import Zernike
import numpy as np


def remove_trivial_phase(profile, radius, remove_defocus=False):
    Ny, Nx = profile.shape
    xx, yy = np.meshgrid(np.arange(Nx) - Nx // 2, np.arange(Ny, 0, -1) - Ny // 2)
    trivial_terms = [(0, 0), (1, -1), (1, 1)]

    if remove_defocus:
        trivial_terms.append((2, 0))

    aperture = xx ** 2 + yy ** 2 < radius ** 2

    for n, m in trivial_terms:
        z = Zernike(1, radius, n, m, extrapolate=False)(xx, yy)
        c = np.mean(profile[aperture] * z[aperture])
        print(n,m,c)

        profile = profile - c * z
    # profile = profile - np.mean(profile[r<radius])
    return profile
