import numpy as np
import pytest
import pySLM2


def test_remove_trivial_phase_piston():
    radius = 500
    xx, yy = np.meshgrid(np.arange(1920) - 1920 // 2, np.arange(1080, 0, -1) - 1080 // 2)

    profile = pySLM2.Zernike(1,radius,0,0)(xx,yy)
    profile_removed = pySLM2.remove_trivial_phase(profile, radius=radius)
    np.testing.assert_array_almost_equal(profile_removed, np.zeros_like(profile_removed))

    profile = pySLM2.Zernike(1,radius,1,1)(xx,yy) + pySLM2.Zernike(1,radius,0,0)(xx,yy)
    profile_removed = pySLM2.remove_trivial_phase(profile, radius=radius)
    np.testing.assert_array_almost_equal(profile_removed, np.zeros_like(profile_removed), decimal=3)
    # Can we improve the precision?

    profile = pySLM2.Zernike(1, radius, 2, 0)(xx, yy)
    profile_removed = pySLM2.remove_trivial_phase(profile, radius=radius)
    np.testing.assert_array_almost_equal(profile_removed, profile, decimal=3)

    profile = pySLM2.Zernike(1, radius, 2, 0)(xx, yy)
    profile_removed = pySLM2.remove_trivial_phase(profile, radius=radius, remove_defocus=True)
    np.testing.assert_array_almost_equal(profile_removed, np.zeros_like(profile_removed), decimal=3)




