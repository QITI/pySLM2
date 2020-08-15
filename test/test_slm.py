import pySLM2
import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.constants import nano, micro, milli


@pytest.mark.parametrize("w_f", [1 * micro, 2 * micro])
def test_scaling_factor(w_f):
    wavelength = 369 * nano
    f = 25 * milli

    # assume the beam width remains the same from Fourier plane to the lens since the w_i is big.
    w_i = w_f * np.sqrt(1 + (f / (np.pi * w_f ** 2 / wavelength)) ** 2)
    a_i = 1.1
    a_f = w_i / w_f * a_i
    dmd = pySLM2.DLP7000(wavelength, f, 10, np.pi / 4)
    sim1 = pySLM2.DMDSimulation(dmd, padding_x=100, padding_y=1000)

    input_profile = pySLM2.HermiteGaussian(0, 0, a_i, w_i)

    target_profile = pySLM2.HermiteGaussian(0, 0, a_f, w_f)

    dmd.set_dmd_state_on()

    sim1.propagate_to_image(input_profile)

    target_profile = target_profile(*sim1.image_plane_padded_grid)
    image_plane_profile = sim1.image_plane_field

    # Error less than 1% of the peak amplitude
    np.testing.assert_array_almost_equal(image_plane_profile, target_profile, decimal=-(np.log10(a_f) - 2))
    # plt.imshow(image_plane_profile)
    # plt.show()

