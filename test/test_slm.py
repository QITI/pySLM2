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


@pytest.mark.parametrize("dmd_model,f", [(pySLM2.DLP9500, 37 * milli),
                                         (pySLM2.DLP7000, 137 * milli)])
def test_eta(dmd_model, f):
    dmd = dmd_model(wavelength=369 * nano, focal_length=f, periodicity=4, theta=-np.pi / 4)
    sim = pySLM2.DMDSimulation(dmd)

    input_profile = pySLM2.HermiteGaussian(0, 0, 100, 3 * milli)
    # target_profile = pySLM2.HermiteGaussian(0, 0, 1, 10 * micro, n=1, m=1)

    dmd.set_dmd_state_on()

    sim.propagate_to_image(input_profile)
    target_profile = sim.image_plane_field

    eta = dmd.calculate_dmd_state(input_profile, target_profile, method="ideal")
    assert eta == pytest.approx(1, 1e-3)  # perfect mode matching

    target_profile = target_profile / 2
    eta = dmd.calculate_dmd_state(input_profile, target_profile, method="ideal")
    assert eta == pytest.approx(2, 1e-3)


@pytest.mark.parametrize("method,kwargs", [("ideal", dict()),
                                           ("random", dict(r=1)),
                                           ("ifta", dict(N=100))])
def test_dmd_hologram_calc(method, kwargs):
    dmd = pySLM2.DLP9500(wavelength=369 * nano, focal_length=37 * milli, periodicity=4, theta=-np.pi / 4)
    sim = pySLM2.DMDSimulation(dmd)

    x0, y0 = dmd.first_order_origin
    signal_window = pySLM2.RectWindow(x0, y0, 100 * micro, 100 * micro)

    if method == "ifta":
        kwargs["signal_window"] = signal_window

    input_profile = pySLM2.HermiteGaussian(0, 0, 200, 3 * milli)
    target_profile = pySLM2.HermiteGaussian(0, 0, 1, 10 * micro, n=1, m=1)

    eta = dmd.calculate_dmd_state(input_profile, target_profile, method=method, **kwargs)
    # print("eta:", eta)
    # print("scaling_factor", sim.scaling_factor)

    target_profile_first_order = target_profile.shift(x0, y0)(*sim.image_plane_padded_grid) * eta / 4

    window = signal_window(*sim.image_plane_padded_grid) > 0.5

    sim.propagate_to_image(input_profile)
    sim.block_zeroth_order()

    mx = np.max(target_profile_first_order)

    def rms(a, b):
        diff = a - b
        diff2 = np.real(diff) ** 2 + np.imag(diff) ** 2
        return np.sqrt(np.mean(diff2))

    rel_err = rms(target_profile_first_order[window], sim.image_plane_field[window]) / mx

    assert rel_err < 0.05

    # plt.imshow(np.abs(target_profile_first_order))
    # plt.colorbar()
    # plt.show()
    # plt.imshow(np.abs(sim.image_plane_field)*window)
    # plt.colorbar()
    # plt.show()
