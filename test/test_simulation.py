import pySLM2
import numpy as np
import pytest
from scipy.constants import nano, micro, milli


def test_padded_grid():
    dmd = pySLM2.DLP9500(369 * nano, 37 * milli, 2, -np.pi / 4)
    sim = pySLM2.DMDSimulation(dmd=dmd, padding_x=dmd.Nx // 2, padding_y=dmd.Ny // 2)

    for i in (0, 1):
        np.testing.assert_array_equal(
            sim.fourier_plane_padded_grid[i][sim.padding_y:-sim.padding_y,
                                             sim.padding_x:-sim.padding_x],
            dmd.fourier_plane_grid[i])

    for i in (0, 1):
        np.testing.assert_array_equal(sim.image_plane_padded_grid[i][::2, ::2],
                                      dmd.image_plane_grid[i])


def test_pixel_area():
    dmd = pySLM2.DLP9500(369 * nano, 37 * milli, 2, -np.pi / 4)
    sim1 = pySLM2.DMDSimulation(dmd=dmd, padding_x=0, padding_y=0)
    sim2 = pySLM2.DMDSimulation(dmd=dmd, padding_x=dmd.Nx // 2, padding_y=dmd.Ny // 2)

    assert sim2.fourier_plane_pixel_area > 0
    assert sim2.image_plane_pixel_area > 0

    assert sim1.fourier_plane_pixel_area == sim2.fourier_plane_pixel_area
    assert sim1.image_plane_pixel_area == 4 * sim2.image_plane_pixel_area


__test_cases_slm = [pySLM2.DLP9500(369 * nano, 25 * milli, 10, np.pi / 4)]


@pytest.mark.parametrize("slm", __test_cases_slm)
def test_get_input_power(slm):
    sim = pySLM2.SLMSimulation(slm, padding_x=1000, padding_y=1000)
    a_i = 1.1
    w_i = 3 * milli
    input_profile = pySLM2.HermiteGaussian(0, 0, a_i, w_i)

    sim.propagate_to_image(input_profile)
    # test get_input_power with the analytical solution.
    # ref: https://en.wikipedia.org/wiki/Gaussian_function#Relation_to_standard_Gaussian_integral

    assert sim.get_input_power() == pytest.approx(0.5 * np.pi * a_i ** 2 * w_i ** 2, 1e-3)


def test_energy_conservation():
    dmd = pySLM2.DLP9500(369 * nano, 25 * milli, 10, np.pi / 4)
    sim1 = pySLM2.DMDSimulation(dmd, padding_x=0, padding_y=0)
    sim2 = pySLM2.DMDSimulation(dmd, padding_x=1000, padding_y=1000)

    a_i = 1.1
    w_i = 10 * milli

    input_profile = pySLM2.HermiteGaussian(0, 0, a_i, w_i)

    dmd.set_dmd_state_on()

    sim1.propagate_to_image(input_profile)
    sim2.propagate_to_image(input_profile)

    # Padding should not affect the total energy
    assert sim1.get_input_power() == pytest.approx(sim2.get_input_power())

    # test energy conservation
    assert sim2.get_input_power() == pytest.approx(sim2.get_output_power())
    assert sim2.get_input_power() == pytest.approx(sim2.get_image_plane_power())
