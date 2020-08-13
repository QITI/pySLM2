import pySLM2
import numpy as np
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

    assert sim1.fourier_plane_pixel_area == sim2.fourier_plane_pixel_area
    assert sim1.image_plane_pixel_area == 4 * sim2.image_plane_pixel_area