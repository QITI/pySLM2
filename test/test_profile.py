import numpy as np
import pytest
import pySLM2
import matplotlib.pyplot as plt


@pytest.mark.parametrize("extrapolate", [False, True])
def test_zernike_function(extrapolate, visualize=False):
    """Test if the Zernike profile match the Zernike polynomials from the reference papser at different orderes.
    References
    ----------
    Thibos, L. N., Applegate, R. A., Schwiegerling, J. T., & Webb, R. (2002). Standards for reporting the optical aberrations of eyes. Journal of refractive surgery, 18(5), S652-S660.
    """

    # Zernike polynomials in table 2.
    zernike_function = {
        (0, 0): lambda rho, phi: np.ones_like(rho),
        (1, -1): lambda rho, phi: 2 * rho * np.sin(phi),
        (1, 1): lambda rho, phi: 2 * rho * np.cos(phi),
        (2, -2): lambda rho, phi: np.sqrt(6) * rho ** 2 * np.sin(2 * phi),
        (2, 0): lambda rho, phi: np.sqrt(3) * (2 * rho ** 2 - 1),
        (2, 2): lambda rho, phi: np.sqrt(6) * rho ** 2 * np.cos(2 * phi),
        (3, -3): lambda rho, phi: np.sqrt(8) * rho ** 3 * np.sin(3 * phi),
        (3, -1): lambda rho, phi: np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.sin(phi),
        (3, 1): lambda rho, phi: np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.cos(phi),
        (3, 3): lambda rho, phi: np.sqrt(8) * rho ** 3 * np.cos(3 * phi)
    }

    radius = 10

    x, y = np.meshgrid(np.linspace(-radius, radius), np.linspace(-radius, radius))

    for (n, m), function in zernike_function.items():
        z = pySLM2.Zernike(a=1, radius=radius, n=n, m=m, extrapolate=extrapolate)

        # Defination of rho and phi at page 656
        rho = np.sqrt(x ** 2 + y ** 2) / radius
        phi = np.arctan2(y, x)
        mask = (rho <= 1)

        ANSI_zernike = (function(rho, phi))
        pySLM2_zernike = (z(x, y))

        if visualize:
            plt.figure()
            plt.subplot(121)
            plt.imshow(ANSI_zernike)
            plt.subplot(122)
            plt.imshow(pySLM2_zernike)
            plt.show()

        if not extrapolate:
            ANSI_zernike = ANSI_zernike[mask]
            pySLM2_zernike = pySLM2_zernike[mask]

        np.testing.assert_array_almost_equal(
            ANSI_zernike,
            pySLM2_zernike,
            decimal=4,
            err_msg="n={n}, ,m={m} failed the test!".format(n=n, m=m)
        )
