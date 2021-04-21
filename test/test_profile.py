import numpy as np
import pytest
import pySLM2

try:
    import matplotlib.pyplot as plt
except:
    pass

radius = 10
x, y = np.meshgrid(np.linspace(-radius, radius), np.linspace(-radius, radius))

__test_cases_arithmetic_profile = [(pySLM2.HermiteGaussian(x0=-1, y0=1, a=0.5, w=2, n=3, m=4),
                                    pySLM2.HermiteGaussian(x0=-1, y0=1, a=1, w=2, n=3, m=4)),
                                   (pySLM2.Zernike(a=1, radius=radius, n=5, m=-3, extrapolate=True),
                                    pySLM2.Zernike(a=2, radius=radius, n=5, m=-3, extrapolate=True))]


@pytest.mark.parametrize("profile, profile2", __test_cases_arithmetic_profile)
def test_profile_add(profile, profile2):
    np.testing.assert_array_almost_equal((profile + profile)(x, y), profile2(x, y))
    np.testing.assert_array_almost_equal((profile + profile)(x, y), 2 * (profile(x, y)))
    np.testing.assert_array_almost_equal((profile + 1)(x, y), profile(x, y) + 1)


@pytest.mark.parametrize("profile, profile2", __test_cases_arithmetic_profile)
def test_profile_sub(profile, profile2):
    np.testing.assert_array_almost_equal((profile - profile)(x, y), np.zeros_like(x))
    np.testing.assert_array_almost_equal((profile2 - profile)(x, y), profile(x, y))


@pytest.mark.parametrize("profile, profile2", __test_cases_arithmetic_profile)
def test_profile_multiply(profile, profile2):
    # Test multiply
    np.testing.assert_array_almost_equal((2 * profile)(x, y), profile2(x, y))
    np.testing.assert_array_almost_equal((profile * 2)(x, y), profile2(x, y))
    np.testing.assert_array_almost_equal((profile * profile)(x, y), (profile(x, y)) ** 2)


@pytest.mark.parametrize("profile, profile2", __test_cases_arithmetic_profile)
def test_profile_divide(profile, profile2):
    # Test divide
    np.testing.assert_array_almost_equal((profile2 / 2)(x, y), profile(x, y))
    np.testing.assert_array_almost_equal((profile / profile)(x, y), np.ones_like(x))


@pytest.mark.parametrize("profile, profile2", __test_cases_arithmetic_profile)
def test_profile_power(profile, profile2):
    np.testing.assert_array_almost_equal((profile ** 2)(x, y), (profile(x, y)) ** 2)


@pytest.mark.parametrize("profile, profile2", __test_cases_arithmetic_profile)
def test_neg(profile, profile2):
    np.testing.assert_array_almost_equal((profile(x, y) + (-profile)(x, y)), np.zeros_like(x))
    np.testing.assert_array_almost_equal((profile + (-profile))(x, y), np.zeros_like(x))


def test_as_complex():
    phase = pySLM2.Zernike(a=1, radius=radius, n=5, m=-3, extrapolate=True)
    np.testing.assert_array_almost_equal(phase.as_complex()(x, y), np.exp(1j * phase(x, y)))
    np.testing.assert_array_almost_equal(
        (phase.as_complex() * (-phase).as_complex())(x, y),
        np.ones_like(x)
    )


def test_laguerre_gaussian():
    gaussian_hermite_00 = pySLM2.HermiteGaussian(x0=-1, y0=1, a=0.5, w=2, n=0, m=0)
    gaussian_laguerre_00 = pySLM2.LaguerreGaussian(x0=-1, y0=1, a=0.5, w=2, l=0, p=0)
    np.testing.assert_array_almost_equal(gaussian_hermite_00(x, y), gaussian_laguerre_00(x, y))


def test_super_gaussian():
    gaussian00 = pySLM2.HermiteGaussian(x0=-1, y0=1, a=0.5, w=2, n=0, m=0)
    gaussian_super = pySLM2.SuperGaussian(x0=-1, y0=1, a=0.5, w=2, p=1)
    np.testing.assert_array_almost_equal(gaussian00(x, y), gaussian_super(x, y))


@pytest.mark.parametrize("extrapolate", [False, True])
def test_zernike_function(extrapolate, visualize=False):
    """Test if the Zernike profile match the Zernike polynomials from the reference paper at different orders.
    References
    ----------
    Thibos, L. N., Applegate, R. A., Schwiegerling, J. T., & Webb, R. (2002). Standards for reporting the optical
    aberrations of eyes. Journal of refractive surgery, 18(5), S652-S660.
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

    # Definition of rho and phi is at page 656
    rho = np.sqrt(x ** 2 + y ** 2) / radius
    phi = np.arctan2(y, x)
    mask = (rho <= 1)

    for (n, m), function in zernike_function.items():
        z = pySLM2.Zernike(a=1, radius=radius, n=n, m=m, extrapolate=extrapolate)


        OSA_zernike = (function(rho, phi))
        pySLM2_zernike = (z(x, y))

        if visualize:
            plt.figure()
            plt.subplot(121)
            plt.imshow(OSA_zernike)
            plt.subplot(122)
            plt.imshow(pySLM2_zernike)
            plt.show()

        if not extrapolate:
            OSA_zernike = OSA_zernike[mask]
            pySLM2_zernike = pySLM2_zernike[mask]

        np.testing.assert_array_almost_equal(
            OSA_zernike,
            pySLM2_zernike,
            decimal=5,
            err_msg="n={n}, ,m={m} failed the test!".format(n=n, m=m)
        )
