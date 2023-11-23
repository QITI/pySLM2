import pySLM2
from scipy.constants import micro, nano, milli
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

dmd = pySLM2.DLP9500(
    369 * nano,  # wavelength
    200 * milli, # effective focal length
    4, # periodicity of the grating = 4 pixels
    np.pi/4, # The dmd is rotated by 45 degrees
)

# Aberration: Spherical aberration
aberration = pySLM2.Zernike(10, 5 * milli, n=3, m=0) 

# The beam illumilating the DMD is an gaussian beam with a waist of 5 mm
input_profile_unaware_of_aberration = pySLM2.HermiteGaussian(0,0,1,5*milli)
input_profile = input_profile_unaware_of_aberration  * aberration.as_complex()


# targeted profile at the image plane
output_profile = pySLM2.HermiteGaussian(0,0,1,10*micro, n=0, m=0)

pfig, axs = plt.subplots(2, 2)

axs[0, 0].set_title("Aberration\n phase map")
p = axs[0, 0].imshow(dmd.profile_to_tensor(aberration) / np.pi / 2)
axs[0, 0].set_xlabel("x [px]")
axs[0, 0].set_ylabel("y [px]")
cbar = plt.colorbar(p, ax=axs[0, 0])
cbar.set_label("$\lambda$")


axs[1, 0].set_title("Input beam\n intensity profile")
p = axs[1, 0].imshow(np.abs(dmd.profile_to_tensor(input_profile, complex=True))**2)
axs[1, 0].set_xlabel("x [px]")
axs[1, 0].set_ylabel("y [px]")
plt.colorbar(p, ax=axs[1, 0])

sim = pySLM2.DMDSimulation(dmd, padding_x=0, padding_y=(dmd.Nx-dmd.Ny)//2)

# Calculate the hologram without the knowledge of the aberration
dmd.calculate_dmd_state(
    input_profile_unaware_of_aberration,
    output_profile,
    method="random"
)

# perform the simulation
sim.propagate_to_image(input_profile)
sim.block_zeroth_order()

def to_um(x, pos):
    return f"{x/micro: .1f} $\mu$m"

p = axs[0, 1].pcolormesh(*sim.image_plane_padded_grid, sim.image_plane_intensity)
(x,y) = dmd.first_order_origin
axs[0, 1].set_title("1st order beam intensity profile \n (aberration uncorrected)")
axs[0, 1].set_xlim(x-100*micro, x+100*micro)
axs[0, 1].set_ylim(y-100*micro, y+100*micro)
axs[0, 1].xaxis.set_major_formatter(FuncFormatter(to_um))
axs[0, 1].yaxis.set_major_formatter(FuncFormatter(to_um))
plt.colorbar(p, ax=axs[0, 1])

# Calculate the hologram including the aberration
dmd.calculate_dmd_state(
    input_profile,
    output_profile,
    method="random"
)

# perform the simulation
sim.propagate_to_image(input_profile)
sim.block_zeroth_order()

p = axs[1, 1].pcolormesh(*sim.image_plane_padded_grid, sim.image_plane_intensity)
(x,y) = dmd.first_order_origin
axs[1, 1].set_title("1st order beam intensity profile \n (aberration corrected)")
axs[1, 1].set_xlim(x-100*micro, x+100*micro)
axs[1, 1].set_ylim(y-100*micro, y+100*micro)
axs[1, 1].xaxis.set_major_formatter(FuncFormatter(to_um))
axs[1, 1].yaxis.set_major_formatter(FuncFormatter(to_um))
plt.colorbar(p, ax=axs[1, 1])

plt.tight_layout()
plt.show()
