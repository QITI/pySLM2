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

# The beam illumilating the DMD is an gaussian beam with a waist of 5 mm
input_profile = pySLM2.HermiteGaussian(0,0,1,5*milli)

# targeted profile at the image plane
# Here we create two gaussian spots separated by 30 microns
output_profile = pySLM2.HermiteGaussian(0,0,1,10*micro, n=0, m=0) + pySLM2.HermiteGaussian(30*micro,0,1,10*micro, n=0, m=0)

dmd.calculate_dmd_state(
    input_profile,
    output_profile,
    method="random"
)

pfig, axs = plt.subplots(2, 2)

axs[0, 0].set_title("Hologram displayed on the DMD")
axs[0, 0].imshow(dmd.dmd_state)
axs[0, 0].set_xlabel("x [px]")
axs[0, 0].set_ylabel("y [px]")

axs[1, 0].set_title("Input beam profile")
axs[1, 0].imshow(dmd.profile_to_tensor(input_profile ** 2))
axs[1, 0].set_xlabel("x [px]")
axs[1, 0].set_ylabel("y [px]")

sim = pySLM2.DMDSimulation(dmd, padding_x=0, padding_y=(dmd.Nx-dmd.Ny)//2)

# perform the simulation
sim.propagate_to_image(input_profile)
sim.block_zeroth_order()

def to_um(x, pos):
    return f"{x/micro: .1f} $\mu$m"

p = axs[0, 1].pcolormesh(*sim.image_plane_padded_grid, sim.image_plane_intensity)
(x,y) = dmd.first_order_origin
axs[0, 1].set_title("Intensity profile of the first order beam")
axs[0, 1].set_xlim(x-100*micro, x+100*micro)
axs[0, 1].set_ylim(y-100*micro, y+100*micro)
axs[0, 1].xaxis.set_major_formatter(FuncFormatter(to_um))
axs[0, 1].yaxis.set_major_formatter(FuncFormatter(to_um))
plt.colorbar(p, ax=axs[0, 1])

p = axs[1, 1].pcolormesh(*sim.image_plane_padded_grid, np.angle(sim.image_plane_field))
(x,y) = dmd.first_order_origin
axs[1, 1].set_title("Phase profile of the first order beam")
plt.colorbar(p, ax=axs[1, 1])
axs[1, 1].set_xlim(x-100*micro, x+100*micro)
axs[1, 1].set_ylim(y-100*micro, y+100*micro)
axs[1, 1].xaxis.set_major_formatter(FuncFormatter(to_um))
axs[1, 1].yaxis.set_major_formatter(FuncFormatter(to_um))
plt.tight_layout()
plt.show()