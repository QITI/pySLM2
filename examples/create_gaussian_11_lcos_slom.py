import pySLM2
from scipy.constants import micro, nano, milli
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

lcos_slm = pySLM2.PLUTO_2(
    369 * nano,  # wavelength
    200 * milli, # effective focal length
)

# The beam illumilating the DMD is an gaussian beam with a waist of 5 mm
input_profile = pySLM2.HermiteGaussian(0,0,1,5*milli)

# targeted profile at the image plane
output_profile = pySLM2.HermiteGaussian(0,0,1,30*micro, n=1, m=1)

lcos_slm.calculate_hologram(
    input_profile,
    output_profile,
    method="gs",
    N=200
)

pfig, axs = plt.subplots(2, 2)

axs[0, 0].set_title("Hologram displayed on the DMD")
p = axs[0, 0].imshow(lcos_slm.slm_state)
plt.colorbar(p, ax=axs[0, 0])
axs[0, 0].set_xlabel("x [px]")
axs[0, 0].set_ylabel("y [px]")

axs[1, 0].set_title("Input beam profile")
axs[1, 0].imshow(lcos_slm.profile_to_tensor(input_profile ** 2))
axs[1, 0].set_xlabel("x [px]")
axs[1, 0].set_ylabel("y [px]")

sim = pySLM2.SLMSimulation(lcos_slm, padding_x=0, padding_y=(lcos_slm.Nx-lcos_slm.Ny)//2)

# perform the simulation
sim.propagate_to_image(input_profile)

def to_um(x, pos):
    return f"{x/micro: .1f} $\mu$m"


p = axs[0, 1].pcolormesh(*sim.image_plane_padded_grid, sim.image_plane_intensity)
axs[0, 1].set_title("Intensity profile of the first order beam")
axs[0, 1].set_xlim(-100*micro, 100*micro)
axs[0, 1].set_ylim(-100*micro, 100*micro)
axs[0, 1].xaxis.set_major_formatter(FuncFormatter(to_um))
axs[0, 1].yaxis.set_major_formatter(FuncFormatter(to_um))
plt.colorbar(p, ax=axs[0, 1])

p = axs[1, 1].pcolormesh(*sim.image_plane_padded_grid, np.angle(sim.image_plane_field))
axs[1, 1].set_title("Phase profile of the first order beam")
plt.colorbar(p, ax=axs[1, 1])
axs[0, 1].set_xlim(-100*micro, 100*micro)
axs[0, 1].set_ylim(-100*micro, 100*micro)
axs[1, 1].xaxis.set_major_formatter(FuncFormatter(to_um))
axs[1, 1].yaxis.set_major_formatter(FuncFormatter(to_um))
plt.tight_layout()
plt.show()
