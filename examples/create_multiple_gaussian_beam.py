import pySLM2
from scipy.constants import micro, nano, milli
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter


def to_um(x, pos):
    return f"{x/micro: .1f}"

def to_rad(x, pos):
    return f"{x/np.pi: .1f}π"

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

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[1,1], height_ratios=[1,1])

# Define subplots within the custom gridspec
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

# Plot the images in the subplots
p0 = axs[0].imshow(dmd.dmd_state)
axs[0].set_xlabel("x [px]", fontsize=10, ha='center')
axs[0].set_ylabel("y [px]", fontsize=10, ha='center')

p1 = axs[1].imshow(dmd.profile_to_tensor(input_profile ** 2))
axs[1].set_xlabel("x [px]", fontsize=10, ha='center')
axs[1].set_ylabel("y [px]", fontsize=10, ha='center')

sim = pySLM2.DMDSimulation(dmd, padding_x=0, padding_y=(dmd.Nx - dmd.Ny) // 2)

# Perform the simulation
sim.propagate_to_image(input_profile)
sim.block_zeroth_order()

# Plot intensity profile
p2 = axs[2].pcolormesh(*sim.image_plane_padded_grid, sim.image_plane_intensity)
(x, y) = dmd.first_order_origin
axs[2].set_xlim(x-100*micro, x+100*micro)
axs[2].set_ylim(y-100*micro, y+100*micro)
axs[2].xaxis.set_major_formatter(FuncFormatter(to_um))
axs[2].yaxis.set_major_formatter(FuncFormatter(to_um))
axs[2].set_xlabel("x' [µm]", fontsize=10, ha='center')
axs[2].set_ylabel("y' [µm]", fontsize=10, ha='center')

# Plot phase profile
p3 = axs[3].pcolormesh(*sim.image_plane_padded_grid, np.angle(sim.image_plane_field))
(x, y) = dmd.first_order_origin
axs[3].set_xlim(x-100*micro, x+100*micro)
axs[3].set_ylim(y-100*micro, y+100*micro)
axs[3].xaxis.set_major_formatter(FuncFormatter(to_um))
axs[3].yaxis.set_major_formatter(FuncFormatter(to_um))
axs[3].set_xlabel("x' [µm]", fontsize=10, ha='center')
axs[3].set_ylabel("y' [µm]", fontsize=10, ha='center')

# Add subcaptions below the subplots
subcaptions = ["(a) Hologram displayed on the DMD",
               "(b) Input beam profile",
               "(c) Intensity profile of the first-order beam",
               "(d) Phase profile of the first-order beam"]

for i, ax in enumerate(axs):
    ax.set_title(subcaptions[i], fontsize=12,pad=10)

# Add colorbars
cbar_label = ['Intensity [a.u.]', 'Intensity [a.u.]', 'Phase [rad]']
for i, plot in enumerate([p1, p2, p3]):
    divider = make_axes_locatable(plot.axes)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(plot, cax=cax, orientation='vertical')
    cbar.set_label(cbar_label[i])
    if 'Phase' in cbar_label[i]:
        cbar.formatter = FuncFormatter(to_rad)
        cbar.update_ticks()

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

