import pySLM2
from scipy.constants import micro, nano, milli
import numpy as np
import matplotlib.pyplot as plt

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

plt.title("Hologram displayed on the DMD")
plt.imshow(dmd.dmd_state)
plt.show()


sim = pySLM2.DMDSimulation(dmd, padding_x=0, padding_y=(dmd.Nx-dmd.Ny)//2)

# perform the simulation
sim.propagate_to_image(input_profile)

plt.pcolormesh(*sim.image_plane_padded_grid, sim.image_plane_intensity)
(x,y) = dmd.first_order_origin

plt.xlim(x-100*micro, x+100*micro)
plt.ylim(y-100*micro, y+100*micro)
plt.show()
