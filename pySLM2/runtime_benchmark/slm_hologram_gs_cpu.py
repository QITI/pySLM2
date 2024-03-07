import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from scipy.constants import micro, nano, milli
import numpy as np
import time

import pySLM2
import tensorflow as tf


def task(method):
    lcos_slm = pySLM2.PLUTO_2(
    369 * nano,  # wavelength
    200 * milli, # effective focal length
  )

    # The beam illumilating the DMD is an gaussian beam with a waist of 5 mm
    input_profile = pySLM2.HermiteGaussian(0,0,1,5*milli)

    # targeted profile at the image plane
    output_profile = pySLM2.HermiteGaussian(0,0,1,30*micro, n=1, m=1)

    start = time.time()
    lcos_slm.calculate_hologram(
        input_profile,
        output_profile,
        method=method,
        N=200,
    )
    end = time.time()
    print("time used:", end-start)



num_gpu  = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_gpu)
if num_gpu ==0:
    print("No GPU found. Running on CPU.")
    
else:
    print("Found GPU. Exit.")
    exit()

# Re-enable GPU for the second run (if available)
task('gs')



