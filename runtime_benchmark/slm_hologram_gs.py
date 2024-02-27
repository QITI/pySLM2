import os
import pySLM2
from scipy.constants import micro, nano, milli
import numpy as np
import time
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
        method="mraf",
        N=200,
    )
    end = time.time()
    print("time used:", end-start)

# Disable GPU for the first run
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Run the task on CPU
start_time = time.time()
task('gs')
cpu_time = time.time() - start_time
print(f"Time taken on CPU: {cpu_time} seconds.")

# Re-enable GPU for the second run (if available)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("Is Built with CUDA: ", tf.test.is_built_with_cuda())
print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Run the task on GPU
start_time = time.time()
task('gs')
gpu_time = time.time() - start_time
print(f"Time taken on GPU: {gpu_time} seconds.")


