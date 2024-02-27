import os
import time
import pySLM2
import matplotlib.pyplot as plt
from scipy.constants import micro, nano, milli
import tensorflow as tf
import numpy as np

def task(method):
  dmd = pySLM2.DLP9500(
    369 * nano,  # wavelength
    200 * milli, # effective focal length
    4, # periodicity of the grating = 4 pixels
    np.pi/4, # The dmd is rotated by 45 degrees
  )
  x0, y0 = dmd.first_order_origin
  # The beam illumilating the DMD is an gaussian beam with a waist of 5 mm
  input_profile = pySLM2.HermiteGaussian(0,0,1,5*milli)

  # targeted profile at the image plane
  # Here we create two gaussian spots separated by 30 microns
  output_profile = pySLM2.HermiteGaussian(0,0,1,10*micro, n=0, m=0) - pySLM2.HermiteGaussian(30*micro,0,1,10*micro, n=0, m=0)
  signal_window = pySLM2.RectangularWindowRectangle(x0, y0, 100 * micro, 100 * micro)

  start_time = time.time()

  kwargs = dict()
  kwargs["signal_window"] = signal_window
  kwargs["N"] = 2000
  dmd.calculate_dmd_state(
      input_profile,
      output_profile,
      method=method,
      **kwargs
  )

# Disable GPU for the first run
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Run the task on CPU
print("Running on CPU")
start_time = time.time()
task('ifta')
cpu_time = time.time() - start_time
print(f"Time taken on CPU: {cpu_time} seconds.")

# Re-enable GPU for the second run (if available)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

num_gpu = len(tf.config.list_physical_devices('GPU'))
if num_gpu == 0:
  print("No GPU found. Exiting.")
  exit()
else:
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is Built with CUDA: ", tf.test.is_built_with_cuda())
    print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
    # Run the task on GPU
    start_time = time.time()
    task('ifta')
    gpu_time = time.time() - start_time
    print(f"Time taken on GPU: {gpu_time} seconds.")
