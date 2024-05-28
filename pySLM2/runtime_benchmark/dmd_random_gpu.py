import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from scipy.constants import micro, nano, milli
import numpy as np
import time

import pySLM2
import tensorflow as tf



def task(method, N):
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

    start = time.time()

    kwargs = dict()
    kwargs["signal_window"] = signal_window
    kwargs["N"] = N
    dmd.calculate_dmd_state(
        input_profile,
        output_profile,
        method=method,
        **kwargs
    )
    end = time.time()
    total_time = end-start
    return total_time



num_gpu  = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_gpu)
if num_gpu ==0:
    print("No GPU found. Exit.")
    exit()
else:
    print("Is Built with CUDA: ", tf.test.is_built_with_cuda())

N=1000
num_test = 10
result = []
print(f'Total {num_test} Tests Running on GPU')
for i in range(num_test):
    ti = task('random',N)
    print(f'test {i} runtime: {ti:0.02f}s')
    result.append(ti)
print(f'runtime for {num_test} runs: {np.mean(result):0.2f}s +- {np.std(result):0.2f}s')



