from pySLM2 import DLP9500, HermiteGaussian, BACKEND, RectWindow
from pySLM import DLP9500 as DLP9500_old
from pySLM import RectWindow as RectWindow_old
from scipy.constants import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

#BACKEND.dtype = tf.float64
#BACKEND.fft_backend = BACKEND.FFT_BACKEND_TENSORFLOW

if __name__ == '__main__':


    dmd = DLP9500(wavelength=369 * nano, focal_length=37 * milli, periodicity=20, theta=-np.pi / 4)


    x0, y0 = dmd.first_order_origin
    signal_window = RectWindow(x0,y0,20*micro, 20*micro)

    input_profile = HermiteGaussian(0,0,100,3*milli)(*(dmd.fourier_plane_grid))
    output_profile = HermiteGaussian(0,0,1,3*micro)(*(dmd.image_plane_grid))

    t1 = time.time()
    dmd.calculate_dmd_state(input_profile, output_profile ,method="ifta", signal_window=signal_window,N=200)
    print(time.time() - t1)

    #dmd.calculate_dmd_state(input_profile,output_profile)

    plt.imshow(dmd.dmd_state[400:600,900:1100])
    plt.show()

    #exit(0)
    dmd_old = DLP9500_old(wavelength=369 * nano, focal_length=37 * milli, periodicity=20, theta=-np.pi / 4)
    x0, y0 = dmd_old.first_order_origin
    signal_window_old = RectWindow_old(x0,y0,10*micro,10*micro)

    x, y = dmd_old.image_plane_grid
    signal_window_old = signal_window_old(x, y)
    t1 = time.time()
    dmd_old.calculate_dmd_state(input_profile, output_profile,binarize=True, binarize_method="IFTA", signal_window=signal_window_old,N=200)
    print(time.time() - t1)
    plt.imshow(dmd_old.dmd_state[400:600,900:1100])
    plt.show()
    exit(0)

    for i in range(1):
        #dmd.set_dmd_grating_state(method="random")
        #dmd.circular_patch(1920//2,1080//2, 1, 0, 0, d=200)
        dmd.calculate_dmd_state(input_profile, output_profile)
    for i in range(1):
        #dmd_old.circular_patch(1920 // 2, 1080 // 2, 1, 0, 0, d=200)
        dmd_old.calculate_dmd_state(input_profile, output_profile)
        #dmd_old.set_dmd_grating_state(binarize=True)
    print(np.mean(dmd.dmd_state))
    plt.imshow(dmd.dmd_state)
    plt.show()
    print(np.mean(dmd_old.dmd_state))
    plt.imshow(dmd_old.dmd_state)
    plt.show()