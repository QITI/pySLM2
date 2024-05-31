# Fourier Hologram Generation Algorithms

## Overview

pySLM2 provides a suite of hologram generation algorithms optimized for various types of Spatial Light Modulator (SLM) modules, tailored to their specific phase and amplitude modulation capabilities.

For Liquid Crystal on Silicon (LCoS) modules, which are capable of pure phase modulation, we offer the Gerchberg-Saxton (`gs`) [[1]](#ref1) and Mixed-Region Amplitude Freedom (`mraf`) [[2]](#ref2) algorithms. 

For SLMs, use the method keyword argument in the `slm.calculate_hologram` function to apply the desired algorithm.

For Digital Micromirror Devices (DMDs), which enable both amplitude and phase modulation, we implement specialized methods for different control types. The `ideal` method is used for simulating hologram when having continuous amplitude control on DMD. Three binarization methods are packaged along with it: the `simple` method for hard cut-off binatization, `random` method for randomized binarization [[3]](#ref3), and an iterative Fourier transformation algorithm `ifta` with iterative binarization with adaptive binarization threshold [[4]](#ref4).  

For DMDs, use the method keyword argument in the `slm.calculate_dmd_state` function to appply the desired algorithm.

## Functions

### pySLM2.slm.LCOS_SLM.calculate_hologram()

- `gs`: The Gerchberg-Saxton algorithm is an iterative phase retrieval process. The refined phase profile ideally would converge to the target phase profile after enough iterations. However, the convergence can be slow and is not guaranteed. [[1]](#ref1)  When using this method, the user can use kwarg `N` to define number of iterations desired. 
- `mraf`: The Mixed-Region Amplitude Freedom algorithm is an improved adpataion of the `gs` algorihtms which interatively refines the phase profile within a specified signal window region. It improves the convergence within the signal window at the sacrifice of phase control in the outside region. [[2]](#ref2) When using this method, in addtion to `N`, user can also define `signal_window` as the region that needs correction, as well as `mixing_factor`, which is to adjust how much the outside region weighs in the correction. When `mixing_factor = 1`, no outside region is being weighted and correctino is only applied for signal window region. 

### pySLM2.slm.DMD.calculate_dmd_state()

- `ideal`: It assumes continuous amplitude control, generating the ideal grey-scale hologram with no binarization induced errors.
- `simple`: It binarizes the ideal hologram with a 0.5 threshold for the grating phase. However, the hard cut-off can induce significant binarization induced errors.
- `random`: It binarizes the ideal hologram with random threshold by probabilisticly turn on or off pixel mirrors based on the grating phase. [[3]](#ref3)  It reduces binarization artifacts compared to the `simple` method. The user can set the kwarg `r` to adjust the steepness of the binarization function. The larger the `r`, the sharper the binarization would be.
- `ifta`: It iteratively binarizes the holgram within a signal window by deterministic the pixel mirror with adaptive binarizaitno threshold. [[4]](#ref4) This method yields lower RMS error within the signal window compared to the `random` method.  The user can use kwarg `N` to define number of iterations desired, and `s` to adjust the step size at which the adpative binazation threshold increment by after each iteration.


## References

<a id="ref1"></a>1. R Gerhberg, W Saxton. "A practical algorithm for the determination of phase from image and diffraction plane pictures." *Optik*, vol. 35, pp. 237-246, 1972.

<a id="ref2"></a>2. Alexander L Gaunt, Zoran Hadzibabic. "Robust digital holography for ultracold atom trapping." *Scientific reports*, vol. 2, no. 1, p. 721, 2012. Nature Publishing Group UK London.

<a id="ref3"></a>3. Philip Zupancic, Philipp M Preiss, Ruichao Ma, Alexander Lukin, M Eric Tai, Matthew Rispoli, Rajibul Islam, Markus Greiner. "Ultra-precise holographic beam shaping for microscopic quantum control." *Optics express*, vol. 24, no. 13, pp. 13881-13893, 2016. Optica Publishing Group.

<a id="ref4"></a>4. Chung-You Shih, Sainath Motlakunta, Nikhil Kotibhaskar, Manas Sajjan, Roland Hablützel, Rajibul Islam. "Reprogrammable and high-precision holographic optical addressing of trapped ions for scalable quantum control." *npj Quantum Information*, vol. 7, no. 1, p. 57, 2021. Nature Publishing Group UK London.