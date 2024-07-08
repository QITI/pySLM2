# Fourier Hologram Generation Algorithms

## Overview

pySLM2 provides a suite of hologram generation algorithms optimized for various types of Spatial Light Modulator (SLM) modules, tailored to their specific phase and amplitude modulation capabilities.

For Liquid Crystal on Silicon (LCoS) modules, which are capable of pure phase modulation, we implement the Gerchberg-Saxton (`gs`) [[1]](#ref1) and Mixed-Region Amplitude Freedom (`mraf`) [[2]](#ref2)[[7]](#ref7) algorithms. 

For SLMs, use the `method` keyword argument in the `slm.calculate_hologram` function to select the desired algorithm.

For Digital Micromirror Devices (DMDs), which is capable of binary amplitude modulation, we implement the Lee hologram methods (`ideal`, `simple`), a randomized algorithm (`random`) [[3]](#ref3), and an Iterative Fourier Transform Algorithm (`ifta`) [[4]](#ref4) algorithms.

For DMDs, use the method keyword argument in the `slm.calculate_dmd_state` function to select the desired algorithm.

## Algorithms

### `pySLM2.slm.LCOS_SLM.calculate_hologram`

- `gs`: The Gerchberg-Saxton algorithm is an iterative phase retrieval process. The refined phase profile ideally would converge to the target phase profile after enough iterations. However, the convergence can be slow and is not guaranteed. [[1]](#ref1)  When using this method, the user can use kwarg `N` to define number of iterations desired. 
- `mraf`: The Mixed-Region Amplitude Freedom algorithm is an improved adaption of the `gs` algorithms which iteratively refines the phase profile within a specified signal window region. It improves the convergence within the signal window at the sacrifice of phase control in the outside region. [[2]](#ref2)[[7]](#ref7) When using this method, in addition to `N`, user can also define `signal_window` as the region that needs correction, as well as `mixing_factor`, which is to adjust how much the outside region weighs in the correction. When `mixing_factor = 1`, no outside region is being weighted and correction is only applied for signal window region. Suggested in [[2]](#ref2), the `mixing_factor` set to `0.4` typically yields the lowest RMS error.

### `pySLM2.slm.DMD.calculate_dmd_state`

- `ideal`: Grayscale Lee hologram[[5]](#ref5) without binarization. This method constructs amplitude gratings with an analytical solution such that the first-order diffraction beam exactly matches the target beam profile. The calculated DMD state has a float data type between 0 and 1 rather than a bool data type. Though the generated beam profile is ideal, it is not directly applicable to DMDs. However, one may want to use it for comparison purposes.
- `simple`: This method simply binarizes the ideal hologram with a `0.5` threshold. However, the hard cut-off typically introduces a significant amount of binarization errors.
- `random`: This method constructs the binary amplitude grating with a randomized binarization threshold. [[3]](#ref3)  It reduces binarization artifacts compared to the `simple` method. The user can set the kwarg `r` to adjust the steepness of the binarization function. The larger the `r`, the sharper the binarization would be.
- `ifta`: It iteratively binarizes the ideal Lee hologram with an adaptive binarization threshold and constantly corrects the binarization error within a signal window during the process. [[4]](#ref4) This method yields lower RMS error within the signal window compared to the `random` method. The user can use kwarg `N` to define the number of iterations desire. Typically, the more iterations, the lower the RMS error. In addition, `s` [[6]](#ref6) is a scaling parameter that scales the ideal Lee hologram prior to the binarization. One may increase this parameter (default to `1`) to increase the overall diffraction efficiency of the hologram. Comparing to the `random` method, the `ifta` method is more computationally expensive. It is recommended to uses GPU for faster computation.

## References

<a id="ref1"></a>1. R Gerhberg, W Saxton. "A practical algorithm for the determination of phase from image and diffraction plane pictures." *Optik*, vol. 35, pp. 237-246, 1972.

<a id="ref2"></a>2. Alexander L Gaunt, Zoran Hadzibabic. "Robust digital holography for ultracold atom trapping." *Scientific reports*, vol. 2, no. 1, p. 721, 2012. Nature Publishing Group UK London.

<a id="ref3"></a>3. Philip Zupancic, Philipp M Preiss, Ruichao Ma, Alexander Lukin, M Eric Tai, Matthew Rispoli, Rajibul Islam, Markus Greiner. "Ultra-precise holographic beam shaping for microscopic quantum control." *Optics express*, vol. 24, no. 13, pp. 13881-13893, 2016. Optica Publishing Group.

<a id="ref4"></a>4. Chung-You Shih, Sainath Motlakunta, Nikhil Kotibhaskar, Manas Sajjan, Roland Hablützel, Rajibul Islam. "Reprogrammable and high-precision holographic optical addressing of trapped ions for scalable quantum control." *npj Quantum Information*, vol. 7, no. 1, p. 57, 2021. Nature Publishing Group UK London.

<a id="ref5"></a>5. Lee, Wai-Hon. "III computer-generated holograms: Techniques and applications." Progress in optics. Vol. 16. Elsevier, 1978. 119-232. 
 
<a id="ref6"></a>6. Motlakunta, Sainath, et al. "Preserving a qubit during adjacent measurements at a few micrometers distance." arXiv preprint arXiv:2306.03075 (2023).

<a id="ref7"></a>7. Pasienski, Matthew, and Brian DeMarco. "A high-accuracy algorithm for designing arbitrary holographic atom traps." Optics express 16.3 (2008): 2176-2190.