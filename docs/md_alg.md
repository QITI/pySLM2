# Fourier Hologram Generation Algorithms

## Overview

pySLM2 provides a suite of hologram generation algorithms optimized for various types of Spatial Light Modulator (SLM) modules, tailored to their specific phase and amplitude modulation capabilities.

For Liquid Crystal on Silicon (LCoS) modules, which are capable of pure phase modulation, we offer the Gerchberg-Saxton (`gs`) and Mixed-Region Amplitude Freedom (`mraf`) algorithms. [[3]](#ref3)

For SLMs, use the method keyword argument in the `slm.calculate_hologram` function to apply the desired algorithm.

For Digital Micromirror Devices (DMDs), which enable both amplitude and phase modulation, we implement specialized methods for different control types. The `ideal` method is used for simulating hologram when having continuous amplitude control on DMD. Three binarization methods are packaged along with it: the `simple` method for hard cut-off binatization, `random` method for randomized binarization [[2]](#ref2), and an interative Fourier transformation algorithm `ifta` with adaptive binarization.  

For DMDs, use the method keyword argument in the `slm.calculate_dmd_state` function to appply the desired algorithm.

## Functions

### pySLM2.slm.LCOS_SLM.calculate_hologram()

- `gs`: The Gerchberg-Saxton algorithm is an iterative phase retrieval process. The refined phase profile ideally would converge to the target phase profile after enough iterations. However, this convergence can be slow and is not guaranteed. [[4]](#ref4)
- `mraf`: The Mixed-Region Amplitude Freedom algorithm is an improved adpataion of the `gs` algorihtms which interatively refines the phase profile within a specified signal window region. It improves the convergence within the signal window at the sacrifice of phase control in the outside region. [[3]](#ref3)

### pySLM2.slm.DMD.calculate_dmd_state()

- `ideal`: It assumes continuous amplitude control, generating the ideal grey-scale hologram with no binarization induced errors.
- `simple`: It binarizes the ideal hologram with a 0.5 threshold for the grating phase. However, the hard cut-off can induce significant binarization induced errors.
- `random`: It binarizes the ideal hologram with random threshold by probabilisticly turn on or off pixel mirrors based on the grating phase. It reduces binarization artifacts compared to the `simple` method. [[2]](#ref2).
- `ifta`: It interatively refines the holgram within a signal window by deterministically binarize the pixel mirror with adaptive binarizaitno threshold. This method yields lower RMS error within the signal window compared to the `random` method. [[1](#ref1), [5](#ref5)]


## References

<a id="ref1"></a>1. Chung-You Shih, Sainath Motlakunta, Nikhil Kotibhaskar, Manas Sajjan, Roland Hablützel, Rajibul Islam. "Reprogrammable and high-precision holographic optical addressing of trapped ions for scalable quantum control." *npj Quantum Information*, vol. 7, no. 1, p. 57, 2021. Nature Publishing Group UK London.

<a id="ref2"></a>2. Philip Zupancic, Philipp M Preiss, Ruichao Ma, Alexander Lukin, M Eric Tai, Matthew Rispoli, Rajibul Islam, Markus Greiner. "Ultra-precise holographic beam shaping for microscopic quantum control." *Optics express*, vol. 24, no. 13, pp. 13881-13893, 2016. Optica Publishing Group.

<a id="ref3"></a>3. Wai-Hon Lee. "III computer-generated holograms: Techniques and applications." In *Progress in optics*, vol. 16, pp. 119-232, 1978. Elsevier.

<a id="ref4"></a>4. R Gerhberg, W Saxton. "A practical algorithm for the determination of phase from image and diffraction plane pictures." *Optik*, vol. 35, pp. 237-246, 1972.

<a id="ref5"></a>5. Sainath Motlakunta, Nikhil Kotibhaskar, Chung-You Shih, Anthony Vogliano, Darian McLaren, Lewis Hahn, Jingwen Zhu, Roland Hablützel, Rajibul Islam. "Preserving a qubit during adjacent measurements at a few micrometers distance." *arXiv preprint arXiv:2306.03075*, 2023.

<a id="ref6"></a>6. Sébastien M. Popoff, Gilbert Shih, Dirk B., Gustave Pariente. "wavefrontshaping/ALP4lib: 1.0.1." February 2022. Zenodo. DOI: [10.5281/zenodo.6121191](https://doi.org/10.5281/zenodo.6121191).

<a id="ref7"></a>7. Alexander L Gaunt, Zoran Hadzibabic. "Robust digital holography for ultracold atom trapping." *Scientific reports*, vol. 2, no. 1, p. 721, 2012. Nature Publishing Group UK London.

<a id="ref8"></a>8. Zhong-Hua Qian, Jin-Ming Cui, Xi-Wang Luo, Yong-Xiang Zheng, Yun-Feng Huang, Ming-Zhong Ai, Ran He, Chuan-Feng Li, Guang-Can Guo. "Super-resolved imaging of a single cold atom on a nanosecond timescale." *Physical review letters*, vol. 127, no. 26, p. 263603, 2021. APS.

<a id="ref9"></a>9. Martín Drechsler, Sebastian Wolf, Christian T Schmiegelow, Ferdinand Schmidt-Kaler. "Optical superresolution sensing of a trapped ion’s wave packet size." *Physical Review Letters*, vol. 127, no. 14, p. 143602, 2021. APS.

<a id="ref10"></a>10. Nikhil Kotibhaskar, Chung-You Shih, Sainath Motlakunta, Anthony Vogliano, Lewis Hahn, Yu-Ting Chen, Rajibul Islam. "Programmable XY-type couplings through parallel spin-dependent forces on the same trapped ion motional modes." *arXiv preprint arXiv:2307.04922*, 2023.
