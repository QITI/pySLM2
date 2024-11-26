---
title: 'pySLM2: A full-stack python package for holographic beam shaping'
tags:
  - Python
  - optics
  - trapped ions
  - physics
  - quantum information
authors:
  - name: Chung-You Shih
    orcid: 0000-0002-7561-6833
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Jingwen Zhu
    affiliation: "1"
    orcid: 0009-0009-0699-8258
    affiliation: "1"
  - name: Rajibul Islam
    orcid: 0000-0002-6483-8932
    affiliation: "1"
    

affiliations:
 - name: Institute for Quantum Computing and Department of Physics and Astronomy, University of Waterloo, 200 University Ave. West, Waterloo, Ontario N2L 3G1, Canada
   index: 1
date: 13 August 2017
bibliography: paper.bib

---

# Summary
Holographic beam shaping using spatial light modulators (SLMs) as a reprogrammable hologram offers a powerful tool for precise and flexible optical controls. It has been adopted for a wide range of researches, including atom trapping [@gaunt2012robust], optical addressing of individual quantum objects [@motlakunta2024], preparation of exotic quantum states [@islam2015measuring], and multi-beam laser machining [@obata2010multi].

`pySLM2` is a python package designed for holographic beam shaping applications, encompassing hologram generation, simulation, and hardware controls.

The package implements the hologram generation algorithms of the Lee hologram [@lee1978iii] and its improved alternatives [@zupancic2016ultra;@shih2021reprogrammable], specifically targeting the digital micromirror device (DMD) based SLM with binary amplitude controls. It also implements the Gerchberg-Saxton algorithm [@gerhberg1972practical] and its improved alternatives[@gaunt2012robust;@pasienski2008high] suitable for liquid crystal on silicon (LCoS) based SLMs with pure phase controls.

At its core, the package uses `TensorFlow` for numerical computations. By leveraging `TensorFlow`, the package harnesses the power of GPUs for faster computation without the need for code modification. This results in a significant speed-up for algorithms that are computationally expensive but benefit from parallelization, such as many hologram generation algorithms relying on iterative Fourier transformations.

In addition to hologram generation, the package provides functions to simulate beam profiles created by holograms, aiding users in evaluating algorithm performance. It also includes a variety of pre-defined optical profiles, such as Hermite-Gaussian, Laguerre-Gaussian, super-Gaussian, and Zernike polynomials, enabling users to construct target beam profiles with ease.

For hardware control, pySLM2 offers a unified application interface (API) compatible with various SLMs, ensuring seamless adaptation of code across different devices. Currently, the package supports DMD controllers from two commercial vendors: Visitech, INC and ViALUX GmbH.

# Statement of need
High-quality optical controls are crucial for numerous scientific and engineering applications. For instance, in atom-based quantum information processors, quantum states of individual atoms are often manipulated by individually addressing laser beams. The quality of these addressing beams directly impacts the fidelity of quantum operations [@motlakunta2024].

Holographic beam shaping using SLMs provides a way for precise and adaptive optical controls. Compared to using conventional optical elements, holographic beam shaping has several advantages. Firstly, it can generate arbitrary beam profiles that are challenging to create with standard optical elements. For example, the Laguerre-Gaussian beam with a non-zero azimuthal index (often referred to as a doughnut beam), which can be used to trap atoms in a tube-like potential [@kuga1997novel], apply angular momentum to Bose-Einstein Condensate [@andersen2006quantized], or achieve super-resolution imaging [@qian2021super;@drechsler2021optical].


Secondly, holographic beam shaping can actively correct cumulative optical aberrations in the system arising from almost inevitable causes, such as surface irregularities, misalignment, and imperfect lens curvature, thereby achieving diffraction-limited performance at the target image plane. This enables the faithful production of target beam profiles with high accuracy, relieving the stringent requirements on optics quality and alignment precision. It has been shown that residual wavefront aberrations can be corrected to less than $\lambda/20$ root-mean-square (RMS) [@shih2021reprogrammable;@zupancic2016ultra], which meets the Maréchal criteria (wavefront RMS error < $\lambda/14$) for optical performance. 


At the time of writing, the `pySLM2` package, as detailed in this manuscript, has been used in the trapped ion quantum information processing researches [@shih2021reprogrammable;@motlakunta2024;@kotibhaskar2023programmable]. The authors believe that the package will benefit a broader community of researchers and engineers by offering turnkey solutions for applying holographic beam shaping to their work. Moreover, the primitives included in the package can assist researchers in rapidly prototyping new hologram generation algorithms.

There are existing open-source packages available that specialize in different levels of holographic beam shaping with SLMs. For example, [`SLMLayout`](https://github.com/wavefrontshaping/Layout) is a package focusing on wavefront shaping with macropixel method [@matthes2019optical] with DMDs. [`CGH-diff`](https://github.com/DigitalNatureGroup/CGH-autodiff) offers `Tensorflow` scripts for an automatic differentiation-based algorithm [@yamamoto2021gradient] for DMD hologram generation. [`slmsuite`](https://slmsuite.readthedocs.io/) offers a comprehensive suite of hologram algorithms tailored for phase-only Spatial Light Modulators (SLMs), supported by GPU acceleration via `CuPy` [@cupy_learningsys2017], and includes hardware control capabilities for Liquid Crystal on Silicon (LCoS) SLMs. 

`pySLM2` implements hologram generation algorithms and primitives leveraging `TensorFlow` for GPU acceleration. Some algorithms, including the algorithms used in authors' prior works[@shih2021reprogrammable;@motlakunta2024] in the field of quantum information processing, are only available in `pySLM2` at the time of writing. In terms of hardware controls, `pySLM2` offers a unified API for programming DMDs from different vendors. 

We hope that the addition of `pySLM2` to the community will facilitate researchers in applying Fourier holographic beam shaping ideas on various machines with fewer hardware hurdles.

# Fourier Holography Basics

`pySLM2` is designed for holographic beam shaping using Fourier holography. The name "Fourier" comes from the fact that the electric fields of the beam at the hologram plane and the target plane are related by a Fourier transformation.

In a paraxial lens system, the lens act as a Fourier transform operator mapping the electric field in one focal plane to the electric field in the another focal plane. In the context of Fourier Holography, the two focal planes are referred as the image plane (IP) and the Fourier plane (FP). The electric fields at the two planes, $E_{\mathrm{IP}}(\mathbf{x}')$ and $E_{\mathrm{FP}}(\mathbf{x})$ respectively, are related by the following equation:

$$
E_{\mathrm{FP}}(\mathbf{x})\mathrm{e}^{\mathrm{i} \Phi_{ab}} = \left. \frac{\lambda f}{2 \pi}  \mathcal{F}\left  [E_{\mathrm{IP}} (\mathbf{x}')\right ](\mathbf{k}') \right | _{\mathbf{k}'=\frac{2 \pi}{\lambda f} \mathbf{x}}
$$
In which, $\mathbf{x}'$ and $\mathbf{k}'$ denote the spatial coordinate and the wave vector at the image plane respectively, and $\mathcal{F}$ denotes Fourier transformation. 
The wave vector $\mathbf{k}'$ is related to the spatial coordinate $\mathbf{x}$ at FP by $\mathbf{x} = \frac{\lambda f}{2 \pi}\mathbf{k}'$ where $f$ is the effective focal length of lens and $\lambda$ is the wavelength of the light.

The aberrations of the optical system can be modeled as a phase map $\Phi_{\mathrm{ab}}$ in the Fourier plane. In `pySLM2`'s convention, the plane SLM is placed is Fourier plane, and the image plane is where the targeted beam profile is desired. The SLM modulates the beam at Fourier plane to engineer the desired beam profiles at the image plane.

# Hologram Generation Algorithm
Currently, `pySLM2` supports two type of the spatial light modulator (SLM), liquid crystal on silicon (LCoS) SLM and digital micromirror device (DMD). The LCoS SLM modulates the phase profile purely without modifying the amplitude. As the time of writing, Gerchberg-Saxton (GS) [@gerhberg1972practical] algorithm and the mixed-region amplitude freedom (MRAF) algorithm [@gaunt2012robust;@pasienski2008high] are included. 

On the other hand, DMDs use micromirrors to locally turn on and off the light by toggling the micromirrors between two directions. This allows binary amplitude control. By periodically turning on and off the micromirrors across the DMD to form grating patterns, diffracted beams with controllable phase and amplitude can be engineered to have the desired beam profiles. As the time of writing, a randomized algorithm [@zupancic2016ultra] and an iterative Fourier transformation algorithm [@shih2021reprogrammable;@motlakunta2024] are provided for hologram generation.


# Usages
`pySLM2` offers commonly used optics profiles right out of the box, including Hermite Gaussian, Laguerre Gaussian, super Gaussian (also known as "flat top"), and Zernike polynomials. These profiles are implemented as functional objects, and `pySLM2` automatically handles the profile sampling during hologram calculations.

For profiles that are not included by default, users have the option to either inherit from the base class and implement their custom profiles or generate the sampled profiles in an array format to pass them to the hologram calculation function. As illustrated in \autoref{fig:lg}, here's an example of creating a hologram to generate a Laguerre Gaussian beam with a mode of $l=1$, $p=0$, which often referred to as a "doughnut beam", from the fundamental Gaussian mode. Unless specified, the simulation shown in this paper is simulated with the following conditions: $\lambda=369~\mathrm{nm}$ wavelength, $f=200~\mathrm{mm}$ Fourier lens focal length, and with Texas Instrument DLP9500 as the SLM ($1~\mathrm{px} = 10~\mu \mathrm{m}$ micromirror size).

![Hologram simulation for creating Laguerre Gaussian beam of $l=1$, $p=0$ mode from fundamental mode. (a) DMD mirror configuration. Bright pixels represent "on" and dark pixels represent "off". (b) Intensity profile of input fundamental Gaussian beam. (c) Intensity profile of the output Laguerre ($l=1$, $p=0$) Gaussian beam at the image plane. (d) Phase map of the output beam. An optical vortex can be observed at the center of the Laguerre $l=1$, $p=0$ mode (Source code: `examples/create_donut_beam.py`)\label{fig:lg}](create_donut_beam.png) 

The arithmetic operations of the profiles are also overloaded, so one can easily combine different profiles through addition or rescale the profiles through multiplication. Shown in \autoref{fig:multi}, we create a hologram to generate two Gaussian beams. In the source code, it is written as adding two Gaussian profiles together at different positions.


![Hologram simulation for creating two Gaussian beams from one input Gaussian beam. (a) DMD mirror configuration. Bright pixels represent "on" and dark pixels represent "off". (b) Intensity profile of input single Gaussian beam. (c) Intensity profile of the two output Gaussian beams at the image plane. (d) Phase map of the output beam. An example of two Gaussian beams having opposite phases is shown. (Source code: `examples/create_donut_beam.py`) \label{fig:multi}](create_multiple_gaussian_beam.png)


## Aberration Correction
One of the key advantages of holographic beam shaping is its capability to correct optical aberrations, and `pySLM2` provides an easy method to achieve this correction. By supplying the aberration information during the hologram calculation, `pySLM2` generates a hologram imprinted with a phase profile opposite to the aberration, effectively canceling the aberration out.

In the example depicted in \autoref{fig:aberration}, we simulate the beam profile at the image plane both with and without aberration correction. Without aberration correction, the beam profile becomes distorted and broadened. In this particular simulation, spherical aberration is used, but `pySLM2` is capable of correcting other types of aberrations as well.

![Simulation of the beam profiles at the image plane with and without aberration correction. (a) Phase map of the input beam with $\mathbb{Z}_4^0$ spherical aberration. (b) Intensity profile of the input beam. (c) Intensity profile of the first order beam without aberration correction. (d) Intensity profile of the first order beam with aberration correction. (Source code: `examples/aberration_correction.py`) \label{fig:aberration}](aberration_correction.png)


To obtain the phase map of the aberration, one can either use a wavefront sensor, such as a Shack–Hartmann sensor [@shack1971production;@Paschottashack_hartmann_wavefront_sensors], to measure the wavefront, or one can allow light from different parts of the Fourier plane to interfere with each other to reconstruct the aberration phase profile from the resulting interference patterns. For a detailed description of the latter method, one can refer to Shih et al. [@shih2021reprogrammable].

## Hardware Controls
`pySLM2` provides hardware controls for DMD controllers from both Visitech, INC and ViALUX GmbH. The controllers from these two companies use different communication protocols and architectures. The Visitech controller uses UDP over Ethernet, while the ViALUX controller uses USB3.

One of the goals of `pySLM2` is to abstract the hardware details and offer a unified application interface for interacting with these devices. For instance, we have implemented the same `load_single` and `load_multiple` functions within the controller classes for both manufacturers' devices. These functions allow for the display of single holograms or the loading of multiple holograms that can be switched by triggers. Apart from the hardware-agnostic functions, it also exposed the lower-level access for advanced users to implement device specific controls.

As of the current writing, the package's hardware support is limited to DMD controllers. For users interested in using LCoS-SLM, open-source tools such as [`slmsuite`](https://slmsuite.readthedocs.io/) and [`slmPy`](https://github.com/wavefrontshaping/slmPy) are available options.

# Author Contributions
C.-Y.S. designed and implemented the package. J.Z. contributed to implementing the hardware controls and performance benchmarking. R.I. advised on the scientific aspects of the package. All authors contributed to writing the manuscript and documenting the package.

# Acknowledgements
The hardware controls for the DMDs from ViALUX GmbH in the package is built on top of the `AL4lib` [@sebastien_m_popoff_2022_6121191]. We appreciate the work of the authors of `AL4lib`. We express gratitude to Kaleb Ruscitti for assisting with hardware testing and to Sainath Motlakunta and Nikhil Kotibhaskar for providing valuable feedback on the package. 

We acknowledge financial support from the Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery program (RGPIN-2018-05250) and the Institute for Quantum Computing for this work. 

# References
