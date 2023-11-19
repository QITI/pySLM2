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
  - name: Kazi Rajibul Islam
    affiliation: "1"
    

affiliations:
 - name: Institute for Quantum Computing and Department of Physics and Astronomy, University of Waterloo, 200 University Ave. West, Waterloo, Ontario N2L 3G1, Canada
   index: 1
date: 13 August 2017
bibliography: paper.bib

---

# Summary
Holographic beam shaping using spatial light modulators (SLMs) as a reprogrammable hologram offers a powerful tool for precise and flexible optical controls. It has been adopted for a wide range of research, including atom trapping[@gaunt2012robust], optical addressing of individual quantum objects[@motlakunta2023preserving;@islam2015measuring], and multi-beam laser machining[@obata2010multi].

`pySLM2` is a python package designed for full-stack control of SLMs for holographic beam shaping, encompassing hologram generation, simulation, and hardware controls.

The package implements the hologram generation algorithms of the Lee hologram[@lee1978iii] and its improved predecessors from @zupancic2016ultra and @shih2021reprogrammable, targeting the digital micromirror device (DMD) based SLM with binary amplitude controls. It also implements the Gerchberg-Saxton algorithm[@gerhberg1972practical] suitable for liquid crystal on silicon (LCoS) based SLMs with pure phase controls.

Under the hood, the package uses `TensorFlow` for intensive numerical computations. By leveraging `TensorFlow`, the package harnesses the power of GPUs for faster computation without the need for code modification. This results in a significant speed-up for algorithms that require robust numerical computation, such as many hologram generation algorithms relying on iterative Fourier transformations.

Additionally, the package offers a universal interface for different SLMs, ensuring that code written for one device can be seamlessly adapted to another. As of this writing, the package supports DMDs from both Visitech and Vialux.

# Statement of need
High-quality optical controls are crucial for numerous scientific and engineering applications. For instance, in atom-based quantum information processors, individual atom quantum states are often manipulated by individually addressable laser beams. The quality of these addressing beams directly impacts the fidelity of quantum operations[@motlakunta2023preserving].

Holographic beam shaping using SLMs provides a potent tool for precise and adaptive optical controls. Compared to conventional optical elements, holographic beam shaping has several advantages. Firstly, it can generate arbitrary beam profiles that are challenging to create with standard optical elements. For example, the Laguerre-Gaussian beam with a non-zero azimuthal index (often referred to as a doughnut beam), which can be used to trap atoms in a tube-like potential[@kuga1997novel], apply angular momentum to Bose-Einstein Condensation[@andersen2006quantized], or achieve super-resolution imaging[@qian2021super;drechsler2021optical].

Secondly, holographic beam shaping can actively correct optical aberrations in the system, achieving diffraction-limited performance. This enables the faithful production of target beam profiles with high accuracy. It has been shown that residual aberrations can be corrected to less than $\lambda/20$ root-mean-squer (RMS)[@shih2021reprogrammable;@zupancic2016ultra], which is smaller than most of the manufacturing tolerances of optical elements.

At the time of writing, the pySLM2 package, as detailed in this manuscript, has been employed in the research of @shih2021reprogrammable, @motlakunta2023preserving, and @kotibhaskar2023programmable. The authors believe that the package will benefit a broader community of researchers and engineers by offering turnkey solutions for applying holographic beam shaping to their work. Moreover, the primitives included in the package can assist researchers in rapidly prototyping new hologram generation algorithms.


# Fourier Holography Basics

`pySLM2` is designed for holographic beam shaping using Fourier holography. The name "Fourier" comes from the fact that the hologram and the target beam profile are related by a Fourier transformation.

 In a paraxial lens system, the lens act as a Fourier transform operator mapping the electric field in one focal plane to the electric field in the another focal plane. In the context of Fourier Holography, the two focal plane are referred as the image plane (IP) and the Fourier plane (FP). The electric field at the two planes, $E_{\rm{IP}}(\mathbf{x}')$ and $E_{\rm{FP}}(\mathbf{x})$ respectively, are related by the following equation:

$$
E_{\rm{FP}}(\mathbf{x})\mathrm{e}^{\mathrm{i} \Phi_{ab}} = \left. \frac{\lambda f}{2 \pi}  \mathcal{F}\left [E_{\rm{IP}} (\mathbf{x}')\right ](\mathbf{k}') \right | _{\mathbf{k}'=\frac{2 \pi}{\lambda f} \mathbf{x}}
$$
In which, $\mathbf{x}'$ and $\mathbf{k}'$ denote the spatial coordinate and the wave vector at the image plane respectively, and $\mathcal{F}$ denotes Fourier transformation. The wave vector $\mathbf{k}'$ is related to the spatial coordinate $\mathbf{x}$ at FP by $ \mathbf{x} = \frac{\lambda f}{2 \pi}\mathbf{k}'$. 

The aberrations of the optical system can be modeled as a phase map $\Phi_{\mathrm{ab}}$ in the Fourier plane. In `pySLM2`'s convention, the plane SLM is placed is Fourier plane, and the image plane is where the targeted beam profile is desired. 

* `input_profile`: The beam profile at the Forier plane before modulated by the SLM. Typically, this is beam profile of the beam one used to illuminate the SLM.
* `output_profile`: The beam profile at the Fourier plane after modulated by the SLM.
* `target_profile`: The desired beam profile at the image plane.

To calculate the hologram, we will need both `input_profile` and `target_profile`. The `output_profile` is not required but one can extract it from the simulation.

# Hardware Controls
`pySLM2` provides with the hardware controls of the DMDs from Visitech and Vialux. The controllers from the two companies uses different communication protocols and architectures.
A goal of `pySLM2` is to provide a universal application interface for interacting with the hardwares.

For both controllers, we implemented ....

<!--- 
More 
-->



<!-- 
# Usage -->

<!-- 
# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

The hardware controls for the DMDs from Vialux in the package is built on top of the `AL4lib`[@sebastien_m_popoff_2022_6121191]. We appreciate the work of the authors of `AL4lib`.



<!-- We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project. -->

# References