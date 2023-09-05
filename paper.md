---
title: 'pySLM2: A python package for full stack module for holographic beam shaping'
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

affiliations:
 - name: Institute for Quantum Computing and Department of Physics and Astronomy, University of Waterloo, 200 University Ave. West, Waterloo, Ontario N2L 3G1, Canada
   index: 1
date: 13 August 2017
bibliography: paper.bib

---

# Summary
Holographic beam shaping using spatial light modulators (SLMs) as a reporgramable Fourier hologram offers a powerful tool for precise and flexible optical controls and has been used in many areas of research, including atom trapping, manipulating the quantum state of atomic qubit,

`pySLM2` is a python package for full stack control of using spatial light modulators (SLMs) for holographic beam shaping, including hologram generation, simulation, and hardware controls. 

The packages implemnts the hologram generation algorithms of the Lee hologram[@lee1978iii] and its predecesors from @zupancic2016ultra and @shih2021reprogrammable, tragetting the digital micromirror device (DMD) based SLM. It also implemnts the Gram-Schmidt alogorithm[@gerhberg1972practical] that can be applied liqcuid crystal on silicon (LCoS) based SLM.

Under the hood, the package uses `TensorFlow` for heavy lifting numerical computation. The use of `TensorFlow` allows the package to harness the power of GPU for fast computation without the need of changing the code. This provides orders of magnitude speed up for algorthims that requires heavy numerical computation, such as the algorithms relies on iterative Fourier transformation.

The package provides a universal interface for different SLMs, so the code writen for one device can be easily adapted to another one. As the time of writing, the package supports the DMDs from both Visitech and Vialux.

The package has been used in the research of @shih2021reprogrammable and @motlakunta2023preserving.

# Statement of need
High quality optical controls are essential for many scientific and engineering applications. For example, in atom-based quantum information processor, the control of the quantum state of individual atoms is often done by individually adressable laser beams. The quality of the addressing beams directly affects the fidelity of the quantum operations. 

Holographic beam shapping using spatial light modulators (SLMs) offers a powerful tool for precise and flexible optical controls.
Compare to using convential optical elements, the holographic beam shapping offers several advantages. First, the holographic beam shapping can be used to generate arbitrary beam profiles that is not trivial to created with convential optical elements. For instance, Laguerre-Gaussian beam with non-zero azimuthal index (often referred as doughnut beam) can be used to trap atoms in a tube-like potential, applying angular momentum to the Bose-Einstein Condensation, and performing superresolution imaging. 

Second, the holographic beam shapping can also correct the optical aberrations in the system, leading to a diffraction limited optical system. This allows engineering of high quality beam profiles. It has been demonstrated that the residual aberrtion can be corrected to less than $\lambda/20$[@shih2021reprogrammable].


# Usage

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