Fourier Hologram Generation Algorithms
======================================

Overview
--------
pySLM2 provides a suite of hologram generation algorithms optimized for various types of Spatial Light Modulator (``SLM``) modules, tailored to their specific phase and amplitude modulation capabilities.

For Liquid Crystal on Silicon (LCoS) modules, which are capable of pure phase modulation, we offer the Gerchberg-Saxton (``gs``) and Mixed-Region Amplitude Freedom (``mraf``) algorithms.

To select an algorithm, use the method keyword argument in the ``:func:slm.calculate_hologram`` function.

For Digital Micromirror Devices (DMDs), which enable both amplitude and phase modulation, 
we implement specialized methods for different control types. For binary amplitude control (on/off mirror states), 
the toolkit includes a Randomized Algorithm (``random``) and an Iterative Fourier Transformation Algorithm (``ifta``) for efficient hologram generation. 
Additionally, methods simulating continuous amplitude control are available under the ideal method, 
catering to a wider range of holographic applications.

For DMDs, use the method keyword argument in the ``:func:slm.calculate_dmd_state`` function to select the desired algorithm.

Functions
---------

pySLM2.slm.LCOS_SLM.calculate_hologram()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``gs``: It begins with a random phase distribution then iteratively refines it using Fourier transforms. 
- ``mraf``: It starts with a random phase profile and iteratively refines it, blending the desired amplitude profile in specified signal window regions with the current profile in other areas. 


pySLM2.slm.DMD.calculate_dmd_state()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``ideal``: It is used when having continuous amplitude control, generating grey scale hologram.
- ``simple``: It binarize the grey scale hologram from ``ideal`` method with a 0.5 threshold.
- ``random``: It utilizes hyperbolic tangent functions for smooth grating transition, followed by random threshold application for stochastic binary hologram generation.
- ``ifta``: It employs an iterative process to refine a grating pattern, progressively adjusting it towards a desired profile within a defined signal window using a dynamic thresholding strategy.
