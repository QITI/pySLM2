Fourier Hologram Generation Algorithms
======================================

Overview
--------
``pySLM2`` offers implementations of several hologram generation algorithms that are suitable for different SLM modules 
depending on the availability of phase and amplitude modulations.

For liquid crystal on silicon (LCoS) that only has pure phase controls, we implemented the Gerchberg-Saxton (``gs``) algorithm, 
and the mixed-region amplitude freedom (``mraf``) algorithm.

To select which algorithm to use, change the ``method`` kwarg in the :func:`slm.calculate_hologram` function. 

For digital micromirror device (DMD) that has both amplitude and phase controls. 
For the ones with binary amplitude control (can switch mirrors on or off), we implemented a randomized algorithm (``random``), 
and an iterative Fourier transformation algorithm (``ifta``) for hologram generation.
We also provides simualtion methods with continuous amplitude control , including ``ideal``, ``ideal_square``, and ``simple``.

Functions
---------

pySLM2.slm.LCOS_SLM.calculate_hologram()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``gs``: It begins with a random phase distribution then iteratively refines it using Fourier transforms. 
.. In each iteration, the algorithm modulates the input light field with the current phase profile, 
.. Fourier transforms it to the image plane, 
.. adjusts the phase to match the desired amplitude pattern, 
.. and then transforms back to update the phase profile using inverse Fourier transform. 
- ``mraf``: It starts with a random phase profile and iteratively refines it, blending the desired amplitude profile in specified signal window regions with the current profile in other areas. 
.. This process involves modulating the input profile with the current phase, 
.. performing inverse Fourier transforms to and from the image plane, 
.. and adjusting the phase based on a mix of target and current amplitudes. 
.. The final phase profile, after a specified number of iterations, 
.. aims to produce the target intensity pattern within defined regions at the image plane.

pySLM2.slm.DMD.calculate_dmd_state()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``ideal`` is used when having continuous amplitude control, generating grey scale hologram.
- ``ideal_square``
- ``simple`` generates a grey scale hologram and put a 0.5 threshold to binarize the continuous amplitude to 0 and 1.
- ``random``
- ``ifta``
