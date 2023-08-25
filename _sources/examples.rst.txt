Examples
========

Create a donut beam (Laguerre-Gauss mode) with DMD
--------------------------------------------------
This example shows how to create a donut beam with a digital mircromirror device (DMD). The input beam profile has fundamental Gaussian mode. However, with the hologram displayed on the DMD, the beam profile at the image plane becomes a donut beam (Laguerre-Gaussian $l=1$, $p=0$, mode). 

.. plot:: ../examples/create_donut_beam.py
   :include-source:


Create multiple beam spots with DMD
-----------------------------------
This example shows how to create multiple beam spots with a digital mircromirror device (DMD). The input beam profile has fundamental Gaussian mode. With the hologram displayed on the DMD, one can generate multiple beam spots at the image plane.

.. plot:: ../examples/create_multiple_gaussian_beam.py
   :include-source:


Load image to the Luxbeam DMD controller
----------------------------------------
.. literalinclude:: ../examples/load_images_luxbeam.py
   :language: python



