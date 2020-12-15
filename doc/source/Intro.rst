Introduction
============

pySLM2 is not ready for public release yet...

Installation
------------
The dependencies of pySLM2 in includes: numpy, scipy, matplotlib, and tensorflow.
pySLM2 uses tensorflow for most of the numeric computation, so it can seamlessly use GPU to accelerate the computation without extra configuration.
In order to take advantage of the GPU, the correct version of tensorflow along with the cuda toolkits have to be installed.


With conda
~~~~~~~~~~
coming soon...

With pip
~~~~~~~~
coming soon...

With setuptools
~~~~~~~~~~~~~~~

If you prefer the development version from GitHub, download it here, `cd` to the pySLM2 directory, and use ::

    python setup.py install

Or, if you wish to edit the pySLM2 source code without re-installing each time ::

    python setup.py develop

Optional Dependencies
---------------------
pySLM2.util includes provides an universal interface for interacting with DMD controllers from different vendors.
pySLM2.util itself doesn't implement the communication protocal. Instead, it relies on different libraries and wraps them with a universal interface.

Vialux
~~~~~~
The DMDs from Vialux are communicated with the ALP library and a python binding, ALP4lib.

* To install the ALP library, visit the vendors website: https://vialux.de/en/download.html
* The python binding, ALP4lib, can be installed from git. Note that the ALP4lib used in pySLM2.util is a fork from the original page and curently maintined by the QITI lab.::

    pip install https://github.com/QITI/ALP4lib

Visitech
~~~~~~~~
The DMD from Visitech are communicates with UDP. It relies on the Luxbeam library which can be installed through git::

    pip install https://github.com/QITI/Luxbeam

Precision
---------
Since not all the graphic cards are optimized for double-precision computation, the default precision of pySLM2 is set to single-precision.
If you wish to use double-precision, you can configure the backend right after you import the package ::

   pySLM2.BACKEND.dtype = pySLM2.BACKEND.TENSOR_64BITS

