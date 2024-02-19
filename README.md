pySLM2
======
[![Test](https://github.com/QITI/pySLM2/actions/workflows/pytest.yml/badge.svg)](https://github.com/QITI/pySLM2/actions/workflows/pytest.yml)
[![Build Documentation HTML](https://github.com/QITI/pySLM2/actions/workflows/sphinx.yml/badge.svg)](https://github.com/QITI/pySLM2/actions/workflows/sphinx.yml)
[![status](https://joss.theoj.org/papers/70246ad674d3806798e343f6ecffa686/status.svg)](https://joss.theoj.org/papers/70246ad674d3806798e343f6ecffa686)

**We are writing a paper for [Journal of Open Source Software (JOSS)](https://joss.theoj.org/)! Read the manuscript here: https://drive.proton.me/urls/T92N7SMT5M#nBvZvODDgZCc**

`pySLM2` is a python package for full stack control of using spatial light modulators (SLMs) for holographic beam shaping. 

* API Docs: https://pyslm2.pages.dev/

Instructions to build documentation locally can be found in [`docs/README.md`](docs/README.md).


Dependencies
------------
pySLM2 supports Python 3.9+. 

The dependencies of `pySLM2` in includes: `numpy`, `scipy`, `matplotlib`, and `tensorflow`.


Installation
------------

### With setuptools

If you prefer the development version from GitHub, download it here, `cd` to the pySLM2 directory, and use:
```
pip install .
```

Or, if you wish to edit the pySLM2 source code without re-installing each time

```
pip install -e .
```

Tensorflow Installation for GPU Support
---------------------------------------

`pySLM2` primarily relies on `tensorflow` for most of its numerical computations. For machines with compatible hardware, `tensorflow` can seamlessly utilize GPU acceleration to enhance performance, provided it is installed correctly.
#### Linux and Windows WLS2 users please use the following line to install tensorflow for gpu support. 
```
python3 -m pip install tensorflow[and-cuda]
```
For more detailes about machine compatibility and correct version of tensorflow, please refer to the [instructions](https://www.tensorflow.org/install/pip#step-by-step_instructions) on Tensorflow's website for more details.

Optional Dependencies for Hardware Controls
-------------------------------------------
`pySLM2.util` includes provides an universal interface for interacting with different SLM controllers from different vendors.
`pySLM2.util` itself doesn't implement the communication protocal. Instead, it relies on different libraries and wraps them with a universal interface.

### Vialux
The DMDs from Vialux are communicated with the ALP library and a python binding, ALP4lib.

* To install the ALP library, visit the vendors website: https://vialux.de/en/download.html
* The python binding, ALP4lib, can be installed from PyPi:
```
pip install ALP4lib
```

### Visitech
The DMD from Visitech are communicates with UDP, and therefore no driver is needed to be installed. `pySLM.util` relies on the [`Luxbeam`](https://pypi.org/project/Luxbeam/) library which can be installed from PyPi:
```
pip install Luxbeam
```

Floating Point Precision
------------------------
Since not all the graphic cards are optimized for double-precision computation, the default precision of pySLM2 is set to single-precision.
If you wish to use double-precision, you can configure the backend right after you import the package :
```
pySLM2.BACKEND.dtype = pySLM2.BACKEND.TENSOR_64BITS
```

In practice, the single-precision is sufficient for most of the applications. However, if you are suspecting the precision is not enough, you can change the precision to double-precision and see if the results are different.

Research using pySLM2
---------------------

If you used pySLM2 in your research, we'd like to hear from you!

* Shih, Chung-You, et al. "Reprogrammable and high-precision holographic optical addressing of trapped ions for scalable quantum control." npj Quantum Information 7.1 (2021): 57. [https://doi.org/10.1038/s41534-021-00396-0](https://doi.org/10.1038/s41534-021-00396-0)

* Motlakunta, Sainath, et al. "Preserving a qubit during adjacent measurements at a few micrometers distance." arXiv preprint arXiv:2306.03075 (2023). [https://doi.org/10.48550/arXiv.2306.03075](https://doi.org/10.48550/arXiv.2306.03075)



