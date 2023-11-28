pySLM2
======
[![Test](https://github.com/QITI/pySLM2/actions/workflows/pytest.yml/badge.svg)](https://github.com/QITI/pySLM2/actions/workflows/pytest.yml)
[![Build Documentation HTML](https://github.com/QITI/pySLM2/actions/workflows/sphinx.yml/badge.svg)](https://github.com/QITI/pySLM2/actions/workflows/sphinx.yml)

`pySLM2` is a python package for full stack control of using spatial light modulators (SLMs) for holographic beam shaping. 


Installation
------------
The dependencies of `pySLM2` in includes: `numpy`, `scipy`, `matplotlib`, and `tensorflow`.
`pySLM2` uses `tensorflow` for most of the numeric computation, so it can seamlessly use GPU to accelerate the computation without extra configuration.
In order to take advantage of the GPU, the correct version of `tensorflow` along with the cuda toolkits have to be installed.

For Anaconda users, it is recommended to install `tensorflow` with conda first for better performance and compatibility.
If you have a suitable GPU in the computer, you can install the `tensorflow` with GPU support by:
```
conda install -c anaconda tensorflow-gpu
```
Otherwise, if you don't have GPU available, you can isntall the `tensorflow` with:
```
conda install -c anaconda tensorflow
```

### With setuptools

If you prefer the development version from GitHub, download it here, `cd` to the pySLM2 directory, and use:
```
pip install .
```

Or, if you wish to edit the pySLM2 source code without re-installing each time

```
pip install -e .
```

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



