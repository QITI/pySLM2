pySLM2
======
[![Test](https://github.com/QITI/pySLM2/actions/workflows/pytest.yml/badge.svg)](https://github.com/QITI/pySLM2/actions/workflows/pytest.yml)
[![Build Documentation HTML](https://github.com/QITI/pySLM2/actions/workflows/sphinx.yml/badge.svg)](https://github.com/QITI/pySLM2/actions/workflows/sphinx.yml)
[![status](https://joss.theoj.org/papers/70246ad674d3806798e343f6ecffa686/status.svg)](https://joss.theoj.org/papers/70246ad674d3806798e343f6ecffa686)

**We are writing a paper for [Journal of Open Source Software (JOSS)](https://joss.theoj.org/)! Read the manuscript here: https://drive.proton.me/urls/T92N7SMT5M#nBvZvODDgZCc**

`pySLM2` is a Python package designed for using spatial light modulators (SLMs) in holographic beam shaping. It includes modules for hologram generation, simulation, and hardware control, making it a comprehensive toolkit for high-quality optical control.

The goal of pySLM2 is to provide a tool box for for engineering high-quality optical controls, which are essential for various scientific and engineering applications. These applications include atom trapping, addressing individual quantum objects, preparing exotic quantum states, and multi-beam laser machining. `pySLM2` was originally developed for and is actively used in the trapped ion quantum information processing research at the Quantum Information with Trapped Ions Lab at the University of Waterloo.

* API Docs: https://pyslm2.pages.dev/

Instructions to build documentation locally can be found in [`docs/README.md`](docs/README.md).

Dependencies
------------
pySLM2 supports Python 3.9+. 

The dependencies of `pySLM2` in includes: `numpy`, `scipy`, `matplotlib`, and `tensorflow`.


Installation
------------

### pySLM2 installation with setuptools

If you prefer the development version from GitHub, download it here, `cd` to the pySLM2 directory, and use:
```
pip install .
```

Or, if you wish to edit the pySLM2 source code without re-installing each time

```
pip install -e .
```

GPU Support via Tensorflow
---------------------------------------
### Tensorflow installation tips

`pySLM2` primarily relies on `tensorflow` for most of its numerical computations. For machines with compatible hardware, `tensorflow` can seamlessly utilize GPU acceleration to enhance performance, provided it is installed correctly.

The exact package dependencies vary depending on each system configuration and the GPU card. For details about machine compatibility and correct version of tensorflow, please refer to the [Tensorflow's website](https://www.tensorflow.org/install/pip#step-by-step_instructions), which provides installation guide for different operating systems. Another authors' recommended `tensorflow` installation guide can also be found in this [website](https://medium.com/@shaikhmuhammad/installing-tensorflow-cuda-cudnn-with-anaconda-for-geforce-gtx-1050-ti-79c1eb94eb7a) which provides thorough information about package dependecies such as `cuda` and `cudnn` versions. 


 As a reference, in our setup with Windows 10 Build 17763 and an NVIDIA Quadro M4000 GPU, we executed the following commands to install `tensorflow`-related packages in a Conda environment with `python=3.7`.
```
conda install -c conda-forge cudnn==7.6.5
conda install -c conda-forge cudatoolkit==10.1.243
 
pip install tensorflow==2.1 
pip install tensorflow-gpu==2.1 
```

### Runtime Benckmarking: CPU vs GPU 
Several runtime benchmarking scripts for iterative hologram generations algorithms are included in a separate foler `pySLM2/runtime_benchmark`. Instructions for running those tests can be found in [`pySLM2/runtime_benchmark/README.md`](pySLM2/runtime_benchmark/README.md).

#### Runtime Benckmarking Example
##### Algorithm performance comparision: Intel Core i9-9900K CPU vs NVidia Quadro M4000 GPU
Our machine has Windows 10 Build 17763 with Intel Core i9-9900K CPU and an NVidia Quadro M4000 GPU. Key package dependencies are:
- `python`: 3.7.1
- `tensorflow`: 2.1.0
- `tensorflow-gpu`: 2.1.0
- `cudnn`: 7.6.5
- `cudatookkit`: 10.1.243

From our testing, we observed:
|  | Case 1 (gs)   | Case 2 (mraf)  | Case 3 (ifta)
|-------------|-------------|-------------|-------------|
| CPU |$225.11 \pm 3.98$ s | $221.64 \pm 1.78$ s  | $206.44 \pm 0.49$ s|
| GPU  | $6.48 \pm 0.29 $ s | $8.29 \pm 0.31$ s |$6.76 \pm 0.50$ s|

These findings show that the iterative algorithms can be greatly accelerated by GPU usage.

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
The DMD from Visitech are communicates with UDP, and therefore no driver is needed to be installed. `pySLM.util` relies on the [Luxbeam](https://pypi.org/project/Luxbeam/) library which can be installed from PyPi:
```
pip install Luxbeam
```

### Notes for LCOS-SLMs

For LCoS-SLMs, most models can be directly controlled via standard monitor connections. To display holograms, one might consider using slmPy. Since pySLM2 calculates hologram values in radians, one will need a conversion table to determine the corresponding grayscale values for output. This conversion typically varies based on the light wavelength and the specific model of the LCoS-SLM.

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



