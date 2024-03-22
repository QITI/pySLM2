pySLM2 GPU Acceleration via Tensorflow
======================================
The `runtime_benchmark` folder contains tests for users to run on their machines to assess GPU availability and performance. To check for GPU availability and CUDA compatibility, execute the `check_gpu.py` file.

Runtime Benchmarking Cases
--------------------------
At users' convinience, we separated out CPU/GPU-only tests for two different yet representative hologram generation cases.
1) on a SLM using Gerchberg-Saxton (GS) method with 200 iterations, and 
2) on a DMD using IFTA with 2000 iterations.

### Test Results -- Example
Our setup uses Windows 10 Build 17763 and an NVidia Quadro M4000 GPU. Key package dependencies are:
- `python`: 3.7.1
- `tensorflow`: 2.1.0
- `tensorflow-gpu`: 2.1.0
- `cudnn`: 7.6.5
- `cudatookkit`: 10.1.243

For an example, results for case (1) are displayed as follows:
```
(env) [path]\runtime_benchmark>python slm_hologram_gs_cpu.py 2>NUL
Num GPUs Available: 0
No GPU found. Running on CPU.
time used: 44.43152475357056

(env) [path]\runtime_benchmark>python slm_hologram_gs_gpu.py 2>NUL
Num GPUs Available: 1
Is Built with CUDA: True
time used: 2.934244155883789
```
<!-- ```
(env) [path]\runtime_benchmark>python runtime_ifta_cpu.py 2>NUL
Num GPUs Available: 0
No GPU found. Running on CPU.
Running on CPU
time used: 420.37334179878235

(pySLM2) [path]\runtime_benchmark>python runtime_ifta_gpu.py 2>NUL
Num GPUs Available: 1
Is Built with CUDA: True
time used: 20.420279026031494
``` -->

From our testing, we observed:
- Case 1: Approximately 45 seconds execution time using only the CPU, and 3 seconds with one GPU.
- Case 2: Around 7 minutes on CPU only, and 20 seconds with one GPU.

These findings indicate that the algorithms are accelerated by about 20 times when using the GPU with our system specifications.


Notes for Tensorflow Installation
---------------------------------
The exact package dependencies vary depending on each system and the GPU card. Please refer to the TensorFlow website for step-by-step instructions, and consult the NVIDIA website to check the CUDA compatibility of your GPU. 

 As a reference, in our setup with Windows 10 Build 17763 and an NVIDIA Quadro M4000 GPU, we executed the following commands to install TensorFlow-related packages in a Conda environment with `python` version 3.7.
```
conda install -c conda-forge cudnn==7.6.5
conda install -c conda-forge cudatoolkit=10.1.243
 
pip install tensorflow==2.1 
pip install tensorflow-gpu==2.1 
```