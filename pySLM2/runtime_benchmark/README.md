pySLM2 GPU Acceleration via Tensorflow
======================================
The `runtime_benchmark` folder contains tests for users to run on their machines to assess GPU availability and performance. To check for GPU availability and CUDA compatibility, run the `check_gpu.py` file by executing
```
python check_gpu.py
```

Runtime Benchmarking Cases
--------------------------
At users' convinience, we separated out CPU/GPU-only tests for 3 iterative hologram generation cases, all with 1000 iterations.
1) on a SLM using Gerchberg-Saxton (gs) algorithm 
2) on a SLM using Modulated Residual Amplitude Fourier (mraf) algorithm 
3) on a DMD using Iterative Fourier Transform algorithm

As an example, run
```
python slm_gs_gpu.py
```
to test Case (1) using gpu, if detected any.

### Test Results -- Example
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
