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

From our testing, we observed:
|  | Case 1 (gs)   | Case 2 (ifta)  |
|-------------|-------------|-------------|
| GPU  | $1.40 \pm 0.26$ s | $13.09 \pm 0.57$ s |
| CPU |$44.09 \pm 0.37$ s | $412.04 \pm 1.00$ s  |

These findings show that the algorithms can be greatly accelerated GPU usage.