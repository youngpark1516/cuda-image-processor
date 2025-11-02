# CUDA Image Processor

A minimal GPU-accelerated **image processor** written in CUDA.  
It performs common 2D filters — **grayscale**, **box blur**, and **Sobel edge detection** — directly on the GPU using parallel kernels.

---

## Features

- **PPM image input/output** (simple, dependency-free)
- **3 CUDA filters**:
  - `gray` – converts RGB to grayscale  
  - `boxblur` – applies a 3×3 blur  
  - `sobel` – performs edge detection using Sobel operators
- Works entirely on the GPU  
- Fully self-contained (no OpenCV or external libs required)
- Clean and modular C++/CUDA implementation

---

## Build

### Requirements
- CUDA Toolkit ≥ 12.0  
- NVIDIA GPU (e.g., RTX 20XX, 30XX, etc.)

### Compile

```bash
nvcc -O3 -arch=sm_75 src/imgproc.cu -o imgproc

