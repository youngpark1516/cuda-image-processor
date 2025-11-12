# CUDA Image Processor

A **GPU-accelerated image processing toolkit** built from scratch with **CUDA and C++**.  
Implements kernels for **grayscale conversion**, **box blur**, and **Sobel edge detection** using parallel computation.

---

## Features

| Feature | Description |
|----------|--------------|
| **Grayscale** | Converts RGB to luminance using CUDA threads |
| **Box Blur (3×3)** | Averages neighboring pixels in parallel |
| **Sobel Edge Detection** | Computes image gradients with GPU-optimized Sobel filters |
| **PPM I/O** | Lightweight binary PPM read/write (no OpenCV) |

---

## Project Structure

```
cuda-image-processor/
├── include/
│   ├── common.hpp       # Error handling (CK macro)
│   ├── image_io.hpp
│   ├── gray.hpp
│   ├── boxblur.hpp
│   └── sobel.hpp
│
├── src/
│   ├── image_io.cpp
│   ├── gray.cu
│   ├── boxblur.cu
│   ├── sobel.cu
│   ├── main.cu
│   └── benchmark.cu
│
├── examples/
├── Makefile
└── README.md
```

---

## Build Instructions

### Prerequisites
- **CUDA Toolkit ≥ 12.0**
- **NVIDIA GPU** (e.g., RTX 2080 Ti, 30XX, 40XX series)
- Linux environment (tested on **Rocky 9**)

### Build

```bash
make
```

This produces:
- `bin/imgproc` – main CUDA processor  
- `bin/benchmark` – CPU vs GPU benchmarking utility  

If `nvcc` is not found:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

---

## Usage

Apply any filter directly:

```bash
./bin/imgproc input.ppm output.ppm <mode>
```

**Modes**
- `gray` — grayscale conversion  
- `boxblur` — 3×3 box blur  
- `sobel` — Sobel edge detection  

Example:
```bash
./bin/imgproc examples/reze.ppm examples/reze_sobel.ppm sobel
```

> Supports binary **PPM (P6)** format, 8-bit RGB.

---

## Benchmarking (Optional)

Measure **CPU vs GPU performance** for each filter:

```bash
./bin/benchmark examples/input.ppm 10
```

Output example:

```
Input: 1920x1080, iters=10
grayscale        CPU:  12.41 ms   GPU:  0.21 ms   Speedup:  59.07x
boxblur(3x3)     CPU:  38.27 ms   GPU:  0.52 ms   Speedup:  73.63x
sobel            CPU:  44.91 ms   GPU:  0.61 ms   Speedup:  73.37x
```

---

## Future Work

- Add **Gaussian blur (5×5)** with separable convolution  
- Extend I/O to **PNG/JPEG** via OpenCV backend  
- Integrate performance plots and visual comparisons  

---

## Author

**Chanyoung Park**  
 UC San Diego — Data Science Major  
 [GitHub: youngpark1516](https://github.com/youngpark1516)
