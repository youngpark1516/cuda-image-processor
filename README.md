# CUDA Image Processor

A minimal GPU-accelerated **image processor** written in CUDA.  
It performs common 2D filters - **grayscale**, **box blur**, and **Sobel edge detection** - directly on the GPU using parallel kernels.

---

## Features

- **PPM image input/output** (simple, dependency-free)  
- **3 CUDA filters**:
  - `gray` - converts RGB to grayscale  
  - `boxblur` - applies a 3×3 blur  
  - `sobel` - performs edge detection using Sobel operators  
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
make
```

This produces:
- `bin/imgproc` - the main image processor  
- `bin/benchmark` - CPU vs GPU benchmarking tool  

---

## Benchmarking

A custom benchmarking tool is included to compare **CPU vs GPU performance**:

- Grayscale  
- Box blur (3×3)  
- Sobel edge detection  

### Run benchmark
```bash
bin/benchmark examples/input.ppm 10
```

Where `10` is the number of iterations averaged.

### Sample output
```
Input: 1920x1080, iters=10
grayscale        CPU:  12.41 ms   GPU:  0.21 ms   Speedup:  59.07x
boxblur(3x3)     CPU:  38.27 ms   GPU:  0.52 ms   Speedup:  73.63x
sobel            CPU:  44.91 ms   GPU:  0.61 ms   Speedup:  73.37x
```

---

## Usage

### Apply a filter
```bash
bin/imgproc input.ppm output.ppm gray
bin/imgproc input.ppm output.ppm boxblur
bin/imgproc input.ppm output.ppm sobel
```

---

## Repository Structure
```
image-processor/
├── src/
│   ├── gray.cu
│   ├── boxblur.cu
│   ├── sobel.cu
│   ├── main.cu
│   ├── benchmark.cu
│   └── image_io.cpp
├── include/
│   ├── gray.hpp
│   ├── boxblur.hpp
│   ├── sobel.hpp
│   ├── common.hpp
│   └── image_io.hpp
├── examples/
│   └── example.ppm
├── bin/
├── Makefile
└── README.md
```

---

## License
MIT License
