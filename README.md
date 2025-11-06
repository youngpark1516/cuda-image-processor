# CUDA Image Processor

A modular CUDA-based image processing toolkit implementing GPU-accelerated filters such as **grayscale**, **box blur**, and **Sobel**.  

---

## Overview

### Implemented Features
- **Grayscale conversion**
- **Box blur** (3×3)
- **Sobel**

Each kernel runs entirely on the GPU using `nvcc` and minimal host-device transfers for efficiency.

---

## Project Structure

```
cuda-image-processor/
├── include/
│   ├── common.hpp      # CUDA error macro
│   ├── image_io.hpp    # PPM read & write
│   ├── gray.hpp        # Grayscale kernel
│   ├── boxblur.hpp     # Box blur kernel
│   └── sobel.hpp       # Sobel edge kernel
│
├── src/
│   ├── image_io.cpp
│   ├── gray.cu
│   ├── boxblur.cu
│   ├── sobel.cu
│   └── main.cu
│
├── Makefile
├── examples             # Example input/output images
└── bin/
    └── imgproc         # Final executable
```

---

## Build Instructions

```bash
make
```

This will compile all `.cu` and `.cpp` files into `bin/imgproc`.

If `nvcc` isn’t found, run:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

---

## Usage

```bash
./bin/imgproc input.ppm output.ppm <mode>
```

Available modes:
- `gray` — Convert image to grayscale  
- `boxblur` — Apply 3×3 box blur (on grayscale)  
- `sobel` — Apply Sobel edge detection (on grayscale)

Example:
```bash
./bin/imgproc examples/reze.ppm examples/reze_sobel.ppm sobel
```

> Input images must be binary PPM (P6) format, 8-bit RGB.

---

## Future Work

- Add Gaussian blur (5×5 separable convolution)  
- Extend support for PNG/JPEG via OpenCV (while keeping CUDA backend)  
- Add benchmarking utilities with CUDA events  

---

##  Author

**Chanyoung Park**  
UC San Diego · Data Science Major  
GitHub: [youngpark1516](https://github.com/youngpark1516)
