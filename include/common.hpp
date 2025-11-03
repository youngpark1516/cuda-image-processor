#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { \
  auto e = (x); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

