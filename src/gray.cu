#include "common.hpp"
#include "gray.hpp"
#include <cuda_runtime.h>

__global__ void k_to_gray(const unsigned char* in, unsigned char* out, int w, int h){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    int idx = (y*w + x)*3;
    float r = in[idx+0], g = in[idx+1], b = in[idx+2];
    out[y*w + x] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);
}

__global__ void k_gray_to_rgb(const unsigned char* g, unsigned char* rgb, int w, int h){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    int i = y*w + x, o = i*3;
    unsigned char v = g[i];
    rgb[o+0] = v; rgb[o+1] = v; rgb[o+2] = v;
}

void launch_to_gray(const unsigned char* d_in, unsigned char* d_gray, int w, int h){
    dim3 block(16,16), grid((w+15)/16, (h+15)/16);
    k_to_gray<<<grid, block>>>(d_in, d_gray, w, h);
    CK(cudaDeviceSynchronize());
}

void launch_gray_to_rgb(const unsigned char* d_gray, unsigned char* d_rgb, int w, int h){
    dim3 block(16,16), grid((w+15)/16, (h+15)/16);
    k_gray_to_rgb<<<grid, block>>>(d_gray, d_rgb, w, h);
    CK(cudaDeviceSynchronize());
}

