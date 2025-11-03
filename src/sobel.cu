#include "common.hpp"
#include "sobel.hpp"
#include <cuda_runtime.h>
#include <cmath>

__global__ void k_sobel(const unsigned char* in, unsigned char* out, int w, int h){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;

    int xm1 = max(0, x-1), xp1 = min(w-1, x+1);
    int ym1 = max(0, y-1), yp1 = min(h-1, y+1);

    int p00 = in[ym1*w + xm1], p01 = in[ym1*w + x],   p02 = in[ym1*w + xp1];
    int p10 = in[y   *w + xm1], p11 = in[y   *w + x], p12 = in[y   *w + xp1];
    int p20 = in[yp1*w + xm1], p21 = in[yp1*w + x],  p22 = in[yp1*w + xp1];

    int gx = -p00 - 2*p10 - p20 + p02 + 2*p12 + p22;
    int gy =  p00 + 2*p01 + p02 - p20 - 2*p21 - p22;
    int mag = (int)sqrtf((float)(gx*gx + gy*gy));
    if (mag > 255) mag = 255;
    out[y*w + x] = (unsigned char)mag;
}

void launch_sobel(const unsigned char* d_in_gray, unsigned char* d_out_gray, int w, int h){
    dim3 block(16,16), grid((w+15)/16, (h+15)/16);
    k_sobel<<<grid, block>>>(d_in_gray, d_out_gray, w, h);
    CK(cudaDeviceSynchronize());
}

