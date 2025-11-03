#include "common.hpp"
#include "boxblur.hpp"
#include <cuda_runtime.h>

__global__ void k_boxblur(const unsigned char* in, unsigned char* out, int w, int h){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    int sum=0, cnt=0;
    for (int j=-1;j<=1;++j){
        for (int i=-1;i<=1;++i){
            int xx = min(w-1, max(0, x+i));
            int yy = min(h-1, max(0, y+j));
            sum += in[yy*w + xx];
            cnt++;
        }
    }
    out[y*w + x] = (unsigned char)(sum / cnt);
}

void launch_boxblur(const unsigned char* d_in_gray, unsigned char* d_out_gray, int w, int h){
    dim3 block(16,16), grid((w+15)/16, (h+15)/16);
    k_boxblur<<<grid, block>>>(d_in_gray, d_out_gray, w, h);
    CK(cudaDeviceSynchronize());
}

