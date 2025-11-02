#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define CK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// =========================
//  PPM I/O (binary P6)
// =========================
struct Image {
    int w, h;
    std::vector<unsigned char> rgb; // size = w*h*3
};

static Image read_ppm(const char* path){
    FILE* f = std::fopen(path, "rb");
    if(!f){ perror("fopen"); std::exit(1); }
    char magic[3]={0};
    if (std::fscanf(f, "%2s", magic) != 1 || std::string(magic)!="P6") {
        fprintf(stderr, "Only P6 PPM supported\n");
        std::exit(1);
    }
    int w,h,maxv;
    // skip comments
    int c = std::fgetc(f);
    while(c=='#'){ while(c!='\n' && c!=EOF) c=std::fgetc(f); c=std::fgetc(f); }
    std::ungetc(c,f);
    if (std::fscanf(f, "%d %d %d", &w, &h, &maxv) != 3) {
        fprintf(stderr, "Bad PPM header\n"); std::exit(1);
    }
    if (maxv != 255) {
        fprintf(stderr, "Only 8-bit PPM supported\n"); std::exit(1);
    }
    fgetc(f); // consume single whitespace
    Image img;
    img.w = w; img.h = h;
    img.rgb.resize(w*h*3);
    if (std::fread(img.rgb.data(), 1, img.rgb.size(), f) != img.rgb.size()) {
        fprintf(stderr, "PPM truncated\n"); std::exit(1);
    }
    std::fclose(f);
    return img;
}

static void write_ppm(const char* path, const Image& img){
    FILE* f = std::fopen(path, "wb");
    if(!f){ perror("fopen"); std::exit(1); }
    std::fprintf(f, "P6\n%d %d\n255\n", img.w, img.h);
    std::fwrite(img.rgb.data(), 1, img.rgb.size(), f);
    std::fclose(f);
}


//  CUDA kernels

// 1) RGB → grayscale
__global__ void k_to_gray(const unsigned char* in, unsigned char* out, int w, int h){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    int idx = (y*w + x)*3;
    float r = in[idx+0];
    float g = in[idx+1];
    float b = in[idx+2];
    unsigned char g8 = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);
    out[y*w + x] = g8;
}

// 2) box blur on grayscale
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

// 3) Sobel edge on grayscale
__global__ void k_sobel(const unsigned char* in, unsigned char* out, int w, int h){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    int xm1 = max(0, x-1), xp1 = min(w-1, x+1);
    int ym1 = max(0, y-1), yp1 = min(h-1, y+1);

    int p00 = in[ym1*w + xm1];
    int p01 = in[ym1*w + x  ];
    int p02 = in[ym1*w + xp1];
    int p10 = in[y   *w + xm1];
    int p11 = in[y   *w + x  ];
    int p12 = in[y   *w + xp1];
    int p20 = in[yp1*w + xm1];
    int p21 = in[yp1*w + x  ];
    int p22 = in[yp1*w + xp1];

    int gx = -p00 - 2*p10 - p20 + p02 + 2*p12 + p22;
    int gy =  p00 + 2*p01 + p02 - p20 - 2*p21 - p22;
    int mag = (int)sqrtf((float)(gx*gx + gy*gy));
    if (mag > 255) mag = 255;
    out[y*w + x] = (unsigned char)mag;
}

// write grayscale back to RGB (for PPM)
__global__ void k_gray_to_rgb(const unsigned char* g, unsigned char* rgb, int w, int h){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    int idxg = y*w + x;
    int idx3 = idxg*3;
    unsigned char v = g[idxg];
    rgb[idx3+0] = v;
    rgb[idx3+1] = v;
    rgb[idx3+2] = v;
}


//  main
int main(int argc, char** argv){
    if (argc < 4){
        std::fprintf(stderr, "usage: %s input.ppm output.ppm [gray|boxblur|sobel]\n", argv[0]);
        return 1;
    }
    std::string inpath = argv[1];
    std::string outpath = argv[2];
    std::string mode = argv[3];

    Image img = read_ppm(inpath.c_str());
    int W = img.w, H = img.h;

    // device buffers
    unsigned char *d_rgb_in=nullptr, *d_gray=nullptr, *d_tmp=nullptr, *d_rgb_out=nullptr;
    CK(cudaMalloc(&d_rgb_in,  W*H*3));
    CK(cudaMalloc(&d_gray,    W*H));
    CK(cudaMalloc(&d_tmp,     W*H));
    CK(cudaMalloc(&d_rgb_out, W*H*3));

    CK(cudaMemcpy(d_rgb_in, img.rgb.data(), W*H*3, cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid( (W+15)/16, (H+15)/16 );

    // always convert to gray first
    k_to_gray<<<grid,block>>>(d_rgb_in, d_gray, W, H);
    CK(cudaDeviceSynchronize());

    if (mode == "gray"){
        // just write gray → rgb
        k_gray_to_rgb<<<grid,block>>>(d_gray, d_rgb_out, W, H);
        CK(cudaDeviceSynchronize());
    } else if (mode == "boxblur"){
        k_boxblur<<<grid,block>>>(d_gray, d_tmp, W, H);
        CK(cudaDeviceSynchronize());
        k_gray_to_rgb<<<grid,block>>>(d_tmp, d_rgb_out, W, H);
        CK(cudaDeviceSynchronize());
    } else if (mode == "sobel"){
        k_sobel<<<grid,block>>>(d_gray, d_tmp, W, H);
        CK(cudaDeviceSynchronize());
        k_gray_to_rgb<<<grid,block>>>(d_tmp, d_rgb_out, W, H);
        CK(cudaDeviceSynchronize());
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return 1;
    }

    // copy back
    Image out; out.w = W; out.h = H; out.rgb.resize(W*H*3);
    CK(cudaMemcpy(out.rgb.data(), d_rgb_out, W*H*3, cudaMemcpyDeviceToHost));

    write_ppm(outpath.c_str(), out);

    cudaFree(d_rgb_in);
    cudaFree(d_gray);
    cudaFree(d_tmp);
    cudaFree(d_rgb_out);

    return 0;
}

