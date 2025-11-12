// src/benchmark.cu
#include "common.hpp"
#include "image_io.hpp"
#include "gray.hpp"
#include "boxblur.hpp"
#include "sobel.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>

static void cpu_to_gray(const Image& img, std::vector<unsigned char>& gray) {
    const int W = img.w, H = img.h;
    gray.resize((size_t)W * H);
    const unsigned char* rgb = img.rgb.data();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i3 = (y*W + x) * 3;
            float r = rgb[i3+0], g = rgb[i3+1], b = rgb[i3+2];
            gray[(size_t)y*W + x] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);
        }
    }
}

static void cpu_boxblur3x3(const std::vector<unsigned char>& in, std::vector<unsigned char>& out, int W, int H) {
    out.resize((size_t)W * H);
    auto at = [&](int yy, int xx) -> unsigned char {
        if (xx < 0) xx = 0; if (xx >= W) xx = W-1;
        if (yy < 0) yy = 0; if (yy >= H) yy = H-1;
        return in[(size_t)yy * W + xx];
    };
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int s = 0;
            for (int j=-1;j<=1;++j)
                for (int i=-1;i<=1;++i)
                    s += at(y+j, x+i);
            out[(size_t)y * W + x] = (unsigned char)(s / 9);
        }
    }
}

static void cpu_sobel(const std::vector<unsigned char>& in, std::vector<unsigned char>& out, int W, int H) {
    out.resize((size_t)W * H);
    auto at = [&](int yy, int xx) -> int {
        if (xx < 0) xx = 0; if (xx >= W) xx = W-1;
        if (yy < 0) yy = 0; if (yy >= H) yy = H-1;
        return (int)in[(size_t)yy * W + xx];
    };
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int p00 = at(y-1,x-1), p01 = at(y-1,x),   p02 = at(y-1,x+1);
            int p10 = at(y  ,x-1), p11 = at(y  ,x),   p12 = at(y  ,x+1); 
            int p20 = at(y+1,x-1), p21 = at(y+1,x),   p22 = at(y+1,x+1);
            int gx = -p00 - 2*p10 - p20 + p02 + 2*p12 + p22;
            int gy =  p00 + 2*p01 + p02 - p20 - 2*p21 - p22;
            int mag = (int)std::lround(std::sqrt((float)(gx*gx + gy*gy)));
            if (mag > 255) mag = 255;
            out[(size_t)y * W + x] = (unsigned char)mag;
        }
    }
}

static float time_cpu_ms(std::function<void()> fn, int iters) {
    fn(); // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count() / std::max(1, iters);
}

static float time_gpu_ms(std::function<void()> fn, int iters) {
    fn(); // warmup
    cudaEvent_t ev_start, ev_end;
    CK(cudaEventCreate(&ev_start));
    CK(cudaEventCreate(&ev_end));
    CK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; ++i) fn();
    CK(cudaEventRecord(ev_end));
    CK(cudaEventSynchronize(ev_end));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, ev_start, ev_end));
    CK(cudaEventDestroy(ev_start));
    CK(cudaEventDestroy(ev_end));
    return ms / std::max(1, iters);
}

static void print_row(const char* name, float cpu_ms, float gpu_ms) {
    double speedup = (gpu_ms > 0.0f) ? (cpu_ms / gpu_ms) : 0.0;
    std::printf("%-16s  CPU: %8.3f ms   GPU: %8.3f ms   Speedup: %6.2fx\n",
                name, cpu_ms, gpu_ms, speedup);
}


static void bench_gray(const Image& img, int iters) {
    const int W = img.w, H = img.h;

    std::vector<unsigned char> gray_cpu;
    float cpu_ms = time_cpu_ms([&]{ cpu_to_gray(img, gray_cpu); }, iters);

    unsigned char *d_in=nullptr, *d_gray=nullptr;
    CK(cudaMalloc(&d_in,  (size_t)W*H*3));
    CK(cudaMalloc(&d_gray,(size_t)W*H));
    CK(cudaMemcpy(d_in, img.rgb.data(), (size_t)W*H*3, cudaMemcpyHostToDevice));

    float gpu_ms = time_gpu_ms([&]{ launch_to_gray(d_in, d_gray, W, H); }, iters);

    print_row("grayscale", cpu_ms, gpu_ms);
    cudaFree(d_in); cudaFree(d_gray);
}

static void bench_boxblur(const Image& img, int iters) {
    const int W = img.w, H = img.h;

    std::vector<unsigned char> gray_cpu, blur_cpu;
    float cpu_ms = time_cpu_ms([&]{
        cpu_to_gray(img, gray_cpu);
        cpu_boxblur3x3(gray_cpu, blur_cpu, W, H);
    }, iters);

    unsigned char *d_in=nullptr, *d_gray=nullptr, *d_tmp=nullptr;
    CK(cudaMalloc(&d_in,  (size_t)W*H*3));
    CK(cudaMalloc(&d_gray,(size_t)W*H));
    CK(cudaMalloc(&d_tmp, (size_t)W*H));
    CK(cudaMemcpy(d_in, img.rgb.data(), (size_t)W*H*3, cudaMemcpyHostToDevice));

    float gpu_ms = time_gpu_ms([&]{
        launch_to_gray(d_in, d_gray, W, H);
        launch_boxblur(d_gray, d_tmp, W, H);
    }, iters);

    print_row("boxblur(3x3)", cpu_ms, gpu_ms);
    cudaFree(d_in); cudaFree(d_gray); cudaFree(d_tmp);
}

static void bench_sobel(const Image& img, int iters) {
    const int W = img.w, H = img.h;

    std::vector<unsigned char> gray_cpu, sobel_cpu;
    float cpu_ms = time_cpu_ms([&]{
        cpu_to_gray(img, gray_cpu);
        cpu_sobel(gray_cpu, sobel_cpu, W, H);
    }, iters);

    unsigned char *d_in=nullptr, *d_gray=nullptr, *d_tmp=nullptr;
    CK(cudaMalloc(&d_in,  (size_t)W*H*3));
    CK(cudaMalloc(&d_gray,(size_t)W*H));
    CK(cudaMalloc(&d_tmp, (size_t)W*H));
    CK(cudaMemcpy(d_in, img.rgb.data(), (size_t)W*H*3, cudaMemcpyHostToDevice));

    float gpu_ms = time_gpu_ms([&]{
        launch_to_gray(d_in, d_gray, W, H);
        launch_sobel(d_gray, d_tmp, W, H);
    }, iters);

    print_row("sobel", cpu_ms, gpu_ms);
    cudaFree(d_in); cudaFree(d_gray); cudaFree(d_tmp);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <input.ppm> [iters]\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int iters = (argc >= 3) ? std::max(1, std::atoi(argv[2])) : 10;

    Image img = read_ppm(path);

    std::printf("Input: %dx%d, iters=%d\n", img.w, img.h, iters);
    bench_gray(img, iters);
    bench_boxblur(img, iters);
    bench_sobel(img, iters);
    return 0;
}
