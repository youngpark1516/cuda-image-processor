#include "common.hpp"
#include "image_io.hpp"
#include "gray.hpp"
#include "boxblur.hpp"
#include "sobel.hpp"
#include <string>

int main(int argc, char** argv){
    if (argc < 4){
        std::fprintf(stderr, "usage: %s input.ppm output.ppm [gray|boxblur|sobel]\n", argv[0]);
        return 1;
    }
    std::string inpath = argv[1], outpath = argv[2], mode = argv[3];

    Image img = read_ppm(inpath.c_str());
    int W = img.w, H = img.h;

    unsigned char *d_rgb_in=nullptr, *d_gray=nullptr, *d_tmp=nullptr, *d_rgb_out=nullptr;
    CK(cudaMalloc(&d_rgb_in,  (size_t)W*H*3));
    CK(cudaMalloc(&d_gray,    (size_t)W*H));
    CK(cudaMalloc(&d_tmp,     (size_t)W*H));
    CK(cudaMalloc(&d_rgb_out, (size_t)W*H*3));

    CK(cudaMemcpy(d_rgb_in, img.rgb.data(), (size_t)W*H*3, cudaMemcpyHostToDevice));

    launch_to_gray(d_rgb_in, d_gray, W, H);

    if (mode == "gray"){
        launch_gray_to_rgb(d_gray, d_rgb_out, W, H);
    } else if (mode == "boxblur"){
        launch_boxblur(d_gray, d_tmp, W, H);
        launch_gray_to_rgb(d_tmp, d_rgb_out, W, H);
    } else if (mode == "sobel"){
        launch_sobel(d_gray, d_tmp, W, H);
        launch_gray_to_rgb(d_tmp, d_rgb_out, W, H);
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return 1;
    }

    Image out; out.w = W; out.h = H; out.rgb.resize((size_t)W*H*3);
    CK(cudaMemcpy(out.rgb.data(), d_rgb_out, (size_t)W*H*3, cudaMemcpyDeviceToHost));
    write_ppm(outpath.c_str(), out);

    cudaFree(d_rgb_in); cudaFree(d_gray); cudaFree(d_tmp); cudaFree(d_rgb_out);
    return 0;
}

