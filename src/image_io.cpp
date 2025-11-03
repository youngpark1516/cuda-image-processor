#include "image_io.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

Image read_ppm(const char* path){
    FILE* f = std::fopen(path, "rb");
    if (!f) { perror("fopen"); std::exit(1); }

    char magic[3] = {0};
    if (std::fscanf(f, "%2s", magic) != 1 || std::string(magic) != "P6") {
        std::fprintf(stderr, "Only P6 PPM supported\n"); std::exit(1);
    }

    int c = std::fgetc(f);
    while (c == '#') { while (c != '\n' && c != EOF) c = std::fgetc(f); c = std::fgetc(f); }
    std::ungetc(c, f);

    int w, h, maxv;
    if (std::fscanf(f, "%d %d %d", &w, &h, &maxv) != 3) {
        std::fprintf(stderr, "Bad PPM header\n"); std::exit(1);
    }
    if (maxv != 255) { std::fprintf(stderr, "Only 8-bit PPM supported\n"); std::exit(1); }
    std::fgetc(f);

    Image img; img.w = w; img.h = h; img.rgb.resize((size_t)w*h*3);
    if (std::fread(img.rgb.data(), 1, img.rgb.size(), f) != img.rgb.size()) {
        std::fprintf(stderr, "PPM truncated\n"); std::exit(1);
    }
    std::fclose(f);
    return img;
}

void write_ppm(const char* path, const Image& img){
    FILE* f = std::fopen(path, "wb");
    if (!f) { perror("fopen"); std::exit(1); }
    std::fprintf(f, "P6\n%d %d\n255\n", img.w, img.h);
    std::fwrite(img.rgb.data(), 1, img.rgb.size(), f);
    std::fclose(f);
}

