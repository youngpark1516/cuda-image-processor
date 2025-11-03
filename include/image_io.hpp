#pragma once
#include <vector>
#include <string>

struct Image {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb; // size = w*h*3
};

Image read_ppm(const char* path);
void   write_ppm(const char* path, const Image& img);

