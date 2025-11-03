#pragma once
#include <cstddef>

void launch_to_gray(const unsigned char* d_in, unsigned char* d_gray, int w, int h);
void launch_gray_to_rgb(const unsigned char* d_gray, unsigned char* d_rgb, int w, int h);

