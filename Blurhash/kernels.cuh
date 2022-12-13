#ifndef __BLURHASH_KERNELS_CUH__
#define __BLURHASH_KERNELS_CUH__

#include "cuda_runtime.h"
#include <stdint.h>
#include <stdlib.h>

__global__ void computeCosines(int components, int length, float* dev_cosines);
__global__ void computeFactors(int xComponents, int yComponents, int width, int height, float* dev_xCosines, float* dev_yCosines, float* dev_factors, uint8_t* dev_rgb);
__global__ void computeRGB(int xComponents, int yComponents, int width, int height, float* dev_xCosines, float* dev_yCosines, float* dev_rgb, float* dev_colors);
__global__ void reduce(int dim1, int dim2, int size, float* dev_data);

#endif
