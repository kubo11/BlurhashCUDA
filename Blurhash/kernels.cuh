#ifndef __BLURHASH_KERNELS_H__
#define __BLURHASH_KERNELS_H__

#include "cuda_runtime.h"
#include <stdint.h>
#include <stdlib.h>

__global__ void computeCosines(int components, int length, float* dev_cosines);
__global__ void computeFactors(int xComponents, int yComponents, int width, int height, float* dev_xCosines, float* dev_yCosines, float* dev_factors, uint8_t* dev_rgb);
__global__ void reduceFactors(int xComponents, int yComponents, int imageSize, int size, float* dev_factors);
__global__ void computeRGB(int xComponents, int yComponents, int width, int height, float* dev_xCosines, float* dev_yCosines, float* dev_rgb, float* dev_colors);
__global__ void reduceRGB(int width, int height, int components, int size, float* dev_factors);

#endif // !__BLURHASH_KERNELS_H__
