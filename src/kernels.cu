#include "kernels.cuh"
#include "device_launch_parameters.h"
#include "common.cuh"
#include <stdio.h>

__global__ void computeCosines(int components, int length, float* dev_cosines)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= components * length) return;
	int component = i / length;
	int n = i % length;
	dev_cosines[i] = cosf(M_PI * component * n / length);
}

__global__ void computeFactors(int xComponents, int yComponents, int width, int height, float* dev_xCosines, float* dev_yCosines, float* dev_factors, uint8_t* dev_rgb)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= xComponents * yComponents * width * height) return;
	int xComponent = i % xComponents;
	int yComponent = ((i - xComponent) / xComponents) % yComponents;
	int x = ((i - yComponent * xComponents - xComponent) / (xComponents * yComponents)) % width;
	int y = (i - x * yComponents * xComponents - yComponent * xComponents - xComponent) / (xComponents * yComponents * width);
	
	dev_factors[3 * i + 0] = dev_xCosines[xComponent * width + x] * dev_yCosines[yComponent * height + y] * sRGBToLinear(dev_rgb[3 * x + 0 + y * 3 * width]);
	dev_factors[3 * i + 1] = dev_xCosines[xComponent * width + x] * dev_yCosines[yComponent * height + y] * sRGBToLinear(dev_rgb[3 * x + 1 + y * 3 * width]);
	dev_factors[3 * i + 2] = dev_xCosines[xComponent * width + x] * dev_yCosines[yComponent * height + y] * sRGBToLinear(dev_rgb[3 * x + 2 + y * 3 * width]);
}

__global__ void computeRGB(int xComponents, int yComponents, int width, int height, float* dev_xCosines, float* dev_yCosines, float* dev_rgb, float* dev_colors)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= xComponents * yComponents * width * height) return;
	int x = i % width;
	int y = ((i - x) / width) % height;
	int xComponent = ((i - y * width - x) / (width * height)) % xComponents;
	int yComponent = (i - xComponent * height * width - y * width - x) / (width * height * xComponents);

	dev_rgb[3 * i + 0] = dev_xCosines[xComponent * width + x] * dev_yCosines[yComponent * height + y] * dev_colors[(yComponent * xComponents + xComponent) * 3 + 0];
	dev_rgb[3 * i + 1] = dev_xCosines[xComponent * width + x] * dev_yCosines[yComponent * height + y] * dev_colors[(yComponent * xComponents + xComponent) * 3 + 1];
	dev_rgb[3 * i + 2] = dev_xCosines[xComponent * width + x] * dev_yCosines[yComponent * height + y] * dev_colors[(yComponent * xComponents + xComponent) * 3 + 2];
}

__global__ void reduce(int dim1, int dim2, int size, float* dev_data) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_dim1 = i % dim1;
	int idx_dim2 = ((i - idx_dim1) / dim1) % dim2;
	int ilocal = i / (dim1 * dim2);
	if (ilocal >= size / 2) return;
	int ilocal2 = size - ilocal - 1;
	int i2 = ilocal2 * dim2 * dim1 + idx_dim2 * dim1 + idx_dim1;

	dev_data[3 * i] = dev_data[3 * i] + dev_data[3 * i2];
	dev_data[3 * i + 1] = dev_data[3 * i + 1] + dev_data[3 * i2 + 1];
	dev_data[3 * i + 2] = dev_data[3 * i + 2] + dev_data[3 * i2 + 2];
}