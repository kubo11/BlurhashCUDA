#include "decode.cuh"
#include "common.cuh"
#include "kernels.cuh"
#include "cuda_runtime.h"

uint8_t* decodeCPU(const char* blurhash, int width, int height, int punch, int nChannels) {
	int bytesPerRow = width * nChannels;
	uint8_t* pixelArray = (uint8_t*)malloc(bytesPerRow * height * sizeof(uint8_t));
	if (!isValidBlurhash(blurhash)) return NULL;
	if (punch < 1) punch = 1;

	int sizeFlag = decodeToInt(blurhash, 0, 1);
	int numY = (int)floorf(sizeFlag / 9) + 1;
	int numX = (sizeFlag % 9) + 1;
	int iter = 0;

	float r = 0, g = 0, b = 0;
	int quantizedMaxValue = decodeToInt(blurhash, 1, 2);
	if (quantizedMaxValue == -1) return NULL;

	float maxValue = ((float)(quantizedMaxValue + 1)) / 166;

	int colors_size = numX * numY;
	float* colors = (float*)malloc(colors_size * 3 * sizeof(float));

	for (iter = 0; iter < colors_size; iter++) {
		if (iter == 0) {
			int value = decodeToInt(blurhash, 2, 6);
			if (value == -1) return NULL;
			decodeDC(value, &r, &g, &b);
			colors[iter * 3 + 0] = r;
			colors[iter * 3 + 1] = g;
			colors[iter * 3 + 2] = b;

		}
		else {
			int value = decodeToInt(blurhash, 4 + iter * 2, 6 + iter * 2);
			if (value == -1) return NULL;
			decodeAC(value, maxValue * punch, &r, &g, &b);
			colors[iter * 3 + 0] = r;
			colors[iter * 3 + 1] = g;
			colors[iter * 3 + 2] = b;
		}
	}

	int x = 0, y = 0, i = 0, j = 0;
	int intR = 0, intG = 0, intB = 0;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {

			float r = 0, g = 0, b = 0;

			for (j = 0; j < numY; j++) {
				for (i = 0; i < numX; i++) {
					float basics = cos((M_PI * x * i) / width) * cos((M_PI * y * j) / height);
					int idx = i + j * numX;
					r += colors[idx * 3 + 0] * basics;
					g += colors[idx * 3 + 1] * basics;
					b += colors[idx * 3 + 2] * basics;
				}
			}

			intR = linearTosRGB(r);
			intG = linearTosRGB(g);
			intB = linearTosRGB(b);

			pixelArray[nChannels * x + 0 + y * bytesPerRow] = clampToUByte(intR);
			pixelArray[nChannels * x + 1 + y * bytesPerRow] = clampToUByte(intG);
			pixelArray[nChannels * x + 2 + y * bytesPerRow] = clampToUByte(intB);

			if (nChannels == 4)
				pixelArray[nChannels * x + 3 + y * bytesPerRow] = 255;

		}
	}

	free(colors);

	return pixelArray;
}

uint8_t* decodeGPU(const char* blurhash, int width, int height, int punch, int nChannels) {
	int bytesPerRow = width * nChannels;
	uint8_t* pixelArray = (uint8_t*)malloc(bytesPerRow * height * sizeof(uint8_t));
	if (!isValidBlurhash(blurhash)) return NULL;
	if (punch < 1) punch = 1;

	int sizeFlag = decodeToInt(blurhash, 0, 1);
	int yComponents = (int)floorf(sizeFlag / 9) + 1;
	int xComponents = (sizeFlag % 9) + 1;
	int iter = 0;

	float r = 0, g = 0, b = 0;
	int quantizedMaxValue = decodeToInt(blurhash, 1, 2);
	if (quantizedMaxValue == -1) return NULL;

	float maxValue = ((float)(quantizedMaxValue + 1)) / 166;

	int colors_size = xComponents * yComponents;
	float* colors = (float*)malloc(colors_size * 3 * sizeof(float));

	for (iter = 0; iter < colors_size; iter++) {
		if (iter == 0) {
			int value = decodeToInt(blurhash, 2, 6);
			if (value == -1) return NULL;
			decodeDC(value, &r, &g, &b);
			colors[iter * 3 + 0] = r;
			colors[iter * 3 + 1] = g;
			colors[iter * 3 + 2] = b;

		}
		else {
			int value = decodeToInt(blurhash, 4 + iter * 2, 6 + iter * 2);
			if (value == -1) return NULL;
			decodeAC(value, maxValue * punch, &r, &g, &b);
			colors[iter * 3 + 0] = r;
			colors[iter * 3 + 1] = g;
			colors[iter * 3 + 2] = b;
		}
	}

	float* dev_colors;
	CHECK_CUDA(cudaMalloc(&dev_colors, colors_size * 3 * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(dev_colors, colors, colors_size * 3 * sizeof(float), cudaMemcpyHostToDevice));

	float* dev_rgb, * dev_xCosines, * dev_yCosines;
	CHECK_CUDA(cudaMalloc(&dev_rgb, xComponents * yComponents * width * height * 3 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&dev_xCosines, xComponents * width * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&dev_yCosines, yComponents * height * sizeof(float)));

	int threads = xComponents * width, blocks = 1;
	getThreadLayout(&blocks, &threads);

	computeCosines<<<blocks, threads>>>(xComponents, width, dev_xCosines);
	CHECK_CUDA(cudaDeviceSynchronize());

	threads = yComponents * height;
	blocks = 1;
	getThreadLayout(&blocks, &threads);

	computeCosines<<<blocks, threads>>>(yComponents, height, dev_yCosines);
	CHECK_CUDA(cudaDeviceSynchronize());

	threads = xComponents * yComponents * width * height;
	blocks = 1;
	getThreadLayout(&blocks, &threads);

	computeRGB<<<blocks, threads>>>(xComponents, yComponents, width, height, dev_xCosines, dev_yCosines, dev_rgb, dev_colors);
	CHECK_CUDA(cudaDeviceSynchronize());

	threads = xComponents * yComponents * width * height;
	blocks = 1;
	getThreadLayout(&blocks, &threads);

	int size = xComponents * yComponents;
	while (size > 1) {
		reduce<<<blocks, threads>>>(width, height, size, dev_rgb);
		cudaDeviceSynchronize();
		size = (size - 1) / 2 + 1;
	}

	float* rgb = (float*)malloc(width * height * 3 * sizeof(float));
	CHECK_CUDA(cudaMemcpy(rgb, dev_rgb, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			pixelArray[nChannels * x + 0 + y * bytesPerRow] = clampToUByte(linearTosRGB(rgb[3 * x + 0 + y * 3 * width]));
			pixelArray[nChannels * x + 1 + y * bytesPerRow] = clampToUByte(linearTosRGB(rgb[3 * x + 1 + y * 3 * width]));
			pixelArray[nChannels * x + 2 + y * bytesPerRow] = clampToUByte(linearTosRGB(rgb[3 * x + 2 + y * 3 * width]));

			if (nChannels == 4)
				pixelArray[nChannels * x + 3 + y * bytesPerRow] = 255;
		}
	}

	CHECK_CUDA(cudaFree(dev_rgb));
	CHECK_CUDA(cudaFree(dev_xCosines));
	CHECK_CUDA(cudaFree(dev_yCosines));
	CHECK_CUDA(cudaFree(dev_colors));
	free(colors);
	free(rgb);

	return pixelArray;
}
