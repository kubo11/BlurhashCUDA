#include "encode.cuh"
#include "common.cuh"
#include "kernels.cuh"

#include <string.h>

static float* multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t* rgb, size_t bytesPerRow);

const char* encodeCPU(int xComponents, int yComponents, int width, int height, uint8_t* rgb, size_t bytesPerRow) {
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	if (xComponents < 1 || xComponents > 9) return NULL;
	if (yComponents < 1 || yComponents > 9) return NULL;

	float* factors = (float*)malloc(xComponents * yComponents * 3 * sizeof(float));
	memset(factors, 0, sizeof(factors));

	for (int y = 0; y < yComponents; y++) {
		for (int x = 0; x < xComponents; x++) {
			float* factor = multiplyBasisFunction(x, y, width, height, rgb, bytesPerRow);
			factors[xComponents * 3 * y + 3 * x + 0] = factor[0];
			factors[xComponents * 3 * y + 3 * x + 1] = factor[1];
			factors[xComponents * 3 * y + 3 * x + 2] = factor[2];
		}
	}

	float* dc = factors;
	float* ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char* ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	if (acCount > 0) {
		float actualMaximumValue = 0;
		for (int i = 0; i < acCount * 3; i++) {
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	}
	else {
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}

	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	for (int i = 0; i < acCount; i++) {
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	*ptr = 0;

	free(factors);

	return buffer;
}

static float* multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t* rgb, size_t bytesPerRow) {
	float r = 0, g = 0, b = 0;
	float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float basis = cosf(M_PI * xComponent * x / width) * cosf(M_PI * yComponent * y / height);
			r += basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]);
			g += basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]);
			b += basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]);
		}
	}

	float scale = normalisation / (width * height);

	static float result[3];
	result[0] = r * scale;
	result[1] = g * scale;
	result[2] = b * scale;

	return result;
}

const char* encodeGPU(int xComponents, int yComponents, int width, int height, uint8_t* rgb, size_t bytesPerRow) {
	uint8_t* dev_rgb;
	CHECK_CUDA(cudaMalloc(&dev_rgb, width * height * 3 * sizeof(uint8_t)));
	CHECK_CUDA(cudaMemcpy(dev_rgb, rgb, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));

	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	if (xComponents < 1 || xComponents > 9) return NULL;
	if (yComponents < 1 || yComponents > 9) return NULL;

	float* dev_factors, * dev_xCosines, * dev_yCosines;
	CHECK_CUDA(cudaMalloc(&dev_factors, xComponents * yComponents * width * height * 3 * sizeof(float)));
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

	computeFactors<<<blocks, threads>>>(xComponents, yComponents, width, height, dev_xCosines, dev_yCosines, dev_factors, dev_rgb);
	CHECK_CUDA(cudaDeviceSynchronize());

	threads = xComponents * yComponents * width * height;
	blocks = 1;
	getThreadLayout(&blocks, &threads);

	int size = width * height;
	while (size > 1) {
		reduce<<<blocks, threads>>>(xComponents, yComponents, size, dev_factors);
		cudaDeviceSynchronize();
		size = (size - 1) / 2 + 1;
	}

	float* factors = (float*)malloc(xComponents * yComponents * 3 * sizeof(float));
	CHECK_CUDA(cudaMemcpy(factors, dev_factors, xComponents * yComponents * 3 * sizeof(float), cudaMemcpyDeviceToHost));

	float scale = (float)1 / (float)(width * height);
	for (int i = 0; i < 3; ++i) {
		factors[i] *= scale;
	}
	scale = (float)2 / (float)(width * height);
	for (int i = 3; i < xComponents * yComponents * 3; ++i) {
		factors[i] *= scale;
	}

	float* dc = factors;
	float* ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char* ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	if (acCount > 0) {
		float actualMaximumValue = 0;
		for (int i = 0; i < acCount * 3; i++) {
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	}
	else {
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}

	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	for (int i = 0; i < acCount; i++) {
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	*ptr = 0;

	CHECK_CUDA(cudaFree(dev_rgb));
	CHECK_CUDA(cudaFree(dev_factors));
	CHECK_CUDA(cudaFree(dev_xCosines));
	CHECK_CUDA(cudaFree(dev_yCosines));
	free(factors);

	return buffer;
}
