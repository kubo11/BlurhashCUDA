#include "encodegpu.cuh"
#include "common.cuh"
#include "kernels.cuh"

#include <string.h>

const char* encodeGPU(int xComponents, int yComponents, int width, int height, uint8_t* rgb, size_t bytesPerRow) {
	uint8_t* dev_rgb;
	CHECK_CUDA(cudaMalloc(&dev_rgb, width * height * 3 * sizeof(uint8_t)));
	CHECK_CUDA(cudaMemcpy(dev_rgb, rgb, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));

	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	if (xComponents < 1 || xComponents > 9) return NULL;
	if (yComponents < 1 || yComponents > 9) return NULL;

	float* dev_factors, *dev_xCosines, *dev_yCosines;
	CHECK_CUDA(cudaMalloc(&dev_factors, xComponents * yComponents * width * height * 3 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&dev_xCosines, xComponents * width * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&dev_yCosines, yComponents * height * sizeof(float)));

	int threads = xComponents * width, blocks = 1;
	if (threads >= 1024) {
		blocks = (threads - 1) / 1024 + 1;
		threads = 1024;
	}

	computeCosines<<<blocks, threads>>>(xComponents, width, dev_xCosines);
	CHECK_CUDA(cudaDeviceSynchronize());

	threads = yComponents * height;
	blocks = 1;
	if (threads >= 1024) {
		blocks = (threads - 1) / 1024 + 1;
		threads = 1024;
	}

	computeCosines<<<blocks, threads>>>(yComponents, height, dev_yCosines);
	CHECK_CUDA(cudaDeviceSynchronize());

	threads = xComponents * yComponents * width * height;
	blocks = 1;
	if (threads >= 1024) {
		blocks = (threads - 1) / 1024 + 1;
		threads = 1024;
	}

	computeFactors<<<blocks, threads>>>(xComponents, yComponents, width, height, dev_xCosines, dev_yCosines, dev_factors, dev_rgb);
	CHECK_CUDA(cudaDeviceSynchronize());

	threads = xComponents * yComponents * width * height;
	blocks = 1;
	if (threads >= 1024) {
		blocks = (threads - 1) / 1024 + 1;
		threads = 1024;
	}

	int size = width * height;
	while (size > 1) {
		reduceFactors<<<blocks, threads>>>(xComponents, yComponents, width * height, size, dev_factors);
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