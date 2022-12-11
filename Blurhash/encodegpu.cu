#include "encodegpu.cuh"

const char* encodeGPU(int xComponents, int yComponents, int width, int height, uint8_t* rgb, size_t bytesPerRow) {
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	//if (xComponents < 1 || xComponents > 9) return NULL;
	//if (yComponents < 1 || yComponents > 9) return NULL;

	//float* factors = (float*)malloc(xComponents * yComponents * 3 * sizeof(float));
	//memset(factors, 0, sizeof(factors));

	//for (int y = 0; y < yComponents; y++) {
	//	for (int x = 0; x < xComponents; x++) {
	//		float* factor = multiplyBasisFunction(x, y, width, height, rgb, bytesPerRow);
	//		factors[xComponents * 3 * y + 3 * x + 0] = factor[0];
	//		factors[xComponents * 3 * y + 3 * x + 1] = factor[1];
	//		factors[xComponents * 3 * y + 3 * x + 2] = factor[2];
	//	}
	//}

	//float* dc = factors;
	//float* ac = dc + 3;
	//int acCount = xComponents * yComponents - 1;
	//char* ptr = buffer;

	//int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	//ptr = encode_int(sizeFlag, 1, ptr);

	//float maximumValue;
	//if (acCount > 0) {
	//	float actualMaximumValue = 0;
	//	for (int i = 0; i < acCount * 3; i++) {
	//		actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
	//	}

	//	int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
	//	maximumValue = ((float)quantisedMaximumValue + 1) / 166;
	//	ptr = encode_int(quantisedMaximumValue, 1, ptr);
	//}
	//else {
	//	maximumValue = 1;
	//	ptr = encode_int(0, 1, ptr);
	//}

	//ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	//for (int i = 0; i < acCount; i++) {
	//	ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	//}

	//*ptr = 0;

	//free(factors);

	return buffer;
}