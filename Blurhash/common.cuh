#ifndef __BLURHASH_COMMON_H__
#define __BLURHASH_COMMON_H__

#include<math.h>
#include<stdio.h>
#include<string.h>
#include"cuda_runtime.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK_CUDA(call)                                                         \
{                                                                                \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess)                                                       \
    {                                                                            \
        printf("CUDA Runtime failure '#%d' at %s:%d\n", _e, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
}

static char characters[84] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

static inline int linearTosRGB(float value) {
	float v = fmaxf(0, fminf(1, value));
	if(v <= 0.0031308) return v * 12.92 * 255 + 0.5;
	else return (1.055 * powf(v, 1 / 2.4) - 0.055) * 255 + 0.5;
}

__host__ __device__ static inline float sRGBToLinear(int value) {
	float v = (float)value / 255;
	if(v <= 0.04045) return v / 12.92;
	else return powf((v + 0.055) / 1.055, 2.4);
}

static inline float signPow(float value, float exp) {
	return copysignf(powf(fabsf(value), exp), value);
}

static inline int encodeDC(float r, float g, float b) {
	int roundedR = linearTosRGB(r);
	int roundedG = linearTosRGB(g);
	int roundedB = linearTosRGB(b);
	return (roundedR << 16) + (roundedG << 8) + roundedB;
}

static inline int encodeAC(float r, float g, float b, float maximumValue) {
	int quantR = fmaxf(0, fminf(18, floorf(signPow(r / maximumValue, 0.5) * 9 + 9.5)));
	int quantG = fmaxf(0, fminf(18, floorf(signPow(g / maximumValue, 0.5) * 9 + 9.5)));
	int quantB = fmaxf(0, fminf(18, floorf(signPow(b / maximumValue, 0.5) * 9 + 9.5)));

	return quantR * 19 * 19 + quantG * 19 + quantB;
}

static inline char* encode_int(int value, int length, char* destination) {
	int divisor = 1;
	for (int i = 0; i < length - 1; i++) divisor *= 83;

	for (int i = 0; i < length; i++) {
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}

static inline uint8_t clampToUByte(int src) {
	if (src >= 0 && src <= 255)
		return src;
	return (src < 0) ? 0 : 255;
}

static inline int decodeToInt(const char* string, int start, int end) {
	int value = 0, iter1 = 0, iter2 = 0;
	for (iter1 = start; iter1 < end; iter1++) {
		int index = -1;
		for (iter2 = 0; iter2 < 83; iter2++) {
			if (characters[iter2] == string[iter1]) {
				index = iter2;
				break;
			}
		}
		if (index == -1) return -1;
		value = value * 83 + index;
	}
	return value;
}

static inline bool inline isValidBlurhash(const char* blurhash) {

	const int hashLength = strlen(blurhash);

	if (!blurhash || strlen(blurhash) < 6) return false;

	int sizeFlag = decodeToInt(blurhash, 0, 1);	//Get size from first character
	int numY = (int)floorf(sizeFlag / 9) + 1;
	int numX = (sizeFlag % 9) + 1;

	if (hashLength != 4 + 2 * numX * numY) return false;
	return true;
}

static inline void decodeDC(int value, float* r, float* g, float* b) {
	*r = sRGBToLinear(value >> 16); 	// R-component
	*g = sRGBToLinear((value >> 8) & 255); // G-Component
	*b = sRGBToLinear(value & 255);	// B-Component
}

static inline void decodeAC(int value, float maximumValue, float* r, float* g, float* b) {
	int quantR = (int)floorf(value / (19 * 19));
	int quantG = (int)floorf(value / 19) % 19;
	int quantB = (int)value % 19;

	*r = signPow(((float)quantR - 9) / 9, 2.0) * maximumValue;
	*g = signPow(((float)quantG - 9) / 9, 2.0) * maximumValue;
	*b = signPow(((float)quantB - 9) / 9, 2.0) * maximumValue;
}

#endif
