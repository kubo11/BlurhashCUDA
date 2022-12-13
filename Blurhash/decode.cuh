#ifndef __BLURHASH_DECODE_CUH__
#define __BLURHASH_DECODE_CUH__

#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

uint8_t* decodeCPU(const char * blurhash, int width, int height, int punch, int nChannels);
uint8_t* decodeGPU(const char* blurhash, int width, int height, int punch, int nChannels);

#endif
