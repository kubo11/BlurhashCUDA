#ifndef __BLURHASH_DECODE_GPU_H_
#define __BLURHASH_DECODE_GPU_H_

#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

uint8_t* decodeGPU(const char* blurhash, int width, int height, int punch, int nChannels);

#endif // !__BLURHASH_DECODE_GPU_H_
