#ifndef __BLURHASH_ENCODE_CUH__
#define __BLURHASH_ENCODE_CUH__

#include <stdint.h>
#include <stdlib.h>

const char* encodeCPU(int xComponents, int yComponents, int width, int height, uint8_t* rgb, size_t bytesPerRow);
const char* encodeGPU(int xComponents, int yComponents, int width, int height, uint8_t* rgb, size_t bytesPerRow);

#endif
