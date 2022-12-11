#ifndef __BLURHASH_ENCODE_GPU_H_
#define __BLURHASH_ENCODE_GPU_H_

#include <stdint.h>
#include <stdlib.h>

const char* encodeGPU(int xComponents, int yComponents, int width, int height, uint8_t* rgb, size_t bytesPerRow);

#endif // !__BLURHASH_ENCODE_GPU_H_
