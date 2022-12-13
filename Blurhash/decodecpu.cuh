#ifndef __BLURHASH_DECODE_CPU_H__
#define __BLURHASH_DECODE_CPU_H__

#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

uint8_t * decodeCPU(const char * blurhash, int width, int height, int punch, int nChannels);

#endif
