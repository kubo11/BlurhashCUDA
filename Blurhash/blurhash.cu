#include "encode.cuh"
#include "decode.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.cuh"
#include "stb_writer.cuh"

#include "cli.cuh"
#include "common.cuh"
#include "cuda_runtime.h"
#include <stdio.h>
#include <string>

#define CUDA_MEASURE_TIME_START() do {                                           \
cudaEvent_t start, stop;                                                         \
CHECK_CUDA(cudaEventCreate(&start))                                              \
CHECK_CUDA(cudaEventCreate(&stop))                                               \
CHECK_CUDA(cudaEventRecord(start))                                               \

#define CUDA_MEASURE_TIME_END(name)                                              \
CHECK_CUDA(cudaEventRecord(stop))                                                \
CHECK_CUDA(cudaEventSynchronize(stop))                                           \
float time;                                                                      \
CHECK_CUDA(cudaEventElapsedTime(&time, start, stop))                             \
printf("%s time elapsed:\n%f s\n", name, time / 1000);                           \
} while (0);

#if defined(_WIN32)

#include <windows.h>

#define MEASURE_TIME_START() do {                                                \
LARGE_INTEGER frequency, start, end;                                             \
double interval;                                                                 \
QueryPerformanceFrequency(&frequency);                                           \
QueryPerformanceCounter(&start);                                                 \

#define MEASURE_TIME_END(name)                                                   \
QueryPerformanceCounter(&end);                                                   \
interval = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;         \
printf("%s time elapsed:\n%f s\n", name, interval);                              \
} while(0);

#elif defined(__unix)

#include <sys/time.h>

#define MEASURE_TIME_START() do {                                                \
struct timeval start, end;                                                       \
double interval;                                                                 \
gettimeofday(NULL, &start);                                                      \

#define MEASURE_TIME_END(name)                                                   \
gettimeofday(NULL, &end);                                                        \
interval = (double)(end.tv_usec - start.tv_usec) / 1000000.0;                    \
printf("%s time elapsed:\n%f s\n", name, interval);                              \
} while(0);

#else

#define MEASURE_TIME_START() do {                                                \
printf("Time measurement in not supported on this system.\n");                   \

#define MEASURE_TIME_END(name)                                                   \
} while(0);

#endif

int main(int argc, char** argv) {
	ProgramData programData;
	if (parseProgramArgs(argc, argv, &programData)) return EXIT_FAILURE;

    if (programData.action == ENCODE) {
        int width, height, channels;
        unsigned char* data = stbi_load(programData.imagePath, &width, &height, &channels, 3);
        if (!data) {
            fprintf(stderr, "Failed to read PNG file %s\n", programData.imagePath);
            return EXIT_FAILURE;
        }
        if (programData.processor == CPU) {
            const char* hash;
            MEASURE_TIME_START()
            hash = encodeCPU(programData.processingData.EncodingData.xComponents,
                             programData.processingData.EncodingData.yComponents,
                             width, height, data, width * 3);
            MEASURE_TIME_END("CPU encoding")
            printf("CPU blurhash:\n%s\n", hash);
        }
        else if (programData.processor == GPU) {
            CHECK_CUDA(cudaSetDevice(0));
            const char* hash;
            CUDA_MEASURE_TIME_START()
            hash = encodeGPU(programData.processingData.EncodingData.xComponents,
                             programData.processingData.EncodingData.yComponents,
                             width, height, data, width * 3);
            CUDA_MEASURE_TIME_END("GPU encoding")
            printf("GPU blurhash:\n%s\n", hash);
            CHECK_CUDA(cudaDeviceReset());
        }
        else {
            const char* hash;
            MEASURE_TIME_START()
            hash = encodeCPU(programData.processingData.EncodingData.xComponents,
                             programData.processingData.EncodingData.yComponents,
                             width, height, data, width * 3);
            MEASURE_TIME_END("CPU encoding")
            printf("CPU blurhash:\n%s\n", hash);
            CHECK_CUDA(cudaSetDevice(0));
            CUDA_MEASURE_TIME_START()
            hash = encodeGPU(programData.processingData.EncodingData.xComponents,
                             programData.processingData.EncodingData.yComponents,
                             width, height, data, width * 3);
            CUDA_MEASURE_TIME_END("GPU encoding")
            printf("GPU blurhash:\n%s\n", hash);
            CHECK_CUDA(cudaDeviceReset());
        }
        stbi_image_free(data);
    }
    else {
        const int nChannels = 4;
        if (programData.processor == CPU) {
            uint8_t* bytes;
            MEASURE_TIME_START()
            bytes = decodeCPU(programData.processingData.DecodingData.hash,
                              programData.processingData.DecodingData.width,
                              programData.processingData.DecodingData.height,
                              programData.processingData.DecodingData.punch,
                              nChannels);
            MEASURE_TIME_END("CPU decoding")
            
            if (!bytes) {
            	fprintf(stderr, "%s is not a valid blurhash, decoding failed.\n", programData.processingData.DecodingData.hash);
            	return EXIT_FAILURE;
            }
            
            if (stbi_write_png(programData.imagePath,
                               programData.processingData.DecodingData.width,
                               programData.processingData.DecodingData.height,
                               nChannels,
                               bytes,
                               nChannels * programData.processingData.DecodingData.width) == 0) {
            	fprintf(stderr, "Failed to write PNG file %s\n", programData.imagePath);
            	return EXIT_FAILURE;
            }
            
            if (bytes) free(bytes);
            
            fprintf(stdout, "Decoded blurhash on CPU successfully, wrote PNG file %s\n", programData.imagePath);
        }
        else if (programData.processor == GPU) {
            uint8_t* bytes;
            CUDA_MEASURE_TIME_START()
            bytes = decodeGPU(programData.processingData.DecodingData.hash,
                programData.processingData.DecodingData.width,
                programData.processingData.DecodingData.height,
                programData.processingData.DecodingData.punch,
                nChannels);
            CUDA_MEASURE_TIME_END("GPU decoding")

            if (!bytes) {
                fprintf(stderr, "%s is not a valid blurhash, decoding failed.\n", programData.processingData.DecodingData.hash);
                return EXIT_FAILURE;
            }

            if (stbi_write_png(programData.imagePath,
                programData.processingData.DecodingData.width,
                programData.processingData.DecodingData.height,
                nChannels,
                bytes,
                nChannels * programData.processingData.DecodingData.width) == 0) {
                fprintf(stderr, "Failed to write PNG file %s\n", programData.imagePath);
                return EXIT_FAILURE;
            }

            if (bytes) free(bytes);

            fprintf(stdout, "Decoded blurhash on GPU successfully, wrote PNG file %s\n", programData.imagePath);
        }
        else {
            uint8_t* bytes;
            MEASURE_TIME_START()
            bytes = decodeCPU(programData.processingData.DecodingData.hash,
                programData.processingData.DecodingData.width,
                programData.processingData.DecodingData.height,
                programData.processingData.DecodingData.punch,
                nChannels);
            MEASURE_TIME_END("CPU decoding")

            if (!bytes) {
                fprintf(stderr, "%s is not a valid blurhash, decoding failed.\n", programData.processingData.DecodingData.hash);
                return EXIT_FAILURE;
            }

            char* filename = prependFilename("cpu_", programData.imagePath);
            if (stbi_write_png(filename,
                programData.processingData.DecodingData.width,
                programData.processingData.DecodingData.height,
                nChannels,
                bytes,
                nChannels * programData.processingData.DecodingData.width) == 0) {
                fprintf(stderr, "Failed to write PNG file %s\n", filename);
                free(filename);
                return EXIT_FAILURE;
            }

            if (bytes) free(bytes);

            fprintf(stdout, "Decoded blurhash on CPU successfully, wrote PNG file %s\n", filename);
            free(filename);

            CUDA_MEASURE_TIME_START()
            bytes = decodeGPU(programData.processingData.DecodingData.hash,
                programData.processingData.DecodingData.width,
                programData.processingData.DecodingData.height,
                programData.processingData.DecodingData.punch,
                nChannels);
            CUDA_MEASURE_TIME_END("GPU decoding")

            if (!bytes) {
                fprintf(stderr, "%s is not a valid blurhash, decoding failed.\n", programData.processingData.DecodingData.hash);
                return EXIT_FAILURE;
            }

            filename = prependFilename("gpu_", programData.imagePath);
            if (stbi_write_png(filename,
                programData.processingData.DecodingData.width,
                programData.processingData.DecodingData.height,
                nChannels,
                bytes,
                nChannels * programData.processingData.DecodingData.width) == 0) {
                fprintf(stderr, "Failed to write PNG file %s\n", filename);
                return EXIT_FAILURE;
            }

            if (bytes) free(bytes);

            fprintf(stdout, "Decoded blurhash on GPU successfully, wrote PNG file %s\n", filename);
            free(filename);
        }
    }

	return EXIT_SUCCESS;
}
