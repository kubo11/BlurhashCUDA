#include "cli.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int parseProgramArgs(int argc, char** argv, struct ProgramData *programData) {
	if (argc < 2) {
		printf("%s", S_USAGE);
		return EXIT_FAILURE;
	}
	
	if (memcmp(argv[1], S_ENCODE, strlen(S_ENCODE)) == 0) {
		programData->action = ENCODE;
	}
	else if (memcmp(argv[1], S_DECODE, strlen(S_DECODE) == 0)) {
		programData->action = DECODE;
	}
	else {
		printf("%s", S_USAGE);
		return EXIT_FAILURE;
	}

	if (programData->action == ENCODE && argc != 6 ||
		programData->action == DECODE && (argc < 7 || argc > 8)) {
		printf("%s", S_USAGE);
		return EXIT_FAILURE;
	}

	if (memcmp(argv[2], S_CPU, strlen(S_CPU)) == 0) {
		programData->processor = CPU;
	}
	else if (memcmp(argv[2], S_GPU, strlen(S_GPU)) == 0) {
		programData->processor = GPU;
	}
	else if (memcmp(argv[2], S_BOTH, strlen(S_BOTH)) == 0) {
		programData->processor = BOTH;
	}
	else {
		printf("%s", S_USAGE);
		return EXIT_FAILURE;
	}

	if (programData->action == ENCODE) {
		programData->processingData.EncodingData.xComponents = atoi(argv[3]);
		if (programData->processingData.EncodingData.xComponents < 1 ||
			programData->processingData.EncodingData.xComponents > 8) {
			printf("%s", S_USAGE);
			printf("1 < xComponents < 8\n");
			return EXIT_FAILURE;
		}
		programData->processingData.EncodingData.yComponents = atoi(argv[4]);
		if (programData->processingData.EncodingData.yComponents < 1 ||
			programData->processingData.EncodingData.yComponents > 8) {
			printf("%s", S_USAGE);
			printf("1 < yComponents < 8\n");
			return EXIT_FAILURE;
		}
		programData->imagePath = argv[5];
	}
	else {
		programData->processingData.DecodingData.hash = argv[3];
		programData->processingData.DecodingData.width = atoi(argv[4]);
		if (programData->processingData.DecodingData.width < 32 ||
			programData->processingData.DecodingData.width > 1024) {
			printf("%s", S_USAGE);
			printf("32 < width < 1024\n");
			return EXIT_FAILURE;
		}
		programData->processingData.DecodingData.height = atoi(argv[5]);
		if (programData->processingData.DecodingData.height < 32 ||
			programData->processingData.DecodingData.height > 1024) {
			printf("%s", S_USAGE);
			printf("32 < height < 1024\n");
			return EXIT_FAILURE;
		}
		programData->imagePath = argv[6];
		if (argc == 8) {
			programData->processingData.DecodingData.punch = atoi(argv[7]);
		}
		else {
			programData->processingData.DecodingData.punch = 1;
		}
	}

	return EXIT_SUCCESS;
}