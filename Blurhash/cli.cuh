#ifndef __BLURHASH_CLI_H__
#define __BLURHASH_CLI_H__

#include <string>

#define S_ENCODE "encode"
#define S_DECODE "decode"
#define S_CPU "cpu"
#define S_GPU "gpu"
#define S_BOTH "both"
#define S_USAGE "usage:\tblurhash encode {CPU|GPU|BOTH} xComponents yComponents imageFile\n\tblurhash decode {CPU|GPU|BOTH} hash width height imageFile [punch]"

enum Action {
	ENCODE,
	DECODE
};

enum Processor {
	CPU,
	GPU,
	BOTH
};

struct EncodingData {
	int xComponents;
	int yComponents;
};

struct DecodingData {
	char* hash;
	int width;
	int height;
	int punch;
};

union ProcessingData {
	struct EncodingData EncodingData;
	struct DecodingData DecodingData;
};

struct ProgramData {
	enum Action action;
	enum Processor processor;
	char* imagePath;
	union ProcessingData processingData;
};

int parseProgramArgs(int argc, char** argv, struct ProgramData* programdata);
char* prependFilename(const char* prefix, const char* filename);

#endif // !__BLURHASH_CLI_H__
